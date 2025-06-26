use std::cell::RefCell;

use crate::goals::ik_goal::IKGoal;
use crate::goals::{IKGoalKind, IKGoalType};
use crate::LookAtGoalData;
use glam::{Mat4, Quat, Vec3};
use nalgebra::{DMatrix, MatrixXx1};

use crate::utils::{swing_twist_decompose, JointMap, Mat4Helpers, SliceWriter};

const DAMPING: f32 = 3.0;
const THRESHOLD: f32 = 10.5;

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct JointLimit {
    pub axis: Vec3,
    pub min: f32,
    pub max: f32,
}

// #[derive(Debug, PartialEq, Clone)]
pub struct IKJointControl {
    pub joint_id: usize,

    // if set, limits the rotation axis to this (in joint space)
    pub restrict_rotation_axis: Option<Vec3>,

    pub limits: Option<Vec<JointLimit>>,

    pub stiffness: f32,
}

impl IKJointControl {
    pub fn new(joint_id: usize) -> Self {
        IKJointControl {
            joint_id,
            restrict_rotation_axis: None,
            limits: None,
            stiffness: 1.0,
        }
    }

    pub fn with_axis_constraint(self, restrict_rotation_axis: Vec3) -> Self {
        IKJointControl {
            restrict_rotation_axis: Some(restrict_rotation_axis),
            ..self
        }
    }

    pub fn with_limits(self, limits: &[JointLimit]) -> Self {
        IKJointControl {
            limits: Some(limits.to_vec()),
            ..self
        }
    }

    pub fn with_stiffness(self, stiffness: f32) -> Self {
        IKJointControl { stiffness, ..self }
    }
}

// the different methods of joint movement
enum DoFKind {
    Quaternion { axis: Vec3 },
}

struct DegreeOfFreedom {
    joint_id: usize,
    kind: DoFKind,
    influences: Vec<f32>, // one per end effector
}

fn get_num_dof_components(affected_joints: &[IKJointControl], goals: &[IKGoal]) -> usize {
    affected_joints.len() * goals.len()
}

fn build_dof_data<S: Skeleton>(
    itd: &mut IterationData,
    skeleton: &S,
    affected_joints: &[IKJointControl],
    goals: &[IKGoal],
) {
    let mut dof_idx = 0;

    for joint in affected_joints {
        // Quaternion joints are a special case because they allow infinite number of rotation axes.
        for (goal_idx, goal) in goals.iter().enumerate() {
            let dof = &mut itd.dof_data[dof_idx];
            dof_idx += 1;
            let mut influence_writer = SliceWriter::new(&mut dof.influences);

            for g2_idx in 0..goals.len() {
                if g2_idx == goal_idx {
                    let axis = goal.kind.as_internal().build_dof_data(
                        goal.end_effector_id,
                        &mut influence_writer,
                        skeleton,
                        joint,
                    );
                    dof.kind = DoFKind::Quaternion { axis };
                } else {
                    goals[g2_idx]
                        .kind
                        .as_internal()
                        .build_dof_secondary_data(&mut influence_writer);
                }
            }
            dof.influences
                .iter_mut()
                .for_each(|i| *i /= joint.stiffness);
        }
    }
}

fn prebuild_dof_data(
    affected_joints: &[IKJointControl],
    goals: &[IKGoal],
    num_goal_components: usize,
) -> Box<[DegreeOfFreedom]> {
    let mut dof_data: Vec<DegreeOfFreedom> = Vec::new();

    for joint in affected_joints {
        for _ in goals.iter() {
            dof_data.push(DegreeOfFreedom {
                joint_id: joint.joint_id,
                kind: DoFKind::Quaternion { axis: Vec3::Y },
                influences: vec![0.0; num_goal_components],
            });
        }
    }

    dof_data.into_boxed_slice()
}

fn pseudo_inverse_damped_least_squares(
    jacobian: &DMatrix<f32>,
    num_effectors: usize,
) -> DMatrix<f32> {
    let jac_transp = jacobian.transpose();

    let damping_ident_matrix = nalgebra::DMatrix::<f32>::identity(num_effectors, num_effectors);

    // TODO: Avoid reallocation
    let jacobian_square = jacobian * &jac_transp;

    let jacobian_square = jacobian_square + DAMPING * DAMPING * damping_ident_matrix;

    let jac_inv = jac_transp * jacobian_square.try_inverse().unwrap(); // TODO: Handle error

    jac_inv
}

pub trait Skeleton {
    fn current_pose(&self, id: usize) -> Mat4;
    fn local_bind_pose(&self, id: usize) -> Mat4;
    fn parent(&self, id: usize) -> Option<usize>;
    fn update_poses(&mut self, poses: &JointMap);
}

// TODO: [Optimization] Use just Quat instead of Mat4 (?) and use local space as much as possible

struct IterationData {
    previous_poses: JointMap,
    final_poses: JointMap,
    raw_joint_xforms: JointMap,
    jacobian: DMatrix<f32>,
    effector_vec: MatrixXx1<f32>,
    dof_data: Box<[DegreeOfFreedom]>,
}

pub struct IKSolver {
    affected_joints: Box<[IKJointControl]>,
    goals: Box<[IKGoal]>,
    num_iterations: usize,

    // cached immutable data:
    affected_joint_ids: Box<[usize]>,
    num_goal_components: usize,
    num_dof_components: usize,

    // cached mutable data:
    cache: RefCell<IterationData>,
}

impl IKSolver {
    pub fn new(
        affected_joints: Box<[IKJointControl]>,
        goals: Box<[IKGoal]>,
        num_iterations: usize,
    ) -> Self {
        let affected_joint_ids = affected_joints.iter().map(|j| j.joint_id).collect();
        let previous_poses = JointMap::new(
            affected_joints.iter().map(|j| j.joint_id).collect(),
            [Mat4::IDENTITY]
                .repeat(affected_joints.len())
                .into_boxed_slice(),
        );
        let final_poses = previous_poses.clone();
        let raw_joint_xforms = previous_poses.clone();
        let num_goal_components = goals
            .iter()
            .map(|g| g.kind.as_internal().num_effector_components())
            .sum();
        let num_dof_components = get_num_dof_components(&affected_joints, &goals);
        let jacobian = DMatrix::<f32>::zeros(num_goal_components, num_dof_components);
        let effector_vec = MatrixXx1::<f32>::zeros(num_goal_components);
        let dof_data = prebuild_dof_data(&affected_joints, &goals, num_goal_components);

        Self {
            affected_joints,
            goals,
            num_iterations,
            affected_joint_ids,
            num_goal_components,
            num_dof_components,
            cache: RefCell::new(IterationData {
                previous_poses,
                final_poses,
                raw_joint_xforms,
                jacobian,
                effector_vec,
                dof_data,
            }),
        }
    }

    pub fn position_goals_iter_mut(&mut self) -> impl Iterator<Item = (usize, &mut Vec3)> {
        self.goals.iter_mut().filter_map(|g| match g.kind {
            IKGoalKind::Position(ref mut p) => Some((g.end_effector_id, p)),
            _ => None,
        })
    }

    pub fn rotation_goals_iter_mut(&mut self) -> impl Iterator<Item = (usize, &mut Quat)> {
        self.goals.iter_mut().filter_map(|g| match g.kind {
            IKGoalKind::Rotation(ref mut o) => Some((g.end_effector_id, o)),
            _ => None,
        })
    }

    pub fn lookat_goals_iter_mut(&mut self) -> impl Iterator<Item = (usize, &mut LookAtGoalData)> {
        self.goals.iter_mut().filter_map(|g| match g.kind {
            IKGoalKind::LookAt(ref mut l) => Some((g.end_effector_id, l)),
            _ => None,
        })
    }

    // Important: The joint chain must be in topological order
    pub fn solve<S: Skeleton>(&self, skeleton: &mut S) {
        let itd: &mut IterationData = &mut *self.cache.borrow_mut();

        for _ in 0..self.num_iterations {
            build_dof_data(itd, skeleton, &self.affected_joints, &self.goals);

            for j in 0..self.num_dof_components {
                for i in 0..self.num_goal_components {
                    itd.jacobian[(i, j)] = itd.dof_data[j].influences[i];
                }
            }

            let jac_inv =
                pseudo_inverse_damped_least_squares(&itd.jacobian, self.num_goal_components);

            let mut effector_vec_writer = SliceWriter::new(itd.effector_vec.as_mut_slice());
            for goal in self.goals.iter() {
                goal.kind.as_internal().effector_delta(
                    goal.end_effector_id,
                    &mut effector_vec_writer,
                    skeleton,
                );
            }

            // Theta is the resulting angles that we want to rotate by
            let theta = &jac_inv * &itd.effector_vec;
            let threshold = THRESHOLD.to_radians();
            let max_angle = theta.amax();
            let beta = threshold / f32::max(max_angle, threshold);

            // Need to remember the original joint transforms
            self.save_previous_poses(itd, skeleton);
            itd.raw_joint_xforms.set_data_from(&itd.previous_poses);

            // Our rotation axis is in world space, but during the rotation our position needs to stay fixed.
            for (theta_idx, dof) in itd.dof_data.iter().enumerate() {
                let joint_xform = itd.raw_joint_xforms.get(dof.joint_id).unwrap();
                let rotation = joint_xform.rotation();
                let translation = joint_xform.translation();

                #[allow(irrefutable_let_patterns)]
                if let DoFKind::Quaternion { axis } = dof.kind {
                    let world_rot = Quat::from_axis_angle(axis, beta * theta[theta_idx]);

                    let end_rot = world_rot * rotation;

                    itd.raw_joint_xforms.set(
                        dof.joint_id,
                        Mat4::from_rotation_translation(end_rot, translation),
                    );
                }
            }

            // Correct the rotation of the other joints
            // TODO: This should probably only iterate over each joint once, but DofData has multiple entries per joint!
            for dof in itd.dof_data.iter() {
                let parent_id = skeleton.parent(dof.joint_id);
                let (parent_xform, parent_xform_old) = match parent_id {
                    Some(parent_id) => (
                        itd.final_poses
                            .get(parent_id)
                            .unwrap_or_else(|| skeleton.current_pose(parent_id)),
                        itd.previous_poses
                            .get(parent_id)
                            .or_else(|| itd.final_poses.get(parent_id))
                            .unwrap_or_else(|| skeleton.current_pose(parent_id)),
                    ),
                    None => (Mat4::IDENTITY, Mat4::IDENTITY),
                };

                let local_rot = (parent_xform_old.inverse()
                    * itd.raw_joint_xforms.get(dof.joint_id).unwrap())
                .rotation()
                .normalize();

                let local_translation = skeleton.local_bind_pose(dof.joint_id).translation();

                // TODO: Use local transforms in this loop
                let mut world_xform =
                    parent_xform * Mat4::from_rotation_translation(local_rot, local_translation);

                let joint_cfg = self
                    .affected_joints
                    .iter()
                    .find(|joint| joint.joint_id == dof.joint_id)
                    .unwrap();

                if let Some(limits) = &joint_cfg.limits {
                    let local_xform = (parent_xform.inverse() * world_xform).rotation();
                    let local_bind_xform = skeleton.local_bind_pose(dof.joint_id).rotation();

                    // limits are relative to the local bind pose of the joints
                    let local_change = local_bind_xform.inverse() * local_xform;
                    for limit in limits.iter() {
                        let custom_axis = local_change * limit.axis;
                        let angle = swing_twist_decompose(local_change, custom_axis);

                        if angle < limit.min {
                            world_xform = world_xform
                                * Mat4::from_quat(Quat::from_axis_angle(
                                    custom_axis,
                                    limit.min - angle,
                                ));
                        } else if angle > limit.max {
                            world_xform = world_xform
                                * Mat4::from_quat(Quat::from_axis_angle(
                                    custom_axis,
                                    limit.max - angle,
                                ));
                        }
                    }
                }

                itd.final_poses.set(dof.joint_id, world_xform);
            }
            skeleton.update_poses(&itd.final_poses);
        }
    }

    fn save_previous_poses<S: Skeleton>(&self, itd: &mut IterationData, skeleton: &S) {
        itd.previous_poses.set_all(
            self.affected_joint_ids
                .iter()
                .map(|&joint_id| skeleton.current_pose(joint_id)),
        );
    }
}

/*#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_orient() {
        let mut skeleton = [
            Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0)),
            Mat4::from_translation(Vec3::new(0.0, 0.0, 1.0)),
            Mat4::from_translation(Vec3::new(0.0, 0.0, 2.0)),
        ];
        let bind_pose = skeleton.to_vec();
        let parents = [-1, 0, 1];
        let affected_joints = [
            IKJointControl::new(0),
            IKJointControl::new(1),
            IKJointControl::new(2),
        ];

        let expected_rot_mat = Mat3::from_axis_angle(Vec3::Y, 45f32.to_radians());

        let goals = [IKGoal {
            end_effector_id: 2,
            kind: IKGoalKind::Rotation(expected_rot_mat),
        }];

        for _ in 0..200 {
            solve(
                &mut skeleton,
                &bind_pose,
                &parents,
                &affected_joints,
                &goals,
                None,
            );
        }

        let expected_rot = Quat::from_mat3(&expected_rot_mat);
        let actual_rot = Quat::from_mat4(&skeleton[2]);

        assert!(
            fuzzy_compare_vec3(expected_rot.to_scaled_axis(), actual_rot.to_scaled_axis()),
            "Expected: {:?}, Actual: {:?}",
            expected_rot.to_axis_angle(),
            actual_rot.to_axis_angle()
        );
    }
}*/
