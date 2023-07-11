use glam::{vec3, EulerRot, Mat3, Mat4, Quat, Vec3};
use nalgebra::{DMatrix, MatrixXx1};
use std::cell::RefCell;
use std::collections::HashMap;
use std::f32::consts::PI;

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

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum IKGoalKind {
    Position(Vec3),

    // TODO: Should probably be a Quat
    Rotation(Mat3),

    // Orientation goal that only cares about orientation around the world Y axis but leaves the
    // other axes free.
    RotY(f32),
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct IKGoal {
    pub end_effector_id: usize,
    pub kind: IKGoalKind,
}

fn is_parallel(a: Vec3, b: Vec3) -> bool {
    fuzzy_compare_f32(a.normalize().dot(b.normalize()).abs(), 1.0)
}

fn get_rotation_axis(to_e: Vec3, target_direction: Vec3) -> Vec3 {
    let raw_axis = if !is_parallel(to_e, target_direction) {
        target_direction.cross(to_e)
    } else if !is_parallel(Vec3::Y, target_direction) {
        target_direction.cross(Vec3::Y)
    } else {
        target_direction.cross(Vec3::Z)
    };

    // if we are very close to the end effector, then there's no more useful rotation axis.
    // Returning ZERO will cause the influence to be 0. This also prevents normalize from being NaN.
    if raw_axis == Vec3::ZERO {
        Vec3::ZERO
    } else {
        raw_axis.normalize()
    }
}

fn get_num_effector_components(goals: &[IKGoal]) -> usize {
    goals
        .iter()
        .map(|g| match g {
            IKGoal {
                kind: IKGoalKind::Position(_),
                ..
            } => 3,
            IKGoal {
                kind: IKGoalKind::Rotation(_),
                ..
            } => 3,
            IKGoal {
                kind: IKGoalKind::RotY(_),
                ..
            } => 1,
        })
        .sum()
}

struct IterableVec3(Vec3);

impl IntoIterator for IterableVec3 {
    type Item = f32;
    type IntoIter = std::array::IntoIter<f32, 3>;

    fn into_iter(self) -> Self::IntoIter {
        [self.0.x, self.0.y, self.0.z].into_iter()
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

fn ang_diff(a: f32, b: f32) -> f32 {
    let delta = b - a;
    (delta + PI) % (2.0 * PI) - PI
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
        let origin_of_rotation = skeleton.current_pose(joint.joint_id).translation();

        // Quaternion joints are a special case because they allow infinite number of rotation axes.
        for (goal_idx, goal) in goals.iter().enumerate() {
            let dof = &mut itd.dof_data[dof_idx];
            dof_idx += 1;
            let influences = &mut dof.influences;

            let mut inf_idx = 0;

            for g2_idx in 0..goals.len() {
                if g2_idx == goal_idx {
                    let axis = match goal.kind {
                        IKGoalKind::Position(goal_position) => {
                            let end_effector_pos =
                                skeleton.current_pose(goal.end_effector_id).translation();
                            let target_direction = goal_position - end_effector_pos;

                            let to_e = end_effector_pos - origin_of_rotation;
                            let axis_of_rotation = if let Some(axis) = joint.restrict_rotation_axis
                            {
                                let joint_rot = skeleton.current_pose(joint.joint_id).rotation();
                                joint_rot * axis
                            } else {
                                get_rotation_axis(to_e, target_direction)
                            };

                            let influence = axis_of_rotation.cross(to_e);

                            influences[inf_idx] = influence.x;
                            influences[inf_idx + 1] = influence.y;
                            influences[inf_idx + 2] = influence.z;
                            inf_idx += 3;

                            axis_of_rotation
                        }
                        IKGoalKind::Rotation(goal_rotation) => {
                            let end_effector_rot =
                                skeleton.current_pose(goal.end_effector_id).rotation();

                            let rotation =
                                Quat::from_mat3(&goal_rotation) * end_effector_rot.inverse();

                            let axis_of_rotation = if let Some(axis) = joint.restrict_rotation_axis
                            {
                                let joint_rot = skeleton.current_pose(joint.joint_id).rotation();
                                joint_rot * axis
                            } else {
                                let (axis_of_rotation, angle) = rotation.to_axis_angle_180();

                                if angle < 0.0 {
                                    -axis_of_rotation
                                } else {
                                    axis_of_rotation
                                }
                            };

                            let influence = axis_of_rotation;
                            influences[inf_idx] = influence.x;
                            influences[inf_idx + 1] = influence.y;
                            influences[inf_idx + 2] = influence.z;
                            inf_idx += 3;

                            axis_of_rotation
                        }
                        IKGoalKind::RotY(_) => {
                            let axis_of_rotation = if let Some(axis) = joint.restrict_rotation_axis
                            {
                                let joint_rot = skeleton.current_pose(joint.joint_id).rotation();
                                joint_rot * axis
                            } else {
                                Vec3::Y
                            };

                            let influence = Quat::from_axis_angle(axis_of_rotation, 1.0)
                                .to_euler(EulerRot::YXZ)
                                .0;

                            influences[inf_idx] = influence;
                            inf_idx += 1;
                            axis_of_rotation
                        }
                    };

                    dof.kind = DoFKind::Quaternion { axis };
                } else {
                    match goals[g2_idx].kind {
                        IKGoalKind::Position(_) => {
                            // TODO: Calculate these values in order to make the IK converge faster
                            // let end_effector_pos = full_skeleton[goals[g2_idx].end_effector_id]
                            //     .transform_point3(Vec3::ZERO);
                            // let to_e = end_effector_pos - origin_of_rotation;
                            // let influence = axis.cross(to_e);
                            // influences.push(influence.x);
                            // influences.push(influence.y);
                            // influences.push(influence.z);
                            influences[inf_idx] = 0.0;
                            influences[inf_idx + 1] = 0.0;
                            influences[inf_idx + 2] = 0.0;
                            inf_idx += 3;
                        }
                        IKGoalKind::Rotation(_) => {
                            // influences.push(axis.x);
                            // influences.push(axis.y);
                            // influences.push(axis.z);
                            influences[inf_idx] = 0.0;
                            influences[inf_idx + 1] = 0.0;
                            influences[inf_idx + 2] = 0.0;
                            inf_idx += 3;
                        }
                        IKGoalKind::RotY(_) => {
                            influences[inf_idx] = 0.0;
                            inf_idx += 1;
                        }
                    }
                }
            }
            influences.iter_mut().for_each(|i| *i /= joint.stiffness);
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

    let jacobian_square = jacobian * &jac_transp;

    let jacobian_square = jacobian_square + DAMPING * DAMPING * damping_ident_matrix;

    let jac_inv = jac_transp * jacobian_square.try_inverse().unwrap(); // TODO: Handle error

    jac_inv
}

pub trait Skeleton {
    fn current_pose(&self, id: usize) -> Mat4;
    fn local_bind_pose(&self, id: usize) -> Mat4;
    fn parent(&self, id: usize) -> Option<usize>;
    fn update_poses(&mut self, poses: &HashMap<usize, Mat4>);
}

// TODO: Use just Quat instead of Mat4 (?) and use local space as much as possible

struct IterationData {
    previous_poses: HashMap<usize, Mat4>,
    final_poses: HashMap<usize, Mat4>,
    raw_joint_xforms: HashMap<usize, Mat4>,
    jacobian: DMatrix<f32>,
    effector_vec: MatrixXx1<f32>,
    dof_data: Box<[DegreeOfFreedom]>,
}

pub struct IKSolver {
    affected_joints: Box<[IKJointControl]>,
    goals: Box<[IKGoal]>,

    // cached immutable data:
    affected_joint_ids: Box<[usize]>,
    num_goal_components: usize,
    num_dof_components: usize,

    // cached mutable data:
    cache: RefCell<IterationData>,
}

impl IKSolver {
    pub fn new(affected_joints: Box<[IKJointControl]>, goals: Box<[IKGoal]>) -> Self {
        let affected_joint_ids = affected_joints.iter().map(|j| j.joint_id).collect();
        let previous_poses =
            HashMap::from_iter(affected_joints.iter().map(|j| (j.joint_id, Mat4::IDENTITY)));
        let final_poses = previous_poses.clone();
        let raw_joint_xforms = previous_poses.clone();
        let num_goal_components = get_num_effector_components(&goals);
        let num_dof_components = get_num_dof_components(&affected_joints, &goals);
        let jacobian = DMatrix::<f32>::zeros(num_goal_components, num_dof_components);
        let effector_vec = MatrixXx1::<f32>::zeros(num_goal_components);
        let dof_data = prebuild_dof_data(&affected_joints, &goals, num_goal_components);

        Self {
            affected_joints,
            goals,
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

    // Important: The joint chain must be in topological order
    pub fn solve<S: Skeleton>(&self, skeleton: &mut S) {
        let itd: &mut IterationData = &mut *self.cache.borrow_mut();

        build_dof_data(itd, skeleton, &self.affected_joints, &self.goals);

        for j in 0..self.num_dof_components {
            for i in 0..self.num_goal_components {
                itd.jacobian[(i, j)] = itd.dof_data[j].influences[i];
            }
        }

        let jac_inv = pseudo_inverse_damped_least_squares(&itd.jacobian, self.num_goal_components);

        let mut p = 0;
        for goal in self.goals.iter() {
            match goal.kind {
                IKGoalKind::Position(goal_position) => {
                    let end_effector_pos =
                        skeleton.current_pose(goal.end_effector_id).translation();
                    let delta = goal_position - end_effector_pos;
                    itd.effector_vec[(p, 0)] = delta.x;
                    itd.effector_vec[(p + 1, 0)] = delta.y;
                    itd.effector_vec[(p + 2, 0)] = delta.z;
                    p += 3;
                }
                IKGoalKind::Rotation(goal_rotation) => {
                    let end_effector_rot = skeleton.current_pose(goal.end_effector_id).rotation();

                    let r = Quat::from_mat3(&goal_rotation) * end_effector_rot.inverse();

                    let (axis_of_rotation, angle) = r.to_axis_angle_180();

                    let scaled_axis = axis_of_rotation * angle;

                    itd.effector_vec[p] = scaled_axis.x;
                    itd.effector_vec[p + 1] = scaled_axis.y;
                    itd.effector_vec[p + 2] = scaled_axis.z;
                    p += 3;
                }
                IKGoalKind::RotY(goal_rot_y) => {
                    let end_effector_rot = skeleton
                        .current_pose(goal.end_effector_id)
                        .rotation()
                        .to_euler(EulerRot::YXZ)
                        .0;
                    let delta = ang_diff(end_effector_rot, goal_rot_y);
                    itd.effector_vec[(p, 0)] = delta;
                    p += 1;
                }
            }
        }

        // Theta is the resulting angles that we want to rotate by
        let theta = &jac_inv * &itd.effector_vec;
        let threshold = THRESHOLD.to_radians();
        let max_angle = theta.amax();
        let beta = threshold / f32::max(max_angle, threshold);

        // Need to remember the original joint transforms
        self.save_previous_poses(itd, skeleton);
        itd.raw_joint_xforms.clone_from(&itd.previous_poses);

        // Our rotation axis is in world space, but during the rotation our position needs to stay fixed.
        for (theta_idx, dof) in itd.dof_data.iter().enumerate() {
            let joint_xform = itd.raw_joint_xforms.get(&dof.joint_id).unwrap();
            let (_, rotation, translation) = joint_xform.to_scale_rotation_translation();

            #[allow(irrefutable_let_patterns)]
            if let DoFKind::Quaternion { axis } = dof.kind {
                let world_rot = Quat::from_axis_angle(axis, beta * theta[theta_idx]);

                let end_rot = world_rot * rotation;

                itd.raw_joint_xforms.insert(
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
                        .get(&parent_id)
                        .copied()
                        .unwrap_or_else(|| skeleton.current_pose(parent_id)),
                    itd.previous_poses
                        .get(&parent_id)
                        .copied()
                        .or_else(|| itd.final_poses.get(&parent_id).copied())
                        .unwrap_or_else(|| skeleton.current_pose(parent_id)),
                ),
                None => (Mat4::IDENTITY, Mat4::IDENTITY),
            };

            let local_rot = (parent_xform_old.inverse()
                * *itd.raw_joint_xforms.get(&dof.joint_id).unwrap())
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

            itd.final_poses.insert(dof.joint_id, world_xform);
        }
        skeleton.update_poses(&itd.final_poses);
    }

    fn save_previous_poses<S: Skeleton>(&self, itd: &mut IterationData, skeleton: &S) {
        for &joint_id in self.affected_joint_ids.iter() {
            itd.previous_poses
                .insert(joint_id, skeleton.current_pose(joint_id));
        }
    }
}

// Retrieve the angle of rotation around the given axis
// https://stackoverflow.com/questions/3684269/component-of-a-quaternion-rotation-around-an-axis
// TODO: Check if this is really what we want!
fn swing_twist_decompose(q: Quat, dir: Vec3) -> f32 {
    let rotation_axis = vec3(q.x, q.y, q.z);
    let dot_prod = dir.dot(rotation_axis);
    let p = dir * dot_prod;
    let mut twist = Quat::from_xyzw(p.x, p.y, p.z, q.w).normalize();

    if dot_prod < 0.0 {
        twist = -twist;
    }

    twist.to_axis_angle_180().1
}

trait ToAxisAngle180 {
    fn to_axis_angle_180(self) -> (Vec3, f32);
}

// Ensures that the angle is between -PI and PI
impl ToAxisAngle180 for Quat {
    fn to_axis_angle_180(self) -> (Vec3, f32) {
        let (axis, angle) = self.to_axis_angle();
        let angle = (angle + PI) % (2.0 * PI) - PI;
        (axis, angle)
    }
}

#[allow(unused)] // TODO
fn fuzzy_compare_vec3(a: Vec3, b: Vec3) -> bool {
    let epsilon = 0.01;
    (a.x - b.x).abs() < epsilon && (a.y - b.y).abs() < epsilon && (a.z - b.z).abs() < epsilon
}

fn fuzzy_compare_f32(a: f32, b: f32) -> bool {
    let epsilon = 0.0001;
    (a - b).abs() < epsilon
}

trait Mat4Helpers {
    fn translation(self) -> Vec3;
    fn rotation(self) -> Quat;
}

impl Mat4Helpers for Mat4 {
    fn translation(self) -> Vec3 {
        self.w_axis.truncate()
    }

    fn rotation(self) -> Quat {
        Quat::from_mat4(&self)
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
