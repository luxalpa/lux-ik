use glam::{vec3, EulerRot, Mat3, Mat4, Quat, Vec3};
use nalgebra::{DMatrix, Dyn, Matrix, VecStorage, U1};
use std::f32::consts::PI;

const DAMPING: f32 = 10.0;
const THRESHOLD: f32 = 10.5;

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct JointLimit {
    pub axis: Vec3,
    pub min: f32,
    pub max: f32,
}

#[derive(Debug, PartialEq, Clone)]
pub struct IKJointControl {
    pub joint_id: usize,

    // if set, limits the rotation axis to this (in joint space)
    pub restrict_rotation_axis: Option<Vec3>,

    pub limits: Option<Vec<JointLimit>>,
    // TODO: Weights
}

impl IKJointControl {
    pub fn new(joint_id: usize) -> Self {
        IKJointControl {
            joint_id,
            restrict_rotation_axis: None,
            limits: None,
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

const USE_QUATERNIONS: bool = true;

fn build_dof_data(
    full_skeleton: &[Mat4],
    affected_joints: &[IKJointControl],
    goals: &[IKGoal],
    use_quaternions: bool,
) -> Vec<DegreeOfFreedom> {
    let mut dof_data: Vec<DegreeOfFreedom> = Vec::new();

    for joint in affected_joints {
        // TODO: Use w_axis.truncate(); instead of transform_point3
        let origin_of_rotation = full_skeleton[joint.joint_id].transform_point3(Vec3::ZERO);

        // Quaternion joints are a special case because they allow infinite number of rotation axes.
        if use_quaternions {
            for (goal_idx, goal) in goals.iter().enumerate() {
                let mut influences: Vec<f32> = Vec::new();

                let (temp_influences, axis) = match goal.kind {
                    IKGoalKind::Position(goal_position) => {
                        let end_effector_pos =
                            full_skeleton[goal.end_effector_id].transform_point3(Vec3::ZERO);
                        let target_direction = goal_position - end_effector_pos;

                        let to_e = end_effector_pos - origin_of_rotation;
                        let axis_of_rotation = if let Some(axis) = joint.restrict_rotation_axis {
                            let joint_rot = Quat::from_mat4(&full_skeleton[joint.joint_id]);
                            joint_rot * axis
                        } else {
                            get_rotation_axis(to_e, target_direction)
                        };

                        let influence = axis_of_rotation.cross(to_e);

                        (
                            vec![influence.x, influence.y, influence.z],
                            axis_of_rotation,
                        )
                    }
                    IKGoalKind::Rotation(goal_rotation) => {
                        let end_effector_rot =
                            Quat::from_mat4(&full_skeleton[goal.end_effector_id]);

                        let rotation = Quat::from_mat3(&goal_rotation) * end_effector_rot.inverse();

                        let axis_of_rotation = if let Some(axis) = joint.restrict_rotation_axis {
                            let joint_rot = Quat::from_mat4(&full_skeleton[joint.joint_id]);
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
                        (
                            vec![influence.x, influence.y, influence.z],
                            axis_of_rotation,
                        )
                    }
                    IKGoalKind::RotY(_) => {
                        let axis_of_rotation = if let Some(axis) = joint.restrict_rotation_axis {
                            let joint_rot = Quat::from_mat4(&full_skeleton[joint.joint_id]);
                            joint_rot * axis
                        } else {
                            Vec3::Y
                        };

                        let influence = Quat::from_axis_angle(axis_of_rotation, 1.0)
                            .to_euler(EulerRot::YXZ)
                            .0;

                        (vec![influence], axis_of_rotation)
                    }
                };

                for g2_idx in 0..goals.len() {
                    if g2_idx == goal_idx {
                        influences.extend(&temp_influences);
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
                                influences.push(0.0);
                                influences.push(0.0);
                                influences.push(0.0);
                            }
                            IKGoalKind::Rotation(_) => {
                                // influences.push(axis.x);
                                // influences.push(axis.y);
                                // influences.push(axis.z);
                                influences.push(0.0);
                                influences.push(0.0);
                                influences.push(0.0);
                            }
                            IKGoalKind::RotY(_) => {
                                influences.push(0.0); // TODO
                            }
                        }
                    }
                }

                // Quaternion joints require us to create new DegreesOfFreedom on demand for each new rotation axis.
                dof_data.push(DegreeOfFreedom {
                    joint_id: joint.joint_id,
                    kind: DoFKind::Quaternion { axis },
                    influences,
                });
            }
        }
    }

    dof_data
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

// Important: The joint chain must be in topological order
pub fn solve(
    full_skeleton: &mut [Mat4],
    local_bind_pose: &[Mat4],
    parents: &[i32],
    affected_joints: &[IKJointControl],
    goals: &[IKGoal],
) {
    let dof_data = build_dof_data(full_skeleton, affected_joints, goals, USE_QUATERNIONS);

    let num_goal_components = get_num_effector_components(goals);
    let num_dof_components = dof_data.len();

    let jacobian = DMatrix::<f32>::from_fn(num_goal_components, num_dof_components, |i, j| {
        dof_data[j].influences[i]
    });

    let jac_inv = pseudo_inverse_damped_least_squares(&jacobian, num_goal_components);

    let effectors: Vec<_> = goals
        .iter()
        .map(|goal| match goal.kind {
            IKGoalKind::Position(goal_position) => {
                let end_effector_pos =
                    full_skeleton[goal.end_effector_id].transform_point3(Vec3::ZERO);
                IterableVec3(goal_position - end_effector_pos)
                    .into_iter()
                    .collect::<Vec<_>>()
            }
            IKGoalKind::Rotation(goal_rotation) => {
                let end_effector_rot = Quat::from_mat4(&full_skeleton[goal.end_effector_id]);

                let r = Quat::from_mat3(&goal_rotation) * end_effector_rot.inverse();

                let (axis_of_rotation, angle) = r.to_axis_angle_180();

                let scaled_axis = axis_of_rotation * angle;

                IterableVec3(scaled_axis).into_iter().collect::<Vec<_>>()
            }
            IKGoalKind::RotY(goal_rot_y) => {
                let end_effector_rot = Quat::from_mat4(&full_skeleton[goal.end_effector_id])
                    .to_euler(EulerRot::YXZ)
                    .0;
                let delta = ang_diff(end_effector_rot, goal_rot_y);
                vec![delta]
            }
        })
        .flatten()
        .collect();

    let effector_vec = Matrix::<f32, Dyn, U1, VecStorage<f32, Dyn, U1>>::from_vec(effectors);

    // Theta is the resulting angles that we want to rotate by
    let theta = &jac_inv * &effector_vec;
    let threshold = THRESHOLD.to_radians();
    let max_angle = theta.amax();
    let beta = threshold / f32::max(max_angle, threshold);

    // Need to remember the original joint transforms
    let previous_skeleton = full_skeleton.to_vec();

    // Our rotation axis is in world space, but during the rotation our position needs to stay fixed.
    let mut raw_joint_xforms = full_skeleton.to_vec();

    for (theta_idx, dof) in dof_data.iter().enumerate() {
        let joint_xform = raw_joint_xforms[dof.joint_id];
        let translation = joint_xform.transform_point3(Vec3::ZERO);
        let rotation = Quat::from_mat4(&joint_xform);

        #[allow(irrefutable_let_patterns)]
        if let DoFKind::Quaternion { axis } = dof.kind {
            let world_rot = Quat::from_axis_angle(axis, beta * theta[theta_idx]);

            let end_rot = world_rot * rotation;

            raw_joint_xforms[dof.joint_id] = Mat4::from_rotation_translation(end_rot, translation);
        }
    }

    // Correct the rotation of the other joints
    for dof in &dof_data {
        let parent_id = parents[dof.joint_id];
        let parent_xform = *full_skeleton
            .get(parent_id as usize)
            .unwrap_or(&Mat4::IDENTITY);
        let parent_xform_old = *previous_skeleton
            .get(parent_id as usize)
            .unwrap_or(&Mat4::IDENTITY);

        let local_rot =
            Quat::from_mat4(&(parent_xform_old.inverse() * raw_joint_xforms[dof.joint_id]))
                .normalize();

        let local_translation = local_bind_pose[dof.joint_id].transform_point3(Vec3::ZERO);
        let mut world_xform =
            parent_xform * Mat4::from_rotation_translation(local_rot, local_translation);

        let joint_cfg = affected_joints
            .iter()
            .find(|joint| joint.joint_id == dof.joint_id)
            .unwrap();

        if let Some(limits) = &joint_cfg.limits {
            let local_xform = Quat::from_mat4(&(parent_xform.inverse() * world_xform));
            let local_bind_xform = Quat::from_mat4(&(local_bind_pose[dof.joint_id]));

            // limits are relative to the local bind pose of the joints
            let local_change = local_bind_xform.inverse() * local_xform;
            for limit in limits.iter() {
                let custom_axis = local_change * limit.axis;
                let angle = swing_twist_decompose(local_change, custom_axis);

                if angle < limit.min {
                    world_xform = world_xform
                        * Mat4::from_quat(Quat::from_axis_angle(custom_axis, limit.min - angle));
                } else if angle > limit.max {
                    world_xform = world_xform
                        * Mat4::from_quat(Quat::from_axis_angle(custom_axis, limit.max - angle));
                }
            }
        }

        full_skeleton[dof.joint_id] = world_xform;
    }
}

// Retrieve the angle of rotation around the given axis
// https://stackoverflow.com/questions/3684269/component-of-a-quaternion-rotation-around-an-axis
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

#[cfg(test)]
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
}
