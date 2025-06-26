use crate::goals::ik_goal::IKGoalType;
use crate::utils::{Mat4Helpers, SliceWriter, ToAxisAngle180};
use crate::{IKJointControl, Skeleton};
use glam::{Quat, Vec3};

pub(crate) struct RotationGoal(pub Quat);

impl IKGoalType for RotationGoal {
    fn build_dof_data<S: Skeleton>(
        &self,
        end_effector_id: usize,
        influence_writer: &mut SliceWriter<f32>,
        skeleton: &S,
        joint: &IKJointControl,
    ) -> Vec3 {
        let end_effector_rot = skeleton.current_pose(end_effector_id).rotation();

        let rotation = self.0 * end_effector_rot.inverse();

        let axis_of_rotation = if let Some(axis) = joint.restrict_rotation_axis {
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

        // let influence = axis_of_rotation;
        // TODO: Handle axis constraints
        influence_writer.write(rotation.to_axis_angle_180().1);

        axis_of_rotation
    }

    fn build_dof_secondary_data(&self, influence_writer: &mut SliceWriter<f32>) {
        // influences.push(axis.x);
        // influences.push(axis.y);
        // influences.push(axis.z);
        influence_writer.skip::<f32>();
    }

    fn num_effector_components(&self) -> usize {
        1
    }

    fn effector_delta<S: Skeleton>(
        &self,
        end_effector_id: usize,
        effector_vec_writer: &mut SliceWriter<f32>,
        skeleton: &S,
    ) {
        let end_effector_rot = skeleton.current_pose(end_effector_id).rotation();

        let r = self.0 * end_effector_rot.inverse();

        let (_axis_of_rotation, angle) = r.to_axis_angle_180();

        // let scaled_axis = axis_of_rotation * angle;

        effector_vec_writer.write(angle);
    }
}
