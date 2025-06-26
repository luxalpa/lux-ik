use crate::goals::ik_goal::IKGoalType;
use crate::utils::{ang_diff, Mat4Helpers, SliceWriter};
use crate::{IKJointControl, Skeleton};
use glam::{EulerRot, Quat, Vec3};

pub(crate) struct RotYGoal(pub f32);

impl IKGoalType for RotYGoal {
    fn build_dof_data<S: Skeleton>(
        &self,
        _end_effector_id: usize,
        influence_writer: &mut SliceWriter<f32>,
        skeleton: &S,
        joint: &IKJointControl,
    ) -> Vec3 {
        let axis_of_rotation = if let Some(axis) = joint.restrict_rotation_axis {
            let joint_rot = skeleton.current_pose(joint.joint_id).rotation();
            joint_rot * axis
        } else {
            Vec3::Y
        };

        let influence = Quat::from_axis_angle(axis_of_rotation, 1.0)
            .to_euler(EulerRot::YXZ)
            .0;

        influence_writer.write(influence);
        axis_of_rotation
    }

    fn build_dof_secondary_data(&self, influence_writer: &mut SliceWriter<f32>) {
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
        let end_effector_rot = skeleton
            .current_pose(end_effector_id)
            .rotation()
            .to_euler(EulerRot::YXZ)
            .0;
        let delta = ang_diff(end_effector_rot, self.0);
        effector_vec_writer.write(delta);
    }
}
