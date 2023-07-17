use crate::goals::ik_goal::IKGoalType;
use crate::utils::{get_rotation_axis, Mat4Helpers, SlicePusher};
use crate::{IKJointControl, Skeleton};
use glam::Vec3;

pub struct PositionGoal(pub Vec3);

impl IKGoalType for PositionGoal {
    fn build_dof_data<S: Skeleton>(
        &self,
        end_effector_id: usize,
        influence_pusher: &mut SlicePusher<f32>,
        skeleton: &S,
        joint: &IKJointControl,
    ) -> Vec3 {
        let origin_of_rotation = skeleton.current_pose(joint.joint_id).translation();

        let end_effector_pos = skeleton.current_pose(end_effector_id).translation();
        let target_direction = self.0 - end_effector_pos;

        let to_e = end_effector_pos - origin_of_rotation;
        let axis_of_rotation = if let Some(axis) = joint.restrict_rotation_axis {
            let joint_rot = skeleton.current_pose(joint.joint_id).rotation();
            joint_rot * axis
        } else {
            get_rotation_axis(to_e, target_direction)
        };

        let influence = axis_of_rotation.cross(to_e);
        // influence_pusher.push(influence);
        influence_pusher.push(influence.dot((self.0 - end_effector_pos).normalize()));
        axis_of_rotation
    }

    fn build_dof_secondary_data(&self, influence_pusher: &mut SlicePusher<f32>) {
        // TODO: Maybe calculate these values in order to make the IK converge faster
        // let end_effector_pos = full_skeleton[goals[g2_idx].end_effector_id]
        //     .transform_point3(Vec3::ZERO);
        // let to_e = end_effector_pos - origin_of_rotation;
        // let influence = axis.cross(to_e);
        // influences.push(influence.x);
        // influences.push(influence.y);
        // influences.push(influence.z);
        // influence_pusher.skip::<Vec3>();
        influence_pusher.skip::<f32>();
    }

    fn num_effector_components(&self) -> usize {
        // 3
        1
    }

    fn effector_delta<S: Skeleton>(
        &self,
        end_effector_id: usize,
        effector_vec_pusher: &mut SlicePusher<f32>,
        skeleton: &S,
    ) {
        let end_effector_pos = skeleton.current_pose(end_effector_id).translation();
        // let delta = self.0 - end_effector_pos;
        let delta = (self.0 - end_effector_pos).length();
        effector_vec_pusher.push(delta);
    }
}
