use glam::{Quat, Vec3};

use inner::IKGoalKindInternal;
pub(crate) use inner::IKGoalType;

use super::position_goal::PositionGoal;
use super::rot_y_goal::RotYGoal;
use super::rotation_goal::RotationGoal;

use crate::utils::SlicePusher;
use crate::{IKJointControl, Skeleton};

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct IKGoal {
    pub end_effector_id: usize,
    pub kind: IKGoalKind,
}

#[enum_dispatch::enum_dispatch(IKGoalType for IKGoalKindInternal)]
mod inner {
    use super::*;

    pub(crate) trait IKGoalType {
        fn build_dof_data<S: Skeleton>(
            &self,
            end_effector_id: usize,
            influence_pusher: &mut SlicePusher<f32>,
            skeleton: &S,
            joint: &IKJointControl,
        ) -> Vec3;

        fn build_dof_secondary_data(&self, influence_pusher: &mut SlicePusher<f32>);

        fn num_effector_components(&self) -> usize;

        fn effector_delta<S: Skeleton>(
            &self,
            end_effector_id: usize,
            effector_vec_pusher: &mut SlicePusher<f32>,
            skeleton: &S,
        );
    }

    pub(crate) enum IKGoalKindInternal {
        PositionGoal,
        RotationGoal,
        RotYGoal,
    }
}

// This is used for the external interface, so we can avoid doing Position(Position(Vec3)) etc.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum IKGoalKind {
    Position(Vec3),
    Rotation(Quat),
    // Orientation goal that only cares about orientation around the world Y axis but leaves the
    // other axes free.
    RotY(f32),
}

impl IKGoalKind {
    pub(crate) fn as_internal(self) -> IKGoalKindInternal {
        match self {
            IKGoalKind::Position(p) => IKGoalKindInternal::PositionGoal(PositionGoal(p)),
            IKGoalKind::Rotation(r) => IKGoalKindInternal::RotationGoal(RotationGoal(r)),
            IKGoalKind::RotY(r) => IKGoalKindInternal::RotYGoal(RotYGoal(r)),
        }
    }
}
