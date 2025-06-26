use glam::{Quat, Vec3};

use inner::IKGoalKindInternal;
pub(crate) use inner::IKGoalType;

use super::distance_goal::DistanceGoal;
use super::lookat_goal::LookAtGoal;
use super::rot_y_goal::RotYGoal;
use super::rotation_goal::RotationGoal;

use crate::goals::position_goal::PositionGoal;
use crate::utils::SliceWriter;
use crate::{IKJointControl, LookAtGoalData, Skeleton};

#[derive(Debug, Clone, Copy)]
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
            influence_writer: &mut SliceWriter<f32>,
            skeleton: &S,
            joint: &IKJointControl,
        ) -> Vec3;

        fn build_dof_secondary_data(&self, influence_writer: &mut SliceWriter<f32>);

        fn num_effector_components(&self) -> usize;

        /// The difference that all the Control's combined DOFs need to overcome to move the
        /// end-effector to the goal.
        fn effector_delta<S: Skeleton>(
            &self,
            end_effector_id: usize,
            effector_vec_writer: &mut SliceWriter<f32>,
            skeleton: &S,
        );
    }

    pub(crate) enum IKGoalKindInternal {
        PositionGoal,
        DistanceGoal,
        RotationGoal,
        LookAtGoal,
        RotYGoal,
    }
}

// This is used for the external interface, so we can avoid doing Position(Position(Vec3)) etc.
#[derive(Debug, Clone, Copy)]
pub enum IKGoalKind {
    Position(Vec3),
    Distance(Vec3),
    Rotation(Quat),
    LookAt(LookAtGoalData),
    // Orientation goal that only cares about orientation around the world Y axis but leaves the
    // other axes free.
    RotY(f32),
}

impl IKGoalKind {
    pub(crate) fn as_internal(self) -> IKGoalKindInternal {
        match self {
            IKGoalKind::Position(p) => IKGoalKindInternal::PositionGoal(PositionGoal(p)),
            IKGoalKind::Distance(p) => IKGoalKindInternal::DistanceGoal(DistanceGoal(p)),
            IKGoalKind::Rotation(r) => IKGoalKindInternal::RotationGoal(RotationGoal(r)),
            IKGoalKind::LookAt(p) => IKGoalKindInternal::LookAtGoal(LookAtGoal(p)),
            IKGoalKind::RotY(r) => IKGoalKindInternal::RotYGoal(RotYGoal(r)),
        }
    }
}
