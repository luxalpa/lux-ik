use crate::goals::ik_goal::IKGoalType;
use crate::utils::{Mat4Helpers, SlicePusher, ToAxisAngle180};
use crate::{IKJointControl, Skeleton};
use glam::{Mat4, Quat, Vec3};

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct LookAtGoalData {
    pub target: Vec3,
    pub local_lookat_axis: Vec3,
}
pub struct LookAtGoal(pub LookAtGoalData);

impl IKGoalType for LookAtGoal {
    fn build_dof_data<S: Skeleton>(
        &self,
        end_effector_id: usize,
        influence_pusher: &mut SlicePusher<f32>,
        skeleton: &S,
        joint: &IKJointControl,
    ) -> Vec3 {
        let origin_of_rotation = skeleton.current_pose(joint.joint_id).translation();

        let (axis, angle) = lookat(
            origin_of_rotation,
            skeleton.current_pose(end_effector_id),
            self.0.target,
            self.0.local_lookat_axis,
        );

        influence_pusher.push(angle);

        axis
    }

    fn build_dof_secondary_data(&self, influence_pusher: &mut SlicePusher<f32>) {
        influence_pusher.skip::<f32>();
    }

    fn num_effector_components(&self) -> usize {
        1
    }

    fn effector_delta<S: Skeleton>(
        &self,
        end_effector_id: usize,
        effector_vec_pusher: &mut SlicePusher<f32>,
        skeleton: &S,
    ) {
        let end_effector = skeleton.current_pose(end_effector_id);

        let (_, angle) = lookat(
            end_effector.translation(),
            end_effector,
            self.0.target,
            self.0.local_lookat_axis,
        );

        effector_vec_pusher.push(angle);
    }
}

// makes the Z-Axis on the end_effector look at target
fn lookat(origin: Vec3, end_effector: Mat4, target: Vec3, local_lookat_axis: Vec3) -> (Vec3, f32) {
    let r = intersect_ray_sphere(
        end_effector.translation(),
        end_effector.transform_vector3(local_lookat_axis),
        origin,
        (target - origin).length(),
    );

    let to_target = (target - origin).normalize();

    match r {
        Intersection::None => return (Vec3::ZERO, 0.0),
        Intersection::One(r) => {
            let to_r = (r - origin).normalize();
            Quat::from_rotation_arc(to_r, to_target).to_axis_angle_180()
        }
        Intersection::Two(r1, r2) => {
            let to_r1 = (r1 - origin).normalize();
            let to_r2 = (r2 - origin).normalize();
            let q1 = Quat::from_rotation_arc(to_r1, to_target).to_axis_angle_180();
            let q2 = Quat::from_rotation_arc(to_r2, to_target).to_axis_angle_180();
            if q1.1 < q2.1 {
                q1
            } else {
                q2
            }
        }
    }
}

enum Intersection {
    None,
    One(Vec3),
    Two(Vec3, Vec3),
}

fn intersect_ray_sphere(
    ray_origin: Vec3,
    ray_direction: Vec3,
    sphere_center: Vec3,
    sphere_radius: f32,
) -> Intersection {
    let o_minus_c = ray_origin - sphere_center;
    let p = ray_direction.dot(o_minus_c);
    let q = o_minus_c.dot(o_minus_c) - sphere_radius * sphere_radius;

    if q > 0.0 && p > 0.0 {
        return Intersection::None;
    }

    let discr = p * p - q;
    if discr < 0.0 {
        return Intersection::None;
    }

    let dist1 = -p - discr.sqrt();
    let dist2 = -p + discr.sqrt();

    let dist = dist1.max(dist2);

    if dist < 0.0 {
        return Intersection::None;
    }

    if dist1 > 0.0 && dist2 > 0.0 {
        let hit_point1 = ray_origin + ray_direction * dist1;
        let hit_point2 = ray_origin + ray_direction * dist2;
        return Intersection::Two(hit_point1, hit_point2);
    }

    let hit_point = ray_origin + ray_direction * dist;
    Intersection::One(hit_point)
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use glam::{vec3, Mat4};
//     use houdini_debug_logger::{houlog, init_houlog_live, save_houlog};
//
//     #[test]
//     fn test_lookat_goal() -> anyhow::Result<()> {
//         init_houlog_live(None)?;
//
//         let target = vec3(-14.0, 0.0, 0.0);
//         let origin = Mat4::from_translation(vec3(3.0, 0.0, 0.0));
//         let end_effector = Mat4::from_translation(vec3(0.0, 0.0, 7.0))
//             * Mat4::from_rotation_y(0.0f32.to_radians());
//
//         let (axis, angle) = lookat(origin.translation(), end_effector, target);
//
//         let end_effector_local = origin.inverse() * end_effector;
//
//         houlog("target", target);
//         houlog("origin", origin);
//         houlog("end_effector", end_effector);
//         houlog(
//             "end_effector_result",
//             origin * Mat4::from_axis_angle(axis, angle) * end_effector_local,
//         );
//
//         houlog("axis", axis);
//         houlog("angle", angle.to_degrees());
//
//         save_houlog()?;
//         Ok(())
//     }
// }
