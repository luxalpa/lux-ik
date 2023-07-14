use glam::{vec3, Mat4, Quat, Vec3};
use std::f32::consts::PI;

pub(crate) trait ToAxisAngle180 {
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

fn _fuzzy_compare_vec3(a: Vec3, b: Vec3) -> bool {
    let epsilon = 0.01;
    (a.x - b.x).abs() < epsilon && (a.y - b.y).abs() < epsilon && (a.z - b.z).abs() < epsilon
}

fn _fuzzy_compare_f32(a: f32, b: f32) -> bool {
    let epsilon = 0.0001;
    (a - b).abs() < epsilon
}

// calculates the difference between two angles, respecting 360 degree wrapping
pub fn ang_diff(a: f32, b: f32) -> f32 {
    let delta = b - a;
    (delta + PI) % (2.0 * PI) - PI
}

pub trait Mat4Helpers {
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

// Retrieve the angle of rotation around the given axis
// https://stackoverflow.com/questions/3684269/component-of-a-quaternion-rotation-around-an-axis
// TODO: Check if this is really what we want!
pub fn swing_twist_decompose(q: Quat, dir: Vec3) -> f32 {
    let rotation_axis = vec3(q.x, q.y, q.z);
    let dot_prod = dir.dot(rotation_axis);
    let p = dir * dot_prod;
    let mut twist = Quat::from_xyzw(p.x, p.y, p.z, q.w).normalize();

    if dot_prod < 0.0 {
        twist = -twist;
    }

    twist.to_axis_angle_180().1
}

const ROTATION_AXIS_EPSILON: f32 = 0.0001;

pub fn get_rotation_axis(to_e: Vec3, target_direction: Vec3) -> Vec3 {
    let raw_axis = target_direction.cross(to_e);
    if raw_axis.length_squared() > ROTATION_AXIS_EPSILON {
        return raw_axis.normalize();
    }

    // TODO: Use orthonormal vector? Or just ZERO vector?
    let raw_axis = target_direction.cross(Vec3::Y);
    if raw_axis.length_squared() > ROTATION_AXIS_EPSILON {
        return raw_axis.normalize();
    }

    let raw_axis = target_direction.cross(Vec3::Z);
    return if raw_axis.length_squared() > ROTATION_AXIS_EPSILON {
        raw_axis.normalize()
    } else {
        // if we are very close to the end effector, then there's no more useful rotation axis.
        // Returning ZERO will cause the influence to be 0. This also prevents normalize from being NaN.
        Vec3::ZERO
    };
}
