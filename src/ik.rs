use glam::{Mat4, Quat, Vec3};

pub fn solve(goal: Vec3, joint_chain: &[Mat4]) -> Vec<Mat4> {
    let chain = &joint_chain[..joint_chain.len() - 1];
    let end_effector = joint_chain.last().unwrap();

    // let chain_q = chain.iter().map(Quat::from_mat4).collect::<Vec<_>>();

    let end_effector_pos = end_effector.transform_point3(Vec3::ZERO);
    let origin_of_rotation = chain[0].transform_point3(Vec3::ZERO);

    let to_e = end_effector_pos - origin_of_rotation;
    let raw_axis_of_rotation = (goal - end_effector_pos).cross(to_e);
    let axis_of_rotation = raw_axis_of_rotation.normalize();
    let influence = axis_of_rotation.cross(to_e);

    vec![Mat4::IDENTITY; joint_chain.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fuzzy_compare_vec3(a: Vec3, b: Vec3) -> bool {
        let epsilon = 0.01;
        (a.x - b.x).abs() < epsilon && (a.y - b.y).abs() < epsilon && (a.z - b.z).abs() < epsilon
    }

    fn fuzzy_compare_f32(a: f32, b: f32) -> bool {
        let epsilon = 0.01;
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_solve() {
        let goal = Vec3::new(0.0, 5.0, 0.0);

        let j1 = Mat4::from_translation(Vec3::new(0.0, 10.0, 0.0));
        let j2 = Mat4::from_translation(Vec3::new(0.0, 5.0, 0.0));
        let j_end = Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0));

        let joint_chain = [j1, j2, j_end];

        let result = solve(goal, &joint_chain);

        assert_eq!(result.len(), joint_chain.len());

        let j1_result_p = result[0].transform_point3(Vec3::ZERO);
        let j2_result_p = result[1].transform_point3(Vec3::ZERO);
        let j_end_result_p = result[2].transform_point3(Vec3::ZERO);

        // Ensure the end effector is at the goal
        assert!(
            fuzzy_compare_vec3(j_end_result_p, goal),
            "End effector is currently at {:?}, but should be at {:?}",
            j_end_result_p,
            goal
        );

        // Ensure the middle joint is at the midpoint between the end effector and the first joint
        assert!(fuzzy_compare_f32(j2_result_p.y, 7.5));

        // ensure all lengths are still the same
        assert!(fuzzy_compare_f32((j2_result_p - j1_result_p).length(), 5.0));
        assert!(fuzzy_compare_f32(
            (j_end_result_p - j2_result_p).length(),
            5.0
        ));
    }
}
