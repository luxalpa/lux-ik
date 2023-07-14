use glam::Mat4;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct JointMap {
    joint_ids: Rc<[usize]>,
    data: Box<[Mat4]>,
}

impl JointMap {
    pub fn new(joint_ids: Rc<[usize]>, data: Box<[Mat4]>) -> Self {
        Self { joint_ids, data }
    }

    pub fn get(&self, joint_id: usize) -> Option<Mat4> {
        self.joint_ids
            .iter()
            .position(|&id| id == joint_id)
            .map(|idx| self.data[idx])
    }

    pub fn set(&mut self, joint_id: usize, xform: Mat4) {
        let idx = self
            .joint_ids
            .iter()
            .position(|&id| id == joint_id)
            .unwrap();
        self.data[idx] = xform;
    }

    pub fn set_all<I>(&mut self, xforms: I)
    where
        I: Iterator<Item = Mat4>,
    {
        for (i, xform) in xforms.enumerate() {
            self.data[i] = xform;
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, &Mat4)> {
        self.joint_ids.iter().copied().zip(self.data.iter())
    }

    pub fn set_data_from(&mut self, other: &Self) {
        self.data.copy_from_slice(&other.data);
    }
}
