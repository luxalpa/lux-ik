use glam::Vec3;

pub trait SlicePushable<T> {
    const NUM_ELEMENTS: usize;
    fn push_to_slice(self, slice: &mut [T], idx: usize);
}

pub struct SlicePusher<'a, T> {
    index: usize,
    slice: &'a mut [T],
}

impl<T> SlicePusher<'_, T> {
    pub fn new(slice: &mut [T]) -> SlicePusher<T> {
        SlicePusher { index: 0, slice }
    }

    pub fn push<U: SlicePushable<T>>(&mut self, value: U) {
        value.push_to_slice(self.slice, self.index);
        self.index += U::NUM_ELEMENTS;
    }

    pub fn skip<U: SlicePushable<T>>(&mut self) {
        self.index += U::NUM_ELEMENTS;
    }
}

impl SlicePushable<f32> for f32 {
    const NUM_ELEMENTS: usize = 1;

    fn push_to_slice(self, slice: &mut [f32], idx: usize) {
        slice[idx] = self;
    }
}

impl SlicePushable<f32> for Vec3 {
    const NUM_ELEMENTS: usize = 3;

    fn push_to_slice(self, slice: &mut [f32], idx: usize) {
        slice[idx] = self.x;
        slice[idx + 1] = self.y;
        slice[idx + 2] = self.z;
    }
}
