use glam::Vec3;

/// A Writer-like trait for "pushing"/serializing values to a slice. For example, a Vec3 has 3
/// components and writing it into a slice of floats should write 3 separate floats.
pub trait SliceWritable<T> {
    const NUM_ELEMENTS: usize;
    fn write_to_slice(self, slice: &mut [T], idx: usize);
}

pub struct SliceWriter<'a, T> {
    index: usize,
    slice: &'a mut [T],
}

impl<T> SliceWriter<'_, T> {
    pub fn new(slice: &mut [T]) -> SliceWriter<T> {
        SliceWriter { index: 0, slice }
    }

    pub fn write<U: SliceWritable<T>>(&mut self, value: U) {
        value.write_to_slice(self.slice, self.index);
        self.index += U::NUM_ELEMENTS;
    }

    pub fn skip<U: SliceWritable<T>>(&mut self) {
        self.index += U::NUM_ELEMENTS;
    }
}

impl SliceWritable<f32> for f32 {
    const NUM_ELEMENTS: usize = 1;

    fn write_to_slice(self, slice: &mut [f32], idx: usize) {
        slice[idx] = self;
    }
}

impl SliceWritable<f32> for Vec3 {
    const NUM_ELEMENTS: usize = 3;

    fn write_to_slice(self, slice: &mut [f32], idx: usize) {
        slice[idx] = self.x;
        slice[idx + 1] = self.y;
        slice[idx + 2] = self.z;
    }
}
