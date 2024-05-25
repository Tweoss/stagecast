use ordered_float::NotNan;
use rstar::{primitives::GeomWithData, RTree};

pub trait ProjectionHistory {
    fn insert(&mut self, projections: &[NotNan<f32>], frame: u64);
    fn search(
        &self,
        query: &[NotNan<f32>],
        count: usize,
        current_frame: u64,
        ignore_frame_count: u64,
    ) -> impl Iterator<Item = u64>;
}

pub struct RTReeHistory<const LENGTH: usize> {
    rtree: RTree<GeomWithData<[f32; LENGTH], u64>>,
}

impl<const LENGTH: usize> Default for RTReeHistory<LENGTH> {
    fn default() -> Self {
        Self {
            rtree: RTree::new(),
        }
    }
}

impl<const LENGTH: usize> ProjectionHistory for RTReeHistory<LENGTH> {
    fn insert(&mut self, projections: &[NotNan<f32>], frame: u64) {
        self.rtree.insert(GeomWithData::new(
            std::convert::TryInto::<[f32; LENGTH]>::try_into(
                projections
                    .iter()
                    .map(|f| f32::from(*f))
                    .collect::<Vec<_>>(),
            )
            .expect("mismatched length"),
            frame,
        ))
    }

    fn search(
        &self,
        query: &[NotNan<f32>],
        count: usize,
        current_frame: u64,
        ignore_frame_count: u64,
    ) -> impl Iterator<Item = u64> {
        self.rtree
            .nearest_neighbor_iter(
                &query
                    .iter()
                    .map(|f| f32::from(*f))
                    .collect::<Vec<_>>()
                    .try_into()
                    .expect("mismatched length"),
            )
            .map(|d| d.data)
            .filter(move |f| (current_frame.abs_diff(*f)) > ignore_frame_count)
            .take(count)
    }
}
