use std::{collections::BTreeMap, ops::Bound};

use itertools::Itertools;
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

pub struct BTreeHistory {
    btrees: Vec<BTreeMap<NotNan<f32>, u64>>,
}

impl BTreeHistory {
    pub fn new(dimensions: usize) -> Self {
        Self {
            btrees: vec![BTreeMap::new(); dimensions],
        }
    }
}

impl ProjectionHistory for BTreeHistory {
    fn insert(&mut self, projections: &[NotNan<f32>], frame: u64) {
        for (btree, projection) in self.btrees.iter_mut().zip(projections.iter()) {
            btree.insert(*projection, frame);
        }
    }

    fn search(
        &self,
        query: &[NotNan<f32>],
        count: usize,
        current_frame: u64,
        ignore_frame_count: u64,
    ) -> impl Iterator<Item = u64> {
        // Takes the closest frame in each btree, interleaving the results by btree.
        InterleaveIterator {
            iterators: self
                .btrees
                .iter()
                .zip(query.iter())
                .map(|(btree, projection)| {
                    btree
                        .range(..*projection)
                        .rev()
                        .merge_by(
                            btree.range((Bound::Excluded(*projection), Bound::Unbounded)),
                            |x, y| f32::abs(**x.0 - **projection) <= f32::abs(**y.0 - **projection),
                        )
                        .map(|(_, i)| *i)
                })
                .collect(),
            index: 0,
        }
        .filter(move |f| (current_frame.abs_diff(*f)) > ignore_frame_count)
        .take(count)
    }
}

pub struct InterleaveIterator<T: Iterator<Item = u64>> {
    iterators: Vec<T>,
    index: usize,
}

impl<T: Iterator<Item = u64>> Iterator for InterleaveIterator<T> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.iterators.is_empty() {
            return None;
        }
        self.index %= self.iterators.len();
        let item = self.iterators[self.index].next();
        self.index += 1;
        // If the selected iterator was exhausted, remove it and try the next one.
        // Recursion is bounded by number of iterators.
        if item.is_none() {
            self.iterators.remove(self.index - 1);
            return self.next();
        }
        item
    }
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
