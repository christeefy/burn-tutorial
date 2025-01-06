use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::Backend,
    tensor::{Int, Tensor},
};

pub struct MnistBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> MnistBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<MnistItem, MnistBatch<B>> for MnistBatch<B> {
    fn batch(&self, items: Vec<MnistItem>) -> MnistBatch<B> {
        todo!()
    }
}
