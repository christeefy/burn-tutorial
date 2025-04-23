#![recursion_limit = "256"]
use crate::model::ModelConfig;
use crate::training::TrainingConfig;
use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    data::dataset::Dataset,
    optim::AdamConfig,
};

mod data;
mod inference;
mod model;
mod training;

fn main() {
    let artifact_dir = "/tmp/guide";
    let device: WgpuDevice = Default::default();

    crate::training::train::<Autodiff<Wgpu>>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );

    crate::inference::infer::<Wgpu>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
}
