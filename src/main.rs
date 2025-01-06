use crate::model::ModelConfig;
use burn::backend::Wgpu;

mod data;
mod model;
mod training;

fn main() {
    let device = Default::default();
    let model = ModelConfig::new(10, 512).init::<Wgpu>(&device);
    println!("{}", model);
}
