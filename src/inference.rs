use crate::{data::MnistBatcher, training::TrainingConfig};
use burn::{
    config::Config,
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    module::Module,
    prelude::Backend,
    record::{CompactRecorder, Recorder},
};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: MnistItem) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config sohuld exist for the model");

    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(&device).load_record(record);

    let label = item.label;
    let batcher = MnistBatcher::new(device.clone());
    let batch = batcher.batch(vec![item], &device);
    let output = model.forward(batch.images);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted {predicted} Expected {label}");
}
