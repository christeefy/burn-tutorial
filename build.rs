use burn_import::onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("../../rust-ml/onnx-optimum/distilbert_base_uncased_squad_onnx/model.onnx")
        .out_dir("model/")
        .run_from_script();
}
