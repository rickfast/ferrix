use clap::Parser;

use ferrix_model_api::ModelConfig;
use ferrix_model_pytorch::*;
use ferrix_server::inference::{Inference, InferenceConfig};
use ferrix_server::GrpcInferenceServiceImpl;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Config {
    /// gRPC service port
    #[arg(short, long, default_value_t = 6565_i16)]
    port: i16,

    /// Path to Ferrix model configuration
    #[arg(short, long, default_value = "./ferrix.toml")]
    model_config: String,

    /// Path to Ferrix transformer Python file
    #[arg(short, long, default_value = "/workspaces/ferrix/examples/pytorch-resnet/handler.py")]
    transformer: String,
}

#[tokio::main]
async fn main() {
    let config = Config::parse();
    let model_config = toml::from_str::<ModelConfig>(
        r#"
        base_path = "/workspaces/ferrix/ferrix-model-pytorch/resource/model.pt"
        model_name = "Blah"
    "#,
    )
    .unwrap();
    let handler_path = if std::path::Path::new(&config.transformer).exists() {
        Some(config.transformer)
    } else {
        None
    };
    let model = PyTorchModel::new(model_config);
    let boxed_model = Box::new(model);
    let mut inference = Inference::new(InferenceConfig { handler_path }, boxed_model);
    let _ = inference.load();
    let service = GrpcInferenceServiceImpl::with_model(inference);

    match ferrix_server::serve(config.port, service).await {
        Ok(()) => println!("Ferrix started"),
        Err(err) => println!("Error! {}", err.to_string())
    }
}
