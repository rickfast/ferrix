use clap::Parser;

use ferrix_model_api::{Model, ModelConfig};
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

    /// Path to Ferrix model configuration
    #[arg(short, long, default_value = "./handler.py")]
    handler: String,
}

#[tokio::main]
async fn main() {
    let config = Config::parse();
    let model_config = toml::from_str::<ModelConfig>(
        r#"
        base_path: blah.pt
    "#,
    )
    .unwrap();
    let handler_path = if std::path::Path::new(&config.handler).exists() {
        Some(config.handler)
    } else {
        None
    };
    let model = PyTorchModel::new(model_config);
    let boxed_model = Box::new(model);
    let inference = Inference::new(InferenceConfig { handler_path }, boxed_model);
    let service = GrpcInferenceServiceImpl::with_model(inference);

    let _ = ferrix_server::serve(config.port, service).await;
}
