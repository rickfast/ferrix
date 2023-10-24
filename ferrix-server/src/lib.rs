use ferrix_model_api::internal::InferRequest;
use inference::Inference;
use tonic::Response;

use ferrix_model_api::Model;
use ferrix_protos::grpc_inference_service_server::{
    GrpcInferenceService, GrpcInferenceServiceServer,
};
use ferrix_protos::*;
use tonic::transport::Server;

pub mod inference;

// #[derive(Default)]
pub struct GrpcInferenceServiceImpl {
    model: Inference,
}

impl GrpcInferenceServiceImpl {
    pub fn with_model(model: Inference) -> Self {
        GrpcInferenceServiceImpl { model }
    }
}

#[tonic::async_trait]
impl GrpcInferenceService for GrpcInferenceServiceImpl {
    async fn server_live(
        &self,
        _: tonic::Request<ServerLiveRequest>,
    ) -> std::result::Result<tonic::Response<ServerLiveResponse>, tonic::Status> {
        return Ok(Response::new(ServerLiveResponse { live: true }));
    }

    /// The ServerReady API indicates if the server is ready for inferencing.
    async fn server_ready(
        &self,
        _: tonic::Request<ServerReadyRequest>,
    ) -> std::result::Result<tonic::Response<ServerReadyResponse>, tonic::Status> {
        return Ok(Response::new(ServerReadyResponse { ready: true }));
    }

    /// The ModelReady API indicates if a specific model is ready for inferencing.
    async fn model_ready(
        &self,
        _: tonic::Request<ModelReadyRequest>,
    ) -> std::result::Result<tonic::Response<ModelReadyResponse>, tonic::Status> {
        return Ok(Response::new(ModelReadyResponse {
            ready: self.model.loaded(),
        }));
    }

    /// The ServerMetadata API provides information about the server. Errors are
    /// indicated by the google.rpc.Status returned for the request. The OK code
    /// indicates success and other codes indicate failure.
    async fn server_metadata(
        &self,
        request: tonic::Request<ServerMetadataRequest>,
    ) -> std::result::Result<tonic::Response<ServerMetadataResponse>, tonic::Status> {
        return Ok(Response::new(ServerMetadataResponse {
            name: "".to_string(),
            version: "".to_string(),
            extensions: vec![],
        }));
    }

    /// The per-model metadata API provides information about a model. Errors are
    /// indicated by the google.rpc.Status returned for the request. The OK code
    /// indicates success and other codes indicate failure.
    async fn model_metadata(
        &self,
        request: tonic::Request<ModelMetadataRequest>,
    ) -> std::result::Result<tonic::Response<ModelMetadataResponse>, tonic::Status> {
        return Ok(Response::new(ModelMetadataResponse {
            name: "".to_string(),
            versions: vec![],
            platform: "".to_string(),
            inputs: vec![],
            outputs: vec![],
        }));
    }

    /// The ModelInfer API performs inference using the specified model. Errors are
    /// indicated by the google.rpc.Status returned for the request. The OK code
    /// indicates success and other codes indicate failure.
    async fn model_infer(
        &self,
        request: tonic::Request<ModelInferRequest>,
    ) -> std::result::Result<tonic::Response<ModelInferResponse>, tonic::Status> {
        let infer_request = InferRequest::from_proto(request.into_inner());
        let infer_result = self.model.predict(infer_request).await;

        match infer_result {
            Ok(infer_response) => Ok(Response::new(infer_response.to_proto())),
            Err(error) => Err(tonic::Status::internal(error.to_string())),
        }
    }
}

pub async fn  serve(
    port: i16,
    service: GrpcInferenceServiceImpl,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("[::1]:{}", port).parse().unwrap();

    println!("Ferrix listening on {}", addr);

    Server::builder()
        .add_service(GrpcInferenceServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
