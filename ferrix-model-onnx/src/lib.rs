use ferrix_model_api::Model;
use ort::*;

struct OnnxModel;

impl Model for OnnxModel {
    fn load(&mut self) -> ferrix_model_api::ModelResult<()> {
        todo!()
    }

    fn loaded(&self) -> bool {
        todo!()
    }

    fn predict(
        &self,
        request: ferrix_model_api::internal::InferRequest,
    ) -> ferrix_model_api::ModelResult<ferrix_model_api::internal::InferResponse> {
        let environment = Environment::builder()
            .with_name("GPT-2")
            .with_execution_providers([ExecutionProvider::CUDA(Default::default())])
            .build()
            .unwrap()
            .into_arc();

        let session = SessionBuilder::new(&environment)
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .unwrap()
            .with_intra_threads(1)
            .unwrap()
            .with_model_from_file("model_filepath_ref");

        // let pt_tensors: Vec<PyTorchTensor> = request
        //     .inputs
        //     .iter()
        //     .map(|input| {
        //         let shape = &input.shape[..];
        //         let bytes = &input.clone().as_bytes()[..];
        //         let datatype = Self::kserve_type_to_pt(input.datatype.to_string());

        //         PyTorchTensor::from_data_size(bytes, &shape, datatype)
        //     })
        //     .collect::<Vec<PyTorchTensor>>();
        // session.unwrap().run(request.inputs);

        todo!()
    }
}

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
