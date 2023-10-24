use std::collections::HashMap;
use std::sync::atomic::Ordering::*;
use std::vec;

use anyhow::bail;
use atomic_option::AtomicOption;
use ferrix_model_api::internal::*;
use ferrix_model_api::Model;
use ferrix_model_api::ModelConfig;
use ferrix_model_api::ModelError;
use ferrix_model_api::ModelResult;
use tch::CModule;
use tch::Kind;
use tch::Tensor as PyTorchTensor;

pub struct PyTorchModel {
    module: AtomicOption<CModule>,
    model_config: ModelConfig,
}

impl PyTorchModel {
    pub fn new(config: ferrix_model_api::ModelConfig) -> Self {
        PyTorchModel {
            module: AtomicOption::empty(),
            model_config: config,
        }
    }
}

impl Model for PyTorchModel {
    fn load(&mut self) -> ModelResult<()> {
        let file_name = self.model_config.base_path.to_string();

        println!("Loading file {}", file_name);
        let result = tch::CModule::load(self.model_config.base_path.to_string());
        let model = match result {
            Ok(module) => module,
            Err(error) => bail!(ModelError::Load(error.to_string())),
        };

        self.module.replace(Some(Box::new(model)), SeqCst);

        return Ok(());
    }

    fn loaded(&self) -> bool {
        return self.module.take(SeqCst).is_some();
    }

    fn predict(&self, request: &InferRequest) -> ModelResult<InferResponse> {
        let pt_tensors: Vec<PyTorchTensor> = request
            .inputs
            .iter()
            .map(|input| {
                let shape = &input.shape[..];
                let bytes = &input.clone().as_bytes()[..];
                let datatype = Self::kserve_type_to_pt(input.datatype.to_string());

                PyTorchTensor::from_data_size(bytes, &shape, datatype)
            })
            .collect::<Vec<PyTorchTensor>>();
        let model = self.module.take(SeqCst).unwrap();
        let output_tensor = pt_tensors[0].apply(model.as_ref());
        let numel = output_tensor.numel();
        let output_bytes = &mut vec![0u8; numel * output_tensor.kind().elt_size_in_bytes()][..];
        let datatype = Self::pt_type_to_kserve(output_tensor.kind());
        let shape = output_tensor.size();
        let flat = output_tensor.flatten(0, (output_tensor.dim() as i64) - 1);

        flat.copy_data_u8(output_bytes, numel);

        Ok(InferResponse {
            model_name: request.model_name.to_string(),
            id: request.model_name.to_string(),
            parameters: HashMap::new(),
            outputs: vec![OutputTensor {
                name: "".to_string(),
                datatype: datatype.to_string(),
                shape,
                parameters: HashMap::new(),
                data: TensorData::from_bytes(datatype, output_bytes),
            }],
        })
    }
}

impl PyTorchModel {
    fn kserve_type_to_pt(datatype: String) -> Kind {
        match datatype.as_str() {
            "BOOL" => Kind::Bool,
            "UINT8" => Kind::Uint8,
            "UINT16" => todo!(),
            "UINT32" => todo!(),
            "UINT64" => todo!(),
            "INT8" => Kind::Int8,
            "INT16" => Kind::Int16,
            "INT32" => Kind::Int,
            "INT64" => Kind::Int64,
            "FP16" => Kind::Half,
            "FP32" => Kind::Float,
            "FP64" => Kind::Double,
            "BYTES" => todo!(),
            "BF16" => todo!(),
            _ => panic!(""),
        }
    }

    fn pt_type_to_kserve(datatype: Kind) -> String {
        match datatype {
            Kind::Uint8 => "UINT8".to_string(),
            Kind::Int8 => "INT8".to_string(),
            Kind::Int16 => "INT16".to_string(),
            Kind::Int => "INT32".to_string(),
            Kind::Int64 => "INT64".to_string(),
            Kind::Half => "FP16".to_string(),
            Kind::Float => "FP32".to_string(),
            Kind::Double => "FP64".to_string(),
            Kind::ComplexHalf => todo!(),
            Kind::ComplexFloat => todo!(),
            Kind::ComplexDouble => todo!(),
            Kind::Bool => todo!(),
            Kind::QInt8 => todo!(),
            Kind::QUInt8 => todo!(),
            Kind::QInt32 => todo!(),
            Kind::BFloat16 => todo!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use tch::vision::imagenet;

    use super::*;

    #[test]
    fn test_basic_pytorch_inference() {
        let resource_dir = format!("{}/resource", env!("CARGO_MANIFEST_DIR"));
        let saved_model_filename = format!("{}/model.pt", resource_dir);
        let mut model = PyTorchModel::new(ModelConfig {
            model_name: String::from(""),
            base_path: saved_model_filename,
            extended_config: None,
        });
        let load_result = model.load();

        match load_result {
            Ok(()) => assert!(true),
            Err(error) => assert_eq!("", error.to_string()),
        }

        let cat_image = format!("{}/cat.jpeg", resource_dir);
        let image_tensor = imagenet::load_image_and_resize224(cat_image).unwrap();
        let numel = image_tensor.numel();
        let data = &mut vec![0.0_f32; numel][..];

        image_tensor.copy_data(data, numel);

        let mut tensor_data = TensorData::default();

        tensor_data.fp32_contents = data.to_vec();

        let request = InferRequest {
            id: "".to_string(),
            model_name: "".to_string(),
            model_version: "".to_string(),
            outputs: vec![],
            parameters: HashMap::new(),
            raw_input_contents: vec![],
            inputs: vec![InputTensor {
                datatype: "FP32".to_string(),
                name: "".to_string(),
                parameters: HashMap::new(),
                shape: vec![1, 3, 224, 224],
                data: tensor_data,
            }],
        };
        let result = model.predict(&request);

        match result {
            Ok(response) => assert!(response.outputs.first().is_some()),
            Err(error) => assert_eq!("", error.to_string()),
        }
    }
}
