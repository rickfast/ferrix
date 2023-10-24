use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Module, Tensor as CandleTensor};
use ferrix_model_api::{
    internal::{InferRequest, InferResponse, OutputTensor, TensorData},
    Model,
};

struct CandleModel {
    module: Arc<dyn Module + Send + Sync>,
}

impl CandleModel {
    fn new(config: ferrix_model_api::ModelConfig) -> Self {
        todo!()
    }
}

impl Model for CandleModel {
    fn load(&mut self) -> ferrix_model_api::ModelResult<()> {
        todo!()
    }

    fn loaded(&self) -> bool {
        todo!()
    }

    fn predict(
        &self,
        request: &InferRequest,
    ) -> ferrix_model_api::ModelResult<ferrix_model_api::internal::InferResponse> {
        let tensor: CandleTensor = request
            .inputs
            .first()
            .map(|input| {
                let datatype = input.datatype.as_str();
                let data = input.data.clone();

                let shape: Vec<usize> = input.shape.iter().map(|dim| *dim as usize).collect();
                let contents = match datatype {
                    "INT64" => data.int64_contents.into_iter(),
                    _ => todo!(),
                };

                let tensor =
                    CandleTensor::from_iter(contents.into_iter(), &candle_core::Device::Cpu)
                        .unwrap();

                tensor.reshape(shape)
            })
            .unwrap()?;

        let result = self.module.forward(&tensor)?;
        let shape = result
            .shape()
            .dims()
            .to_vec()
            .iter()
            .map(|dim| *dim as i64)
            .collect();
        let mut data = TensorData::default();
        let datatype = match result.dtype() {
            DType::U8 => "UINT8",
            DType::U32 => "UINT32",
            DType::I64 => "INT64",
            DType::BF16 => "FP16",
            DType::F16 => "FP16",
            DType::F32 => "FP32",
            DType::F64 => "FP64",
        };
        let flattened = result.flatten_all()?;

        match datatype {
            "UINT8" | "UINT32" => data.uint_contents = flattened.to_vec1()?,
            "INT64" => data.int64_contents = flattened.to_vec1()?,
            "FP32" => data.fp32_contents = flattened.to_vec1()?,
            "FP64" => data.fp64_contents = flattened.to_vec1()?,
            _ => todo!(),
        }

        Ok(InferResponse {
            model_name: request.model_name.to_string(),
            id: request.model_name.to_string(),
            parameters: HashMap::new(),
            outputs: vec![OutputTensor {
                name: "".to_string(),
                datatype: datatype.to_string(),
                shape,
                parameters: HashMap::new(),
                data,
            }],
        })
    }
}

#[cfg(test)]
mod tests {}
