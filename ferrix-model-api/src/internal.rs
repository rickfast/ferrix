use core::panic;
use std::collections::HashMap;

use ferrix_protos::infer_parameter::*;
use ferrix_protos::model_infer_request::*;
use ferrix_protos::model_infer_response::InferOutputTensor;
use ferrix_protos::*;
use pyo3::PyResult;
use pyo3::types::IntoPyDict;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use pyo3::FromPyObject;
use pyo3::IntoPy;
use pyo3::Py;
use pyo3::ToPyObject;
use pyo3::PyAny;

use crate::python::PyInferInput;
use crate::python::PyInferOutput;
use crate::python::PyInferRequest;
use crate::python::PyInferResponse;
use crate::python::PyParameter;

macro_rules! to_bytevec {
    ($e:expr,$type:ty) => {
        $e.iter()
            .flat_map({ |i| i.clone().to_ne_bytes() })
            .collect::<Vec<u8>>()
    };
}

macro_rules! from_bytevec {
    ($e:expr,$type:ty,$size:expr) => {
        $e.chunks($size).map(|item| <$type>::from_ne_bytes(item.try_into().unwrap())).collect::<Vec<$type>>()
    };
}

#[derive(Clone, FromPyObject)]
pub struct InferRequest {
    pub model_name: String,
    pub model_version: String,
    pub id: String,
    pub parameters: HashMap<String, Parameter>,
    pub inputs: Vec<InputTensor>,
    pub outputs: Vec<InputTensor>,
    pub raw_input_contents: Vec<Vec<u8>>,
}

impl ToPyObject for InferRequest {
    fn to_object(&self, py: pyo3::Python<'_>) -> pyo3::PyObject {
        let parameters: Py<PyDict> = self.parameters.clone().into_py_dict(py).into_py(py);
        let x: Vec<Py<PyAny>> = self.inputs.iter().map(|input| input.to_object(py)).collect();
        let inputs = PyList::new(py, x).into_py(py);
        let outputs = PyList::empty(py).into_py(py);
        let raw = PyList::empty(py).into_py(py);
        let request = PyInferRequest::new(
            self.model_name.to_string(),
            self.model_version.to_string(),
            self.id.to_string(),
            parameters,
            inputs,
            outputs,
            raw,
        );

        return Py::new(py, request).unwrap().as_ref(py).into();
    }
}

impl InferRequest {
    pub fn from_proto(request: ModelInferRequest) -> Self {
        InferRequest {
            model_name: request.model_name,
            model_version: request.model_version,
            id: request.id,
            parameters: HashMap::new(),
            inputs: request
                .inputs
                .iter()
                .map(|input| InputTensor::from_proto(input))
                .collect::<Vec<InputTensor>>(),
            outputs: vec![],
            raw_input_contents: vec![vec![]],
        }
    }
}

#[derive(Clone, FromPyObject)]
pub struct InferResponse {
    pub model_name: String,
    pub id: String,
    pub parameters: HashMap<String, Parameter>,
    pub outputs: Vec<OutputTensor>,
}

impl ToPyObject for InferResponse {
    fn to_object(&self, py: pyo3::Python<'_>) -> pyo3::PyObject {
        let parameters: Py<PyDict> = self.parameters.clone().into_py_dict(py).into_py(py);
        let outputs = PyList::new(py, self.outputs.clone()).into_py(py);
        let response = PyInferResponse::new(
            self.id.to_string(),
            self.model_name.to_string(),
            parameters,
            outputs,
        );

        return Py::new(py, response).unwrap().as_ref(py).into();
    }
}

impl InferResponse {
    pub fn from_proto(response: ModelInferResponse) -> Self {
        InferResponse {
            id: response.id,
            model_name: response.model_name,
            parameters: HashMap::new(),
            outputs: response
                .outputs
                .iter()
                .map(|input| OutputTensor::from_proto(input))
                .collect::<Vec<OutputTensor>>(),
        }
    }

    pub fn to_proto(self) -> ModelInferResponse {
        let mut response = ModelInferResponse::default();

        response.id = self.id.to_string();
        response.outputs = self
            .outputs
            .iter()
            .map(|output| output.to_proto())
            .collect::<Vec<InferOutputTensor>>();

        response
    }
}

#[derive(Clone, FromPyObject)]
pub struct Parameter {
    pub str_param: Option<String>,
    pub int_param: Option<i64>,
    pub float_param: Option<f64>,
    pub bool_param: Option<bool>,
}

impl Parameter {
    pub fn from_proto(request: &InferParameter) -> Self {
        request
            .parameter_choice
            .clone()
            .map(|choice| match choice {
                ParameterChoice::BoolParam(bool_value) => Parameter {
                    bool_param: Some(bool_value),
                    float_param: None,
                    int_param: None,
                    str_param: None,
                },
                ParameterChoice::Int64Param(int_value) => Parameter {
                    bool_param: None,
                    float_param: None,
                    int_param: Some(int_value),
                    str_param: None,
                },
                ParameterChoice::StringParam(str_value) => Parameter {
                    bool_param: None,
                    float_param: None,
                    int_param: None,
                    str_param: Some(str_value),
                },
            })
            .unwrap()
    }
}

impl ToPyObject for Parameter {
    fn to_object(&self, py: pyo3::Python<'_>) -> pyo3::PyObject {
        let parameter = PyParameter::new(
            self.str_param.clone(),
            self.int_param.clone(),
            self.float_param.clone(),
            self.bool_param.clone(),
        );

        Py::new(py, parameter).unwrap().as_ref(py).into()
    }
}

#[derive(Clone)]
pub struct InputTensor {
    pub name: String,
    pub datatype: String,
    pub shape: Vec<i64>,
    pub parameters: HashMap<String, Parameter>,
    pub data: TensorData,
}

impl InputTensor {
    pub fn as_bytes(self) -> Vec<u8> {
        match self.datatype.as_str() {
            "INT32" => to_bytevec!(self.data.int_contents, i32),
            "FP32" => to_bytevec!(self.data.fp32_contents, f32),
            _ => panic!(""),
        }
    }

    fn from_proto(request: &InferInputTensor) -> Self {
        InputTensor {
            name: request.name.to_string(),
            datatype: request.datatype.to_string(),
            shape: request.shape.clone(),
            parameters: request
                .parameters
                .iter()
                .map(|(key, value)| (key.to_string(), Parameter::from_proto(value)))
                .collect::<HashMap<String, Parameter>>(),
            data: TensorData::from_proto(
                request.datatype.clone(),
                request.contents.as_ref().unwrap(),
            ),
        }
    }
}

impl FromPyObject<'_> for InputTensor {
    fn extract(ob: &'_ PyAny) -> pyo3::PyResult<Self> {
        let mut tensor_data = TensorData::default();
        let name: String = ob.getattr("name")?.extract()?;
        let datatype: String = ob.getattr("datatype")?.extract()?;
        let shape: Vec<i64> = ob.getattr("shape")?.extract()?;
        let parameters: HashMap<String, Parameter> = HashMap::new();
        let _: Result<(), anyhow::Error> = match datatype.as_str() {
            "FP32" => {
                tensor_data.fp32_contents = ob.getattr("data")?.extract()?;
                Ok(())
            },
            _ => todo!()
        };
        let tensor = InputTensor {
            name,
            datatype,
            shape,
            parameters,
            data: tensor_data
        };

        return PyResult::Ok(tensor);
    }
}

impl ToPyObject for InputTensor {
    fn to_object(&self, py: pyo3::Python<'_>) -> pyo3::PyObject {
        let parameters: Py<PyDict> = self.parameters.clone().into_py_dict(py).into_py(py);
        let contents = self.data.clone();
        let data: Py<PyList> = data_to_py(self.datatype.to_string(), contents, py);

        let tensor = PyInferInput::new(
            self.name.to_string(),
            self.datatype.to_string(),
            self.shape.clone(),
            parameters,
            data,
        );

        return Py::new(py, tensor).unwrap().as_ref(py).into();
    }
}

#[derive(Clone, FromPyObject)]
pub struct OutputTensor {
    pub name: String,
    pub datatype: String,
    pub shape: Vec<i64>,
    pub parameters: HashMap<String, Parameter>,
    pub data: TensorData,
}

impl OutputTensor {
    fn from_proto(response: &InferOutputTensor) -> Self {
        OutputTensor {
            name: response.name.to_string(),
            datatype: response.datatype.to_string(),
            shape: response.shape.clone(),
            parameters: response
                .parameters
                .iter()
                .map(|(key, value)| (key.to_string(), Parameter::from_proto(value)))
                .collect::<HashMap<String, Parameter>>(),
            data: TensorData::from_proto(
                response.datatype.clone(),
                response.contents.as_ref().unwrap(),
            ),
        }
    }

    fn to_proto(&self) -> InferOutputTensor {
        let mut tensor = InferOutputTensor::default();

        tensor.datatype = self.datatype.to_string();
        tensor.name = self.name.to_string();
        tensor.shape = self.shape.clone();
        tensor.parameters = self
            .parameters
            .iter()
            .map(|(key, value)| {
                let mut parameter = InferParameter::default();

                let choice = if value.bool_param.is_some() {
                    ParameterChoice::BoolParam(value.bool_param.unwrap())
                } else if value.int_param.is_some() {
                    ParameterChoice::Int64Param(value.int_param.unwrap())
                } else {
                    ParameterChoice::StringParam(value.str_param.clone().unwrap())
                };

                parameter.parameter_choice = Some(choice);

                (key.to_string(), parameter)
            })
            .collect::<HashMap<String, InferParameter>>();

        return tensor;
    }
}

impl ToPyObject for OutputTensor {
    fn to_object(&self, py: pyo3::Python<'_>) -> pyo3::PyObject {
        let parameters: Py<PyDict> = self.parameters.clone().into_py_dict(py).into_py(py);
        let contents = self.data.clone();
        let data: Py<PyList> = data_to_py(self.datatype.to_string(), contents, py);

        let tensor = PyInferOutput::new(
            self.name.to_string(),
            self.datatype.to_string(),
            self.shape.clone(),
            parameters,
            data,
        );

        return Py::new(py, tensor).unwrap().as_ref(py).into();
    }
}

#[derive(Clone, FromPyObject)]
pub struct TensorData {
    pub bool_contents: Vec<bool>,
    pub int_contents: Vec<i32>,
    pub int64_contents: Vec<i64>,
    pub uint_contents: Vec<u32>,
    pub uint64_contents: Vec<u64>,
    pub fp32_contents: Vec<f32>,
    pub fp64_contents: Vec<f64>,
    pub bytes_contents: Vec<Vec<u8>>,
}

impl TensorData {
    pub fn default() -> Self {
        TensorData {
            bool_contents: vec![],
            int_contents: vec![],
            int64_contents: vec![],
            uint_contents: vec![],
            uint64_contents: vec![],
            fp32_contents: vec![],
            fp64_contents: vec![],
            bytes_contents: vec![],
        }
    }

    fn from_proto(datatype: String, contents: &InferTensorContents) -> Self {
        let data = contents.clone();
        TensorData {
            bool_contents: data.bool_contents.clone(),
            int_contents: data.int_contents.clone(),
            int64_contents: data.int64_contents.clone(),
            uint_contents: data.uint_contents.clone(),
            uint64_contents: data.uint64_contents.clone(),
            fp32_contents: data.fp32_contents.clone(),
            fp64_contents: data.fp64_contents.clone(),
            bytes_contents: data.bytes_contents.clone(),
        }
    }

    fn to_proto(self) -> InferTensorContents {
        let mut contents = InferTensorContents::default();

        contents.bool_contents = self.bool_contents.clone();
        contents.bytes_contents = self.bytes_contents.clone();
        contents.fp32_contents = self.fp32_contents.clone();
        contents.fp64_contents = self.fp64_contents.clone();
        contents.int_contents = self.int_contents.clone();
        contents.int64_contents = self.int64_contents.clone();

        contents
    }

    pub fn from_bytes(datatype: String, bytes: &[u8]) -> Self {
        let mut data = TensorData::default();

        match datatype.as_str() {
            "INT32" => data.int_contents = from_bytevec!(bytes, i32, 4),
            "FP32" => data.fp32_contents = from_bytevec!(bytes, f32, 4),
            _ => todo!(),
        }

        data
    }
}

// Utils
fn data_to_py(datatype: String, contents: TensorData, py: pyo3::Python<'_>) -> Py<PyList> {
    match datatype.as_str() {
        "INT8" => todo!(),
        "INT16" => todo!(),
        "INT32" => contents.int_contents.into_py(py).extract(py).unwrap(),
        "INT64" => contents.int64_contents.into_py(py).extract(py).unwrap(),
        "UINT8" => todo!(),
        "UINT16" => todo!(),
        "UINT32" => contents.uint_contents.into_py(py).extract(py).unwrap(),
        "UINT64" => contents.uint64_contents.into_py(py).extract(py).unwrap(),
        "FP16" => todo!(),
        "FP32" => contents.fp32_contents.into_py(py).extract(py).unwrap(),
        "FP64" => contents.fp64_contents.into_py(py).extract(py).unwrap(),
        "BYTES" => todo!(),
        _ => todo!(),
    }
}
