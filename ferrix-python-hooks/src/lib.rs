use ferrix_model_api::internal::*;
use ferrix_model_api::python::{
    PyInferInput, PyInferOutput, PyInferRequest, PyInferResponse, PyParameter,
};
use once_cell::sync::OnceCell;
use pyo3::types::{PyDict, PyModule, PyTuple};
use pyo3::ToPyObject;
use pyo3::{pymodule, wrap_pyfunction, Py, PyAny, PyResult, Python};

static PREPROCESSOR: OnceCell<Py<PyAny>> = OnceCell::new();
static POSTPROCESSOR: OnceCell<Py<PyAny>> = OnceCell::new();

const CODE: &str = "import ferrix

@ferrix.preprocessor
def test(infer_input: ferrix.InferRequest) -> ferrix.InferRequest:
    return infer_input

@ferrix.postprocessor
def test_out(infer_output: ferrix.InferResponse) -> ferrix.InferResponse:
    return infer_output
";

#[pyo3::pyfunction]
fn preprocessor(function: Py<PyAny>) {
    let _ = PREPROCESSOR.set(function);
}

#[pyo3::pyfunction]
fn postprocessor(function: Py<PyAny>) {
    let _ = POSTPROCESSOR.set(function);
}

#[pymodule]
fn ferrix(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(preprocessor, module)?)?;
    module.add_function(wrap_pyfunction!(postprocessor, module)?)?;
    module.add_class::<PyInferRequest>()?;
    module.add_class::<PyInferResponse>()?;
    module.add_class::<PyInferInput>()?;
    module.add_class::<PyInferOutput>()?;
    module.add_class::<PyParameter>()?;

    Ok(())
}

pub fn eval(code: String) {
    pyo3::append_to_inittab!(ferrix);
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| py.run(&code, None, None).unwrap())
}

pub fn preprocess(input: InferRequest) -> PyResult<InferRequest> {
    Python::with_gil(|py| {
        let input = input.to_object(py);
        let args = PyTuple::new(py, &[input]);
        let response = PREPROCESSOR
            .get()
            .unwrap()
            .call(py, args, Some(PyDict::new(py)))?;

        response.extract::<InferRequest>(py)
    })
}

pub fn postprocess(input: InferResponse) -> PyResult<InferResponse> {
    Python::with_gil(|py| {
        let input = input.to_object(py);
        let args = PyTuple::new(py, &[input]);
        let response = PREPROCESSOR
            .get()
            .unwrap()
            .call(py, args, Some(PyDict::new(py)))?;

        response.extract::<InferResponse>(py)
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn test() {
        eval(CODE.to_string());

        PREPROCESSOR.get().unwrap();

        let infer_request: InferRequest = InferRequest {
            model_name: "".to_string(),
            model_version: "".to_string(),
            id: "1".to_string(),
            parameters: HashMap::new(),
            inputs: vec![InputTensor {
                datatype: "FP32".to_string(),
                name: "".to_string(),
                parameters: HashMap::new(),
                shape: vec![1],
                data: TensorData {
                    bool_contents: vec![],
                    int_contents: vec![],
                    int64_contents: vec![],
                    uint_contents: vec![],
                    uint64_contents: vec![],
                    fp32_contents: vec![],
                    fp64_contents: vec![1.0],
                    bytes_contents: vec![],
                },
            }],
            // inputs: vec![],
            outputs: vec![],
            raw_input_contents: vec![vec![1_u8]],
        };

        let response = preprocess(infer_request);

        assert_eq!("1".to_string(), response.unwrap().id)
    }
}
