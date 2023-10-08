use pyo3::types::*;
use pyo3::*;

#[pyclass(name = "InferRequest")]
#[derive(Clone)]
pub struct PyInferRequest {
    #[pyo3(get, set)]
    pub model_name: String,
    #[pyo3(get, set)]
    pub model_version: String,
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub parameters: Py<PyDict>,
    #[pyo3(get, set)]
    pub inputs: Py<PyList>,
    #[pyo3(get, set)]
    pub outputs: Py<PyList>,
    #[pyo3(get, set)]
    pub raw_input_contents: Py<PyList>,
}

#[pymethods]
impl PyInferRequest {
    #[new]
    pub fn new(
        model_name: String,
        model_version: String,
        id: String,
        parameters: Py<PyDict>,
        inputs: Py<PyList>,
        outputs: Py<PyList>,
        raw_input_contents: Py<PyList>,
    ) -> Self {
        PyInferRequest {
            model_name,
            model_version,
            id,
            parameters,
            inputs,
            outputs,
            raw_input_contents,
        }
    }
}

#[pyclass(name = "Parameter")]
#[derive(Clone)]
pub struct PyParameter {
    #[pyo3(get, set)]
    pub str_param: Option<String>,
    #[pyo3(get, set)]
    pub int_param: Option<i64>,
    #[pyo3(get, set)]
    pub float_param: Option<f64>,
    #[pyo3(get, set)]
    pub bool_param: Option<bool>,
}

#[pymethods]
impl PyParameter {
    #[new]
    pub fn new(
        str_param: Option<String>,
        int_param: Option<i64>,
        float_param: Option<f64>,
        bool_param: Option<bool>,
    ) -> Self {
        PyParameter {
            str_param,
            int_param,
            float_param,
            bool_param,
        }
    }
}

#[pyclass(name = "InferResponse")]
#[derive(Clone)]
pub struct PyInferResponse {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub model_name: String,
    #[pyo3(get, set)]
    pub parameters: Py<PyDict>,
    #[pyo3(get, set)]
    pub outputs: Py<PyList>,
}

#[pymethods]
impl PyInferResponse {
    #[new]
    pub fn new(
        id: String,
        model_name: String,
        parameters: Py<PyDict>,
        outputs: Py<PyList>,
    ) -> Self {
        PyInferResponse {
            id,
            model_name,
            parameters,
            outputs,
        }
    }
}

#[pyclass(name = "InferInput")]
#[derive(Clone)]
pub struct PyInferInput {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub datatype: String,
    #[pyo3(get, set)]
    pub shape: Vec<i64>,
    #[pyo3(get, set)]
    pub parameters: Py<PyDict>,
    #[pyo3(get, set)]
    pub data: Py<PyList>,
}

#[pymethods]
impl PyInferInput {
    #[new]
    pub fn new(
        name: String,
        datatype: String,
        shape: Vec<i64>,
        parameters: Py<PyDict>,
        data: Py<PyList>,
    ) -> Self {
        PyInferInput {
            name,
            datatype,
            shape,
            parameters,
            data,
        }
    }

    pub fn as_numpy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let np = py.import("numpy").unwrap();
        let dtype_str = match self.datatype.as_str() {
            "FP32" => "f4",
            _ => todo!(),
        };
        let dtype = np
            .call_method1("dtype", PyTuple::new(py, [dtype_str]))
            .unwrap();

        Ok(np
            .call_method(
                "array",
                PyTuple::new(py, [self.data.clone()]),
                Some([("dtype", dtype)].into_py_dict(py)),
            )
            .unwrap()
            .into_py(py))
    }
}

#[pyclass(name = "InferOutput")]
#[derive(Clone)]
pub struct PyInferOutput {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub datatype: String,
    #[pyo3(get, set)]
    pub shape: Vec<i64>,
    #[pyo3(get, set)]
    pub parameters: Py<PyDict>,
    #[pyo3(get, set)]
    pub data: Py<PyList>,
}

#[pymethods]
impl PyInferOutput {
    #[new]
    pub fn new(
        name: String,
        datatype: String,
        shape: Vec<i64>,
        parameters: Py<PyDict>,
        data: Py<PyList>,
    ) -> Self {
        PyInferOutput {
            name,
            datatype,
            shape,
            parameters,
            data,
        }
    }
}
