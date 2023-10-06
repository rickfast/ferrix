use pyo3::prelude::*;

#[pyfunction]
fn preprocessor(_func: &PyAny) -> PyResult<()> {
    Ok(())
}

#[pyfunction]
fn postprocessor(_func: &PyAny) -> PyResult<()> {
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn ferrix(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(preprocessor, module)?)?;
    module.add_function(wrap_pyfunction!(postprocessor, module)?)?;
    Ok(())
}
