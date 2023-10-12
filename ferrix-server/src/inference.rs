use async_trait::async_trait;
use ferrix_model_api::{Model, ModelResult, internal::{InferRequest, InferResponse}};
use ferrix_python_hooks::{eval, postprocess, preprocess};

use std::future::Future;

pub struct Inference {
    hooks_enabled: bool,
    model: Box<dyn Model>,
}

pub struct InferenceConfig {
    pub handler_path: Option<String>,
}

impl Inference {
    pub fn new(config: InferenceConfig, model: Box<dyn Model>) -> Self {
        let hooks_enabled = config.handler_path.is_some();

        if hooks_enabled {
            let handler_path = config.handler_path.unwrap();
            let code = std::fs::read_to_string(handler_path).unwrap();

            eval(code);
        }

        Inference {
            hooks_enabled,
            model,
        }
    }

    pub fn load(&mut self) -> ferrix_model_api::ModelResult<()> {
        self.model.load()
    }

    pub fn loaded(&self) -> bool {
        self.model.loaded()
    }

    pub async fn predict(
        &self,
        request: InferRequest,
    ) -> ModelResult<InferResponse> {
        let input = match self.hooks_enabled {
            true => preprocess(request)?,
            false => request,
        };

        let response = self.model.predict(input)?;

        let output = match self.hooks_enabled {
            true => postprocess(response)?,
            false => response,
        };

        Ok(output)
    }
}
