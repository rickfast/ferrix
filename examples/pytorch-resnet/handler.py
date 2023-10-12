import ferrix

@ferrix.preprocessor
def test(infer_input: ferrix.InferRequest) -> ferrix.InferRequest:
    return infer_input

@ferrix.postprocessor
def test_out(infer_output: ferrix.InferResponse) -> ferrix.InferResponse:
    return infer_output