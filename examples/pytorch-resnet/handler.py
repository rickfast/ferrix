from ferrix import InferRequest, InferResponse, preprocessor, postprocessor

@preprocessor
def test(infer_input: InferRequest) -> InferRequest:
    array = infer_input.inputs[0].as_numpy()

    # do some shit

    return infer_input

@postprocessor
def test_out(infer_output: InferResponse) -> InferResponse:
    return infer_output