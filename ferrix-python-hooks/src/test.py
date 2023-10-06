from ferrix import InferInput, preprocessor

@preprocessor
def test(infer_input: InferInput) -> InferInput:
    return infer_input