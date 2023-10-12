from ferrix import InferRequest, InferResponse, InferInput, InferOutput, preprocessor, postprocessor
from PIL import Image
from torchvision import transforms

import torch


@preprocessor
def test(infer_input: InferRequest) -> InferRequest:
    image_url = infer_input.parameters["image"]
    input_image = Image.open(image_url)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    result_input = InferInput(
        name = "image", 
        datatype = "FP32", 
        shape = input_batch.shape(), 
        parameters = None, 
        contents = input_batch.flatten().tolist()
    )

    infer_input.inputs = [result_input]

    return infer_input


@postprocessor
def test_out(infer_output: InferResponse) -> InferResponse:
    output = torch.tensor(infer_output.outputs[0].data)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    result_output = InferOutput(
        name = "image", 
        datatype = "FP32", 
        shape = probabilities.shape(), 
        parameters = None, 
        contents = probabilities.flatten().tolist()
    )

    infer_output.outputs = [result_output]

    return infer_output