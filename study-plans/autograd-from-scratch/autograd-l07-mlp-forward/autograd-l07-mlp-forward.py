import torch

def mlp_forward(inputs: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor]) -> tuple:
    """
    Returns the final output tensor and a list of layer-output tensors.
    """
    h = inputs
    layers = [h := torch.tanh(w @ h + b) for w, b in zip(weights, biases)]
    return h, layers