import torch
from torch import nn
from pathlib import Path
import time

def load_model(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> dict:
    """
    Loads a saved model checkpoint (weights + metadata).

    Args:
        model (torch.nn.Module): An *uninitialized* model instance with the same architecture.
        checkpoint_path (Path): Full path to the `.pt` file.
        device (torch.device): Device to map model and data to (e.g. 'cpu' or 'cuda').

    Returns:
        dict: Metadata dictionary containing training details, metrics, etc.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()  # Set to evaluation mode

    print(f"[INFO] Model loaded from {checkpoint_path}")

    # Remove state dict from metadata for clarity
    metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    return metadata

def save_model(model: torch.nn.Module,  metadata: dict, name: str, loc: Path, device: torch.device):
    """
        Saves the model into the desired location along with its metadata
        Args:
            model: model to be saved
            metadata: dictionary that saves relevant info about the mode such as {train acc, training images, no. of parameters}
            name: name of the file
            loc: location to save the file
    """
    model.to(device)
    metadata["model_state_dict"] = model.state_dict()
    save_path = loc/f"{name}.pt"

    torch.save(metadata, save_path)
    print(f"model saved at {save_path}")

def make_prediction(model: torch.nn.Module, data: tuple, classes: list, device: torch.device):
    """
    Makes a prediction on the given model and data
    Args:
        model: PyTorch model to make prediction
        data: tuple that stores (image tensor, label)
        classes: list containing all the clases
        device: device in which model and tensors are running
    """

    device = torch.device(device)
    model.to(device)

    with torch.inference_mode():
        start = time.time()
        image, label = data
        image = image.unsqueeze(0).to(device)
        
        y_pred_logits = model(image)
        y_pred_probs = torch.softmax(y_pred_logits, dim=1)
        y_pred_class = torch.argmax(y_pred_probs , dim=1)
        end = time.time()
        inference_time = round((end - start) * 1000, 3)

        confidences = {}
        for i, prob in enumerate(y_pred_probs[0].cpu().numpy()):
            confidences[classes[i]] = round(float(prob) * 100, 3)

        return classes[int(y_pred_class.item())], confidences, inference_time
