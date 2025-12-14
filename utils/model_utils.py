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

def analyze_model(
        model: torch.nn.Module, 
        train_dataset: torch.utils.data.Dataset, 
        test_dataset: torch.utils.data.Dataset, 
        classes: list ,
        device: str
)-> None:
    """
    Analyzes the model on the training and test datasets
    Args:
        model: PyTorch model to analyze
        train_dataset: Training dataset
        test_dataset: Test dataset
        classes: List of classes
        device: Device to run the model on
    """
    correct = 0
    wrong = 0
    inf_time = 0
    for data in train_dataset:
      prediction, conf, inf = make_prediction(model, data,classes, torch.device(device))
      if prediction == classes[data[1]]:
        correct += 1
      else:
        wrong += 1
      inf_time += inf
    
    for data in test_dataset:
      prediction, conf, inf = make_prediction(model, data,classes, torch.device(device))
      if prediction == classes[data[1]]:
        correct += 1
      else:
        wrong += 1
      inf_time += inf

    print(f"total: {len(train_dataset) + len(test_dataset)}")
    print(f"correct: {correct}")
    print(f"Incorrect: {wrong}")
    print(f"Correct percent: {(correct*100)/(correct+wrong): .2f}")
    print(f"average inference time: {inf_time/(correct+wrong): .2f}")

def get_model_final_result(result: dict[str ,float])->dict[str, float]:
    """
    Gets the final result of the model
    Args:
        result: Dictionary containing the result of the model
    Returns:
        Dictionary containing the final result of the model
    """
    model_result = {}

    for i in result:
        if i == "time":
            model_result[i] = result[i]
        else:
            model_result[i] = result[i][-1] #type: ignore

    return model_result

def convert_state_dict_to_onnx(
      model: torch.nn.Module,
      state_dict_path: str,
      output_path: str,
      input_shape = (1, 3, 224, 224),
      opset_version = 17
):
   """
   Converts a PyTorch state_dict (.pth/.pt) to ONNX.

    Args:
        model (torch.nn.Module): Model architecture (must match state_dict)
        state_dict_path (str): Path to .pth/.pt state_dict file
        output_path (str): Destination path for .onnx file
        input_shape (tuple): Dummy input shape (default: ImageNet style)
        opset_version (int): ONNX opset version (>=17 recommended)
    Ouput:
        none
   """ 
   state_dict_path = Path(state_dict_path) #type: ignore
   onnx_output_path = Path(output_path)

   # Load weights from the pytorch model
   device = "cuda" if torch.cuda.is_available else "cpu"
   state_dict = torch.load(state_dict_path, map_location = device)
   model.load_state_dict(state_dict)
   model.eval()

   # create a dummy input of the given size
   dummy_input = torch.randn(*input_shape)

   # Export to onnx
   torch.onnx.export(
       model,
       dummy_input,
       onnx_output_path.as_posix(),
       export_params=True,
       opset_version=opset_version,
       do_constant_folding=True,
       input_names=["input"],
       output_names=["output"],
       dynamic_axes={
           "input": {0: "batch_size"},
           "output": {0: "batch_size"}
       },
       dynamo = False
    )
   
   print(f"Model saved to: {onnx_output_path}")

