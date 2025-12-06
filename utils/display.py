from matplotlib import pyplot as plt

import random
import math
import torch.utils.data

from typing import Optional

def plot_random_images(dataset: torch.utils.data.Dataset, num: int , seed: Optional[int] = None) -> None:
    """
    This function is used to randomly display some desired amount of data froma Dataset
    Args:
        input:
            dataset: torch.utils.data.Dataset
            num: number of images: for space reasons limited to 10
            seed: random seed if any
        output:
            None
    """

    if seed:
        random.seed(seed)
    cols = math.ceil(math.sqrt(num))
    rows = math.ceil(num/cols)
    if num>10:
        print(f"for visibility reasons nums is limited to 10. Setting to 10")
        num = 10
    random_indices = random.sample(range(len(dataset)), k=num)# type: ignore

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if num > 1 else [axes]
    for idx, rand_idx in enumerate(random_indices):
        image, label = dataset[rand_idx]
        # Unnormalize for display
        img = image.permute(1, 2, 0).cpu().numpy()
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # img = std * img + mean
        img = img.clip(0, 1)
        axes[idx].imshow(img)
        axes[idx].set_title(f"Label: {dataset.classes[label]}") # type: ignore
        axes[idx].axis("off")
    # Hide unused axes
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.show()

def plot_image(image: torch.Tensor , label: str) -> None:
    """
    Plots an image using matplotlib by using a image Tensor
    Args:
        image: (Tensor) The image to be displayed
        label: (str)
    """
    fig , ax = plt.subplots(nrows=1 , ncols=1)
    ax.set_title(label)
    img = image.permute(1,2,0).cpu().numpy()
    ax.imshow(img)
    ax.axis(False)

    plt.show()


def plot_train_vs_test_loss(result: dict[str, list]):
    """
    Plots train loss vs test loss across epochs from a result dictionary.
    Args:
        result: (dict) It must contain "train_loss" and "test_loss" keys with list of values
    """
    epochs = range(1, len(result["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, result["train_loss"], label="Train Loss", marker='', linestyle='-')
    plt.plot(epochs, result["test_loss"], label="Test Loss", marker='', linestyle='-')
    plt.title("Train vs Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compare_test_loss(model_1_result: dict, model_2_result: dict, model_1_label: str = "Model 1", model_2_label: str = "Model 2"):
    """
    Compares test loss between two models across epochs and visualizes the results.
    This function creates a line plot comparing the test loss trajectories of two models
    throughout their training epochs. It's useful for evaluating which model generalizes
    better on the test set and identifying potential overfitting or underfitting issues.
    Args:
        model_1_result (dict): A dictionary containing training results for the first model.
                               Must include a "test_loss" key with a list of loss values
                               for each epoch.
        model_2_result (dict): A dictionary containing training results for the second model.
                               Must include a "test_loss" key with a list of loss values
                               for each epoch.
        model_1_label (str, optional): Display label for the first model. Defaults to "Model 1".
        model_2_label (str, optional): Display label for the second model. Defaults to "Model 2".
    Returns:
        None: Displays the plot using matplotlib's plt.show().
    Raises:
        KeyError: If "test_loss" key is missing from either model_1_result or model_2_result.
        ValueError: If "test_loss" lists are empty.
    Example:
        >>> model_1 = {"test_loss": [0.5, 0.4, 0.3]}
        >>> model_2 = {"test_loss": [0.6, 0.45, 0.35]}
        >>> compare_test_loss(model_1, model_2, "ResNet", "VGG")
    """
    epochs_1 = range(1, len(model_1_result["test_loss"]) + 1)
    epochs_2 = range(1, len(model_2_result["test_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_1, model_1_result["test_loss"], label=f"{model_1_label} Test Loss", marker='o')
    plt.plot(epochs_2, model_2_result["test_loss"], label=f"{model_2_label} Test Loss", marker='s')
    plt.title("Test Loss Comparison Between Models")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compare_test_accuracy(model_1_result: dict, model_2_result: dict, model_1_label: str = "Model 1", model_2_label: str = "Model 2"):
    """
    Compares test accuracy between two models across epochs and visualizes the results.
    This function creates a line plot comparing the test accuracy of two models over training epochs.
    It allows for customizable model labels and displays the comparison using matplotlib with distinct
    markers for each model's accuracy curve.
    Args:
        model_1_result (dict): Dictionary containing the first model's results. Must include a 
                              "test_acc" key with a list/array of test accuracy values for each epoch.
        model_2_result (dict): Dictionary containing the second model's results. Must include a 
                              "test_acc" key with a list/array of test accuracy values for each epoch.
        model_1_label (str, optional): Label for the first model displayed in the legend. 
                                       Defaults to "Model 1".
        model_2_label (str, optional): Label for the second model displayed in the legend. 
                                       Defaults to "Model 2".
    Returns:
        None: Displays the comparison plot directly using matplotlib's plt.show().
    Raises:
        KeyError: If either model_result dictionary is missing the "test_acc" key.
        ValueError: If the "test_acc" values cannot be used to generate plot coordinates.
    Example:
        >>> model_1 = {"test_acc": [0.85, 0.88, 0.90, 0.92]}
        >>> model_2 = {"test_acc": [0.80, 0.84, 0.87, 0.89]}
        >>> compare_test_accuracy(model_1, model_2, "ResNet", "VGG")
    """
    
    epochs_1 = range(1, len(model_1_result["test_acc"]) + 1)
    epochs_2 = range(1, len(model_2_result["test_acc"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_1, model_1_result["test_acc"], label=f"{model_1_label} Test Acc", marker='o')
    plt.plot(epochs_2, model_2_result["test_acc"], label=f"{model_2_label} Test Acc", marker='s')
    plt.title("Test Accuracy Comparison Between Models")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    