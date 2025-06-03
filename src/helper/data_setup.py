import torch
import torchvision
from torch import nn

from torchvision import datasets
import random
from pathlib import Path
import shutil
from tqdm import tqdm

def get_classnames() ->list[str]:
    return [
        "apple_pie",
        "baby_back_ribs",
        "baklava",
        "beef_carpaccio",
        "beef_tartare",
        "beet_salad",
        "beignets",
        "bibimbap",
        "bread_pudding",
        "breakfast_burrito",
        "bruschetta",
        "caesar_salad",
        "cannoli",
        "caprese_salad",
        "carrot_cake",
        "ceviche",
        "cheesecake",
        "cheese_plate",
        "chicken_curry",
        "chicken_quesadilla",
        "chicken_wings",
        "chocolate_cake",
        "chocolate_mousse",
        "churros",
        "clam_chowder",
        "club_sandwich",
        "crab_cakes",
        "creme_brulee",
        "croque_madame",
        "cup_cakes",
        "deviled_eggs",
        "donuts",
        "dumplings",
        "edamame",
        "eggs_benedict",
        "escargots",
        "falafel",
        "filet_mignon",
        "fish_and_chips",
        "foie_gras",
        "french_fries",
        "french_onion_soup",
        "french_toast",
        "fried_calamari",
        "fried_rice",
        "frozen_yogurt",
        "garlic_bread",
        "gnocchi",
        "greek_salad",
        "grilled_cheese_sandwich",
        "grilled_salmon",
        "guacamole",
        "gyoza",
        "hamburger",
        "hot_and_sour_soup",
        "hot_dog",
        "huevos_rancheros",
        "hummus",
        "ice_cream",
        "lasagna",
        "lobster_bisque",
        "lobster_roll_sandwich",
        "macaroni_and_cheese",
        "macarons",
        "miso_soup",
        "mussels",
        "nachos",
        "omelette",
        "onion_rings",
        "oysters",
        "pad_thai",
        "paella",
        "pancakes",
        "panna_cotta",
        "peking_duck",
        "pho",
        "pizza",
        "pork_chop",
        "poutine",
        "prime_rib",
        "pulled_pork_sandwich",
        "ramen",
        "ravioli",
        "red_velvet_cake",
        "risotto",
        "samosa",
        "sashimi",
        "scallops",
        "seaweed_salad",
        "shrimp_and_grits",
        "spaghetti_bolognese",
        "spaghetti_carbonara",
        "spring_rolls",
        "steak",
        "strawberry_shortcake",
        "sushi",
        "tacos",
        "takoyaki",
        "tiramisu",
        "tuna_tartare",
        "waffles"
        ]



def create_custom_data(data_loc: Path, train_loc: Path, test_loc: Path, classes: list, size: int = 40, train_test_split: int = 75) -> None:
    """
    This function is used to create a custom data for using in PyTorch's ImageFolder.
    Function will randomly select the size% of data given byt the user.
    It will then split that data into train and test folders based on the train_test_split value given by the user
    """
    for classname in classes:
        #setup the individual class path
        class_path = Path(data_loc/classname)
        #get all the files from a class_path
        files = list(class_path.rglob("*"))
        #randomly select size% of the files
        random_files = random.sample(files, int((size/100)*len(files)))

        #split the random file to train and test
        train_images = random_files[:int(train_test_split/100 * len(random_files))]
        print(f"train images for {classname}: {len(train_images)}")
        test_images = random_files[int(train_test_split/100 * len(random_files)):]
        print(f"test images for {classname}: {len(test_images)}\n")

        #copy the images to respective folders
        for image in tqdm(train_images, desc=f"    Train [{classname}]", unit="img"):
            parent_folder = image.parent.name
            
            #copy it into train_loc/parent_folder
            dest_path = train_loc/parent_folder
            dest_path.mkdir(parents=True, exist_ok=True)
        
            shutil.copy2(image, dest_path)
        print(f"Completed creating {dest_path}\n")
        
        #copy the test images to respective filder
        for image in tqdm(test_images, desc=f"    Test [{classname}]", unit="img"):
            parent_folder = image.parent.name

            dest_path = test_loc/parent_folder
            dest_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, dest_path)
        print(f"Completed creating custom {dest_path}")
        

def create_custom_dirs(base_dir):
    """
    Creates custom train and test directories under the given base_dir.
    Returns the paths to train and test directories.
    """
    custom_train_loc = base_dir / "train"
    custom_test_loc = base_dir / "test"
    base_dir.mkdir(exist_ok=True)
    
    custom_train_loc.mkdir(parents=True, exist_ok=True)
    custom_test_loc.mkdir(parents=True, exist_ok=True)
    print(f"created new directory: {custom_test_loc}\n")
    print(f"created new directory: {custom_train_loc}")
    return custom_train_loc, custom_test_loc