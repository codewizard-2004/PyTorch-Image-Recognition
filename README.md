# 🍽️ Food Recognition Models with PyTorch

Welcome to the FoodNet project! This repository contains state-of-the-art food recognition models built using PyTorch and trained on the Food101 dataset. Easily classify food images into 101 delicious categories with modern deep learning techniques.

---

## 🚀 Features
- **PyTorch-based**: Flexible, modular, and easy-to-extend codebase.
- **Food101 Dataset**: Trained and evaluated on the popular Food101 dataset.
- **Data Augmentation**: Advanced image augmentations for robust models.
- **Custom Architectures**: Includes TinyVGG and more.
- **Jupyter Notebooks**: Interactive experiments and visualizations.

---

## 📂 Project Structure
```
FoodNet/
|── data/                  # contains datasets
|── notebooks/             # various notebooks used
|── models/                # models created in pth form
|── utils                  # utility functions
├── requirements.txt       # Python dependencies
├── setup.py               # Project setup
└── README.md              # Project documentation
```

---

## 🥗 Dataset
- **Food101**: 101 food categories, 101,000 images.
- Downloaded and managed automatically via torchvision.
- Direct download URL (from official source):
   ```
   http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
   ```
   or

   ```bash 
   wget -c http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
   ```

---

## 🛠️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FoodNet.git
   cd FoodNet
   ```
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv mlvenv
   # On Windows:
   .\mlvenv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Usage
- Run and explore the Jupyter notebooks in `src/notebooks/` to train, evaluate, and visualize models.
- Example: `tinyvgg.ipynb` demonstrates training a TinyVGG model on Food101.

---

## 🧑‍💻 Author
- **Amal Varghese**  
  [officialamalv2004@gmail.com](mailto:officialamalv2004@gmail.com)

---

## ⭐️ Contributing
Pull requests, issues, and suggestions are welcome! Feel free to fork the repo and submit improvements.

---

## 📜 License
This project is licensed under the MIT License.

---

