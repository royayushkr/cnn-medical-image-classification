# Pneumonia Detection from Chest X-Rays using CNN (ResNet-18)

## Project Overview
This project utilizes Deep Learning to automate the detection of pneumonia from chest X-ray images. Leveraging Transfer Learning with a pre-trained ResNet-18 architecture, the model is fine-tuned to classify X-rays into two categories: **NORMAL** and **PNEUMONIA**.

The system is intended to function as a clinical decision-support tool, assisting medical professionals in prioritizing and reviewing chest radiographs more efficiently.

## Project Details

### Description
This is a binary image classification pipeline built using PyTorch. It analyzes chest X-ray images to predict the presence of pneumonia. The model benefits from feature representations previously learned on the ImageNet dataset, adapting them to the specific nuances of the medical imaging domain.

### Motivation
Pneumonia is a potentially life-threatening respiratory condition that requires timely diagnosis. Traditional manual X-ray interpretation faces several challenges:
* **Time Constraints:** The process requires expert radiologist review, which can be a bottleneck in urgent care.
* **Subjectivity:** Diagnoses can be affected by observer fatigue and inter-observer variability.
* **Scalability:** There is often a limited availability of specialists in many regions, making manual review hard to scale.

## Methodology

The project is implemented in a Jupyter Notebook and follows a structured deep learning workflow:

### 1. Data Preprocessing
To ensure compatibility with the pre-trained model, the following transformations are applied:
* **Resizing:** Images are resized to 224 x 224 pixels.
* **Cropping:** Center cropping is applied to emphasize the lung regions.
* **Normalization:** Images are normalized using the ImageNet mean and standard deviation to match the pre-trained weights.

### 2. Model Architecture (ResNet-18)
* **Base Model:** `torchvision.models.resnet18(pretrained=True)`
* **Modifications:** The final fully connected layer (originally 1000 classes) is replaced to accommodate the specific binary classification task (2 classes: NORMAL, PNEUMONIA).

### 3. Training Configuration
* **Loss Function:** CrossEntropyLoss
* **Optimizer:** Stochastic Gradient Descent (SGD)
    * Learning Rate: 0.001
    * Momentum: 0.9
* **Scheduler:** StepLR
    * Decay: Learning rate decays by a factor of 0.1 every 7 epochs.

## Dataset Structure
The dataset requires the standard `ImageFolder` structure. Ensure your directory is organized as follows:

```text
root/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
```

## Tech Stack
* **Language:** Python 3.x
* **Deep Learning Framework:** PyTorch, Torchvision
* **Data Manipulation:** NumPy
* **Visualization:** Matplotlib
* **Environment:** Jupyter Notebook

## License
This project is licensed under the MIT License and is open for use, modification, and distribution.
