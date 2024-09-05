Plant Leaf Identification Model

Overview

This project involves a machine learning model built using TensorFlow and Keras to identify plant species from leaf images. The model utilizes the MobileNetV2 architecture as a base, fine-tuned with a custom dataset of medicinal leaf images. The model is trained, saved in TensorFlow SavedModel format, and converted to TensorFlow Lite for efficient inference on edge devices.

Features

Leaf Image Classification: Identifies plant species from leaf images.

Data Augmentation: Utilizes data augmentation techniques to improve model robustness.

Model Conversion: Converts the trained model to TensorFlow Lite for deployment on mobile and edge devices.

Inference: Provides functions to predict the plant species from a given leaf image.

Tech Stack

TensorFlow: For building, training, and deploying the machine learning model.

Keras: High-level API used for defining and training the model.

NumPy: For numerical operations and data manipulation.

PIL (Python Imaging Library): For image processing tasks.

MobileNetV2: Pre-trained model used as a base for transfer learning.

Dataset

The model is trained on a dataset of medicinal plant leaves. The dataset directory structure should include subdirectories for each plant species, with images of leaves in each subdirectory.

How It Works

Data Preparation:

Images are loaded from a specified directory.

Data augmentation is applied to improve model generalization.

Model Architecture:

Uses MobileNetV2 as the base model (pre-trained on ImageNet) with added classification layers.

The top layers of MobileNetV2 are removed, and custom dense layers are added for plant classification.

Model Training:

The model is compiled with Adam optimizer and categorical crossentropy loss.

It is trained for a specified number of epochs.

Model Conversion:

After training, the model is saved in TensorFlow SavedModel format.

The model is then converted to TensorFlow Lite format for efficient inference.

Inference:

The TensorFlow Lite model is used to predict the class of a given leaf image.

The prediction is mapped to the corresponding plant species label.
