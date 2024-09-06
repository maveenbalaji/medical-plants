# Plant Leaf Identification Model

**Overview**

This project involves a machine learning model built using TensorFlow and Keras to identify plant species from leaf images. The model utilizes the MobileNetV2 architecture as a base, fine-tuned with a custom dataset of medicinal leaf images. The model is trained, saved in TensorFlow SavedModel format, and converted to TensorFlow Lite for efficient inference on edge devices.

## Features

- **Leaf Image Classification**: Identifies plant species from leaf images.
- **Data Augmentation**: Utilizes data augmentation techniques to improve model robustness.
- **Model Conversion**: Converts the trained model to TensorFlow Lite for deployment on mobile and edge devices.
- **Inference**: Provides functions to predict the plant species from a given leaf image.

## Tech Stack

- **TensorFlow**: For building, training, and deploying the machine learning model.
- **Keras**: High-level API used for defining and training the model.
- **NumPy**: For numerical operations and data manipulation.
- **PIL (Python Imaging Library)**: For image processing tasks.
- **MobileNetV2**: Pre-trained model used as a base for transfer learning.

## Dataset

The model is trained on a dataset of medicinal plant leaves. The dataset directory structure should include subdirectories for each plant species, with images of leaves in each subdirectory.

## How It Works

1. **Data Preparation**:
   - Images are loaded from a specified directory.
   - Data augmentation is applied to improve model generalization.

2. **Model Architecture**:
   - Uses MobileNetV2 as the base model (pre-trained on ImageNet) with added classification layers.
   - The top layers of MobileNetV2 are removed, and custom dense layers are added for plant classification.

3. **Model Training**:
   - The model is compiled with Adam optimizer and categorical crossentropy loss.
   - It is trained for a specified number of epochs.

4. **Model Conversion**:
   - After training, the model is saved in TensorFlow SavedModel format.
   - The model is then converted to TensorFlow Lite format for efficient inference.

5. **Inference**:
   - The TensorFlow Lite model is used to predict the class of a given leaf image.
   - The prediction is mapped to the corresponding plant species label.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/maveenbalaji/plant-leaf-dection-ml-model.git
   ```

2. Navigate to the project directory:
   ```bash
   cd plant-leaf-identification-model
   ```

3. Install dependencies:
   ```bash
   pip install tensorflow numpy pillow
   ```

## Training the Model

1. Update the `data_directory` variable in the code to point to your dataset directory.
2. Adjust the `batch_size` and `epochs` as needed based on your system’s resources.
3. Run the training script:
   ```bash
   python train_model.py
   ```

## Using the Model for Inference

1. Ensure that the TensorFlow Lite model file (`plant_identification_model.tflite`) is present in the project directory.
2. Update the `input_image_path` variable in the code with the path to the leaf image you want to classify.
3. Run the inference script:
   ```bash
   python infer_model.py
   ```

## Example Usage

```python
input_image_path = "path/to/leaf/image.jpg"
predicted_class_index = predict_tflite(input_image_path)
predicted_class_label = class_labels[predicted_class_index]
print(f'The predicted class is: {predicted_class_label}')
```

## Future Improvements

- **Expand Dataset**: Include more plant species and images for better accuracy.
- **Model Fine-Tuning**: Experiment with different architectures and hyperparameters.
- **User Interface**: Develop a mobile or web app for interactive plant identification.


