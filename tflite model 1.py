import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define the directory containing subdirectories of plant images
data_directory = 'C:/Users/Maveen Balaji/OneDrive/Desktop/python/medical plants/Medicinal Leaf Dataset/Segmented Medicinal Leaf Images'

# Define image dimensions+
image_size = (224, 224)  # Adjust this based on your model's input size requirements

# Define batch size
batch_size = 25 # You can adjust this based on your system's memory

# Updated class labels for the dataset
class_labels = {
    0: 'Alpinia Galanga (Rasna)',
    1: 'Amaranthus Viridis (Arive-Dantu)',
    2: 'Artocarpus Heterophyllus (Jackfruit)',
    3: 'Azadirachta Indica (Neem)',
    4: 'Basella Alba (Basale)',
    5: 'Brassica Juncea (Indian Mustard)',
    6: 'Carissa Carandas (Karanda)',
    7: 'Citrus Limon (Lemon)',
    8: 'Ficus Auriculata (Roxburgh fig)',
    9: 'Ficus Religiosa (Peepal Tree)',
    10: 'Hibiscus Rosa-sinensis',
    11: 'Jasminum (Jasmine)',
    12: 'Mangifera Indica (Mango)',
    13: 'Mentha (Mint)',
    14: 'Moringa Oleifera (Drumstick)',
    15: 'Muntingia Calabura (Jamaica Cherry-Gasagase)',
    16: 'Murraya Koenigii (Curry)',
    17: 'Nerium Oleander (Oleander)',
    18: 'Nyctanthes Arbor-tristis (Parijata)',
    19: 'Ocimum Tenuiflorum (Tulsi)',
    20: 'Piper Betle (Betel)',
    21: 'Plectranthus Amboinicus (Mexican Mint)',
    22: 'Pongamia Pinnata (Indian Beech)',
    23: 'Psidium Guajava (Guava)',
    24: 'Punica Granatum (Pomegranate)',
    25: 'Santalum Album (Sandalwood)',
    26: 'Syzygium Cumini (Jamun)',
    27: 'Syzygium Jambos (Rose Apple)',
    28: 'Tabernaemontana Divaricata (Crape Jasmine)',
    29: 'Trigonella Foenum-graecum (Fenugreek)'
}

# Create an ImageDataGenerator for data augmentation and normalization
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create a data generator for the single directory
data_generator = datagen.flow_from_directory(
    data_directory,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Load the pre-trained MobileNetV2 model without the top (fully connected) layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of the pre-trained model
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(class_labels), activation='softmax'))  # Use the number of classes from the dictionary

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    data_generator,
    steps_per_epoch=len(data_generator),
    epochs=5,  # Adjust the number of epochs as needed
)

# Save the trained model in the SavedModel format
saved_model_dir = 'saved_model'
tf.saved_model.save(model, saved_model_dir)

# Convert the SavedModel to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
tflite_model_file = 'plant_identification_model.tflite'
with open(tflite_model_file, 'wb') as f:
    f.write(tflite_model)

# Function to perform inference on an input image using the TensorFlow Lite model
def predict_tflite(input_image_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(input_image_path, target_size=image_size)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = datagen.standardize(img)  # Normalize pixel values

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(output_data)

    return predicted_class_index

# Example usage: Provide the path to the input image
input_image_path = "C:/Users/Maveen Balaji/OneDrive/Desktop/python/medical plants/Medicinal Leaf Dataset/Segmented Medicinal Leaf Images/Mangifera Indica (Mango)/MI-S-029.jpg"

# Predict the class index using the TensorFlow Lite model
predicted_class_index = predict_tflite(input_image_path)

# Print the predicted class label
predicted_class_label = class_labels[predicted_class_index]
print(f'The predicted class is: {predicted_class_label}')