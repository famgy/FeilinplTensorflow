import tensorflow as tf

# Load the model from the saved file
loaded_model = tf.keras.models.load_model("my_model.h5")


# Function to check handwritten digit
def check_handwritten_digit(image_path):
    # Load the image
    image = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale", target_size=(28, 28))
    input_data = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    input_data = input_data.reshape(1, 28, 28)

    # Make prediction
    prediction = loaded_model.predict(input_data)

    # Get the predicted class
    predicted_class = tf.argmax(prediction[0])

    return predicted_class
