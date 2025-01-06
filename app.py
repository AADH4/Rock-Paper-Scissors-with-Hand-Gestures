%%writefile app.py
import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image
import random

# Load TFLite model for inference (rock-paper-scissors model)
interpreter = tf.lite.Interpreter(model_path='https://drive.google.com/file/d/1o074xQK6h5zSeARdrtfrvcgivI5bnEZF/view?usp=drive_link') #Change path
interpre+96ter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set up the page title, description, and custom favicon
st.set_page_config(
    page_title="Rock Paper Scissors Game",
    page_icon=":guardsman",  # Specify the path to your custom favicon
    layout="wide"
)

st.title("Rock Paper Scissors Game")
st.markdown("""This app uses a trained model to predict whether the uploaded image is a **rock**, **paper**, or **scissors** gesture. You will play against the computer, which makes a random move. The game will show who won the round.""")

# Create a file uploader
uploaded_image = st.file_uploader("Upload a hand gesture image (jpg/png)", type=["jpg", "png"])

# Define game logic
def get_computer_move():
    return random.choice(['Rock', 'Paper', 'Scissors'])

def determine_winner(user_move, computer_move):
    if user_move == computer_move:
        return "It's a draw!"
    elif (user_move == 'Rock' and computer_move == 'Scissors') or \
         (user_move == 'Paper' and computer_move == 'Rock') or \
         (user_move == 'Scissors' and computer_move == 'Paper'):
        return "You win!"
    else:
        return "Computer wins!"

# Display uploaded image and result
if uploaded_image is not None:
    # Open the image with PIL
    image = Image.open(uploaded_image)

    # Resize the image to the input size required by the model (224x224)
    image_resized = image.resize((224, 224))  # Resize to 224x224 (expected input size)

    # Display the resized image
    st.image(image_resized, caption='Uploaded Image')

    # Preprocess the image for prediction
    image_resized = image_resized.convert('RGB')  # Ensure the image is in RGB format
    image_array = np.array(image_resized) / 255.0  # Normalize to 0-1 range

    # Prepare the image for model inference
    image_array = image_array.astype(np.float32)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Ensure the image has the correct shape (1, 224, 224, 3)
    expected_shape = (1, 224, 224, 3)
    if image_array.shape != expected_shape:
        st.error(f"Expected shape {expected_shape}, but got {image_array.shape}.")
    else:
        # Display a spinner while the model is processing
        with st.spinner("Classifying the image... Please wait."):
            try:
                # Set the tensor and invoke the model
                interpreter.set_tensor(input_details[0]['index'], image_array)
                interpreter.invoke()

                # Get the model's raw output (probabilities for rock, paper, and scissors)
                output = interpreter.get_tensor(output_details[0]['index'])

                # Get the predicted class (rock, paper, or scissors)
                predicted_class_index = np.argmax(output)  # Index of the maximum probability
                class_labels = ['Rock', 'Paper', 'Scissors']
                user_move = class_labels[predicted_class_index]

                # Display user's move
                st.success(f"Your move: **{user_move}**")

                # Get computer's move
                computer_move = get_computer_move()
                st.info(f"Computer's move: **{computer_move}**")

                # Determine the winner
                result = determine_winner(user_move, computer_move)
                st.success(result)

            except Exception as e:
                st.error(f"Error during inference: {e}")
else:
    st.info("Please upload an image to play the game.")

# Footer with app information
st.markdown("---")
st.markdown("Created by Abhijay Ahuja | Powered by Streamlit & TensorFlow Lite")
