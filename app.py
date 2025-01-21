import os
import numpy as np
from PIL import Image, ImageOps
import google.generativeai as gen_ai
from tensorflow.keras.models import load_model
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Replace with your Gemini API key (you would need access to Google Cloud APIs)
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')

# Function to classify waste
def classify_waste(img):
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("trained model/model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Prepare the image for classification
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img.convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Predict with the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name.strip(), confidence_score

# Function to generate carbon emission information from a specific label
def generate_carbon_footprint_info(label):
    # Ensure the API key is loaded correctly
    if not GEMINI_API_KEY:
        st.error("API key not found. Please ensure the GOOGLE_API_KEY is set in your .env file.")
        st.stop()

    # Set up Google Gemini-Pro AI model
    gen_ai.configure(api_key=GEMINI_API_KEY)
    model = gen_ai.GenerativeModel('gemini-pro')

    # Define the prompt to ask about carbon emission (with a request for 100 words)
    prompt = f"What is the approximate carbon emission or carbon footprint generated from {label}? I just need an approximate number to create awareness. Elaborate in 100 words.\n"

    # Send the prompt to the Gemini model and get the response
    gemini_response = model.generate_content(contents=prompt)

    # Limit the response to the first 100 words
    response_text = gemini_response.text
    word_limit = 100
    truncated_response = ' '.join(response_text.split()[:word_limit])

    return truncated_response

# Streamlit page setup with custom CSS for dark theme
st.set_page_config(layout='wide')
st.title("Waste Classifier Sustainability App")

# Inject custom CSS for black background and white text
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
        }
        .stButton>button {
            background-color: #6200ea;
            color: white;
            border-radius: 5px;
        }
        .stImage {
            border: 3px solid white;
        }
        .stFileUploader {
            background-color: #333333;
            border: 1px solid #6200ea;
        }
        .stInfo {
            background-color: #333333;
            border: 1px solid #6200ea;
        }
        .stSuccess {
            background-color: #4caf50;
            color: white;
        }
        .stWarning {
            background-color: #ff9800;
            color: white;
        }
        .stError {
            background-color: #f44336;
            color: white;
        }
        .stTextInput>input {
            background-color: #333333;
            color: white;
            border: 1px solid #6200ea;
        }
        .stCameraInput {
            max-width: 300px;
            margin-left: 20px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Option to select image input type (Upload or Camera)
image_input_type = st.radio("Select input method", ("Browse", "Capture"))

if image_input_type == "Browse":
    input_img = st.file_uploader("Enter your image", type=['jpg', 'png', 'jpeg'])
else:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        input_img = st.camera_input("Capture image", key="camera_input", label_visibility="collapsed")

    # Resize the camera input box and place it inside the column
    st.markdown("""
        <style>
            .stCameraInput {
                max-width: 350px;
                margin-top: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

if input_img is not None:
    if st.button("Classify"):
        
        col1, col2, col3 = st.columns([1, 1, 1])

        # Column 1 - Image Display with Icon
        with col1:
            st.subheader("📷 Your uploaded Image")  # Added icon to title
            if image_input_type == "Browse":
                st.image(input_img, use_container_width=True)
            else:
                st.image(input_img, use_container_width=True)

        # Column 2 - Classification Result with Icon
        with col2:
            st.subheader("✅ Your Result")  # Added icon to title
            image_file = Image.open(input_img)
            label, confidence_score = classify_waste(image_file)
            col4, col5 = st.columns([1, 1])

            # Classification and SDG Goal Images
            if label == "0 Battery":
                st.success("The image is classified as BATTERY.")
                with col4:
                    st.image("sdg goals/12.png", use_container_width=True)
                    st.image("sdg goals/13.png", use_container_width=True)
                with col5:
                    st.image("sdg goals/14.png", use_container_width=True)
                    st.image("sdg goals/15.png", use_container_width=True)
            elif label == "1 Biological":
                st.success("The image is classified as BIOLOGICAL.")
                with col4:
                    st.image("sdg goals/6.png", use_container_width=True)
                    st.image("sdg goals/12.png", use_container_width=True)
                with col5:
                    st.image("sdg goals/14.png", use_container_width=True)
                    st.image("sdg goals/15.png", use_container_width=True)
            elif label == "2 Brown Glass":
                st.success("The image is classified as BROWN GLASS.")
                with col4:
                    st.image("sdg goals/12.png", use_container_width=True)
                    st.image("sdg goals/15.png", use_container_width=True)
                with col5:
                    st.image("sdg goals/14.png", use_container_width=True)
                    st.image("sdg goals/13.png", use_container_width=True)
            elif label == "3 Cardboard":
                st.success("The image is classified as CARDBOARD.")
                with col4:
                    st.image("sdg goals/3.png", use_container_width=True)
                    st.image("sdg goals/6.png", use_container_width=True)
                with col5:
                    st.image("sdg goals/12.png", use_container_width=True)
                    st.image("sdg goals/11.png", use_container_width=True)
            elif label == "4 Clothes":
                st.success("The image is classified as CLOTHES.")
                with col4:
                    st.image("sdg goals/12.png", use_container_width=True)
                    st.image("sdg goals/13.png", use_container_width=True)
                with col5:
                    st.image("sdg goals/14.png", use_container_width=True)
                    st.image("sdg goals/15.png", use_container_width=True)
            elif label == "5 Green Glass":
                st.success("The image is classified as GREEN GLASS.")
                with col4:
                    st.image("sdg goals/12.png", use_container_width=True)
                    st.image("sdg goals/15.png", use_container_width=True)
                with col5:
                    st.image("sdg goals/14.png", use_container_width=True)
                    st.image("sdg goals/13.png", use_container_width=True)
            elif label == "6 Metal":
                st.success("The image is classified as METAL.")
                with col4:
                    st.image("sdg goals/12.png", use_container_width=True)
                    st.image("sdg goals/13.png", use_container_width=True)
                with col5:
                    st.image("sdg goals/14.png", use_container_width=True)
                    st.image("sdg goals/15.png", use_container_width=True)
            elif label == "7 Paper":
                st.success("The image is classified as PAPER.")
                with col4:
                    st.image("sdg goals/12.png", use_container_width=True)
                    st.image("sdg goals/13.png", use_container_width=True)
                with col5:
                    st.image("sdg goals/14.png", use_container_width=True)
                    st.image("sdg goals/15.png", use_container_width=True)
            elif label == "8 Plastic":
                st.success("The image is classified as PLASTIC.")
                with col4:
                    st.image("sdg goals/12.png", use_container_width=True)
                    st.image("sdg goals/13.png", use_container_width=True)
                with col5:
                    st.image("sdg goals/14.png", use_container_width=True)
                    st.image("sdg goals/15.png", use_container_width=True)
            elif label == "9 Shoes":
                st.success("The image is classified as SHOES.")
                with col4:
                    st.image("sdg goals/12.png", use_container_width=True)
                    st.image("sdg goals/13.png", use_container_width=True)
                with col5:
                    st.image("sdg goals/14.png", use_container_width=True)
                    st.image("sdg goals/15.png", use_container_width=True)
            elif label == "10 Trash":
                st.success("The image is classified as TRASH.")
                with col4:
                    st.image("sdg goals/12.png", use_container_width=True)
                    st.image("sdg goals/13.png", use_container_width=True)
                with col5:
                    st.image("sdg goals/14.png", use_container_width=True)
                    st.image("sdg goals/15.png", use_container_width=True)
            elif label == "11 White Glass":
                st.success("The image is classified as WHITE GLASS.")
                with col4:
                    st.image("sdg goals/12.png", use_container_width=True)
                    st.image("sdg goals/15.png", use_container_width=True)
                with col5:
                    st.image("sdg goals/14.png", use_container_width=True)
                    st.image("sdg goals/13.png", use_container_width=True)
            else :
                st.success("The image is classified as Nothing.")




        # Column 3 - AI Response with Icon
        with col3:
            st.subheader("🤖 AI Carbon Emission Information")  # Title with an icon
            response = generate_carbon_footprint_info(label)
            st.success(response)
