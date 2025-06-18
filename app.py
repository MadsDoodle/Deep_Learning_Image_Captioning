import streamlit as st
import PIL.Image # Used by Streamlit for image display
import numpy as np
import os
import tensorflow as tf
import keras # Keras is now part of TensorFlow 2.x
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model # For loading the trained model
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json # Specific for loading tokenizer

# --- Configuration & Paths ---
# Define paths relative to the app.py file
MODEL_DIR = "models"
CAPTIONING_MODEL_PATH = os.path.join(MODEL_DIR, "image_captioning_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")

# --- Streamlit UI Basic Setup ---
st.set_page_config(
    page_title="Image Captioning App",
    layout="centered", # Centered layout for a clean look
    initial_sidebar_state="auto" # Sidebar state
)

st.title("📸 AI Image Captioner")
st.markdown("""
    Upload an image, and our AI model will generate a descriptive caption for it!
    ---
""")

# --- Model Loading (Cached for Performance) ---
# Use st.cache_resource to load heavy models only once when the app starts.
@st.cache_resource
def load_all_resources():
    """
    Loads the VGG16 feature extractor, the trained image captioning model,
    and the tokenizer. These resources are cached to avoid re-loading on
    every user interaction.
    """
    try:
        # Load VGG16 Feature Extractor
        # We re-create the VGG16 model here, popping the last layer,
        # ensuring it's consistent with how features were extracted during training.
        st.write("Loading VGG16 Feature Extractor...")
        vgg_model = tf.keras.applications.VGG16(weights='imagenet')
        vgg_model.layers.pop() # Remove the last classification layer
        vgg_model = tf.keras.models.Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-1].output)
        st.success("VGG16 Feature Extractor Loaded!")

        # Load Trained Image Captioning Model
        if not os.path.exists(CAPTIONING_MODEL_PATH):
            st.error(f"Error: Captioning model not found at '{CAPTIONING_MODEL_PATH}'.")
            st.warning("Please ensure you have run `scripts/train_model.py` to train and save the model.")
            return None, None, None, None, None
        
        st.write("Loading Image Captioning Model...")
        caption_model = load_model(CAPTIONING_MODEL_PATH)
        st.success("Image Captioning Model Loaded!")

        # Load Tokenizer
        if not os.path.exists(TOKENIZER_PATH):
            st.error(f"Error: Tokenizer not found at '{TOKENIZER_PATH}'.")
            st.warning("Please ensure you have run `scripts/train_model.py` to train and save the tokenizer.")
            return None, None, None, None, None
        
        st.write("Loading Tokenizer...")
        with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
            tokenizer_json = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_json)
        st.success("Tokenizer Loaded!")

        # Reconstruct index_word mapping for decoding predictions
        index_word = {idx: word for word, idx in tokenizer.word_index.items()}
        
        # Max caption length (fixed from training, assuming 30 from the notebook analysis)
        # In a more robust system, this would be saved/loaded alongside the model metadata.
        maxlen = 30 
        
        return vgg_model, caption_model, tokenizer, index_word, maxlen
    except Exception as e:
        st.error(f"An error occurred during model or tokenizer loading: {e}")
        st.exception(e) # Display full traceback for debugging
        return None, None, None, None, None

# Load all resources at app startup
vgg_model, caption_model, tokenizer, index_word, maxlen = load_all_resources()

# --- Caption Generation Logic ---
def generate_caption(image_path, vgg_model_ref, caption_model_ref, tokenizer_ref, index_word_ref, maxlen_ref):
    """
    Generates a caption for a given image file path using the loaded models.
    """
    # Check if models were loaded successfully
    if vgg_model_ref is None or caption_model_ref is None or tokenizer_ref is None:
        return "Error: Core models/tokenizer not loaded. Cannot generate caption."

    target_size = (224, 224, 3) # Input size for VGG16

    # 1. Load and Preprocess Image for VGG16
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array) # VGG16 specific preprocessing
        # Predict image features using the VGG16 model
        image_feature = vgg_model_ref.predict(img_array.reshape((1,) + img_array.shape[:3]), verbose=0).flatten()
    except Exception as e:
        return f"Error during image feature extraction: {e}"

    # 2. Generate Caption using the Trained Captioning Model
    in_text = 'startseq' # Start sequence token
    for _ in range(maxlen_ref): # Iterate up to maxlen tokens
        # Convert current sequence to token IDs
        sequence = tokenizer_ref.texts_to_sequences([in_text])[0]
        # Pad sequence. Padding must be consistent with training ('post').
        sequence = pad_sequences([sequence], maxlen=maxlen_ref, padding='post')
        
        # Predict the next word's probability distribution
        # Reshape image_feature to (1, num_features) for model.predict
        yhat = caption_model_ref.predict([image_feature.reshape(1, len(image_feature)), sequence], verbose=0)
        yhat_index = np.argmax(yhat) # Get the index of the word with highest probability
        
        # Get the word from its index, handling potential unknown words gracefully
        newword = index_word_ref.get(yhat_index, "") 
        
        # Break conditions: if end sequence token is predicted or an unknown word is generated
        if newword == "endseq" or newword == "":
            break

        in_text += " " + newword # Append the new word to the sequence
        
    # 3. Clean up the generated caption (remove start/end tokens)
    final_caption = in_text.replace("startseq ", "").replace(" endseq", "").strip()
    return final_caption if final_caption else "No caption could be generated. Try another image."


# --- Streamlit UI Layout and Interaction ---

# Sidebar for instructions
st.sidebar.header("Setup & Usage")
st.sidebar.markdown("""
1.  **Initial Setup:**
    * Make sure you have the project structure (models/, utils/, scripts/).
    * Run `pip install -r requirements.txt` (ensure `streamlit`, `tensorflow`, `Pillow`, `nltk` are included).
    * Set up your Kaggle API key (create `~/.kaggle/kaggle.json`).
2.  **Data & Model Preparation:**
    * Run `python scripts/download_data.py` to get the Flickr8k dataset.
    * Run `python scripts/train_model.py` to train the captioning model.
        *(This will save `image_captioning_model.keras` and `tokenizer.json` in the `models/` directory, which this app uses).*
3.  **Run this App:**
    * Navigate to your project's root directory in the terminal.
    * Run `streamlit run app.py`
""")

# File Uploader component
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("") # Add a little space for visual separation

    # Save the uploaded file temporarily so Keras can load it by path
    temp_image_path = "uploaded_temp_image.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Button to trigger caption generation
    if st.button("Generate Caption", help="Click to get a caption for the uploaded image"):
        # Check if models were loaded successfully before proceeding
        if vgg_model and caption_model and tokenizer and index_word and maxlen:
            with st.spinner("Generating caption... This might take a moment."):
                predicted_caption = generate_caption(
                    temp_image_path, vgg_model, caption_model, tokenizer, index_word, maxlen
                )
                
                # Display the result
                if "Error:" in predicted_caption or "No caption" in predicted_caption:
                    st.error(predicted_caption)
                else:
                    st.success("Caption Generated!")
                    # Use markdown for formatting the output as requested
                    st.markdown(f"### Predicted Caption:\n\n**_'{predicted_caption}'_**")
        else:
            st.error("Cannot generate caption: Models/tokenizer failed to load. Please check initial setup and console output.")

    # Clean up the temporary image file after processing (or after app re-run)
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)
else:
    st.info("Please upload an image to get started. Accepted formats: JPG, JPEG, PNG.")

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit and TensorFlow.")