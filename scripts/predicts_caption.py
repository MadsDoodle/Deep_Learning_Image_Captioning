import numpy as np
import os
import sys
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.sequence import pad_sequences
import json # For loading tokenizer
from keras.models import load_model # For loading the trained model


def load_tokenizer_from_json(path):
    """Loads a Keras Tokenizer from a JSON file."""
    import json
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    with open(path, 'r', encoding='utf-8') as f:
        tokenizer_json = json.load(f)
    return tokenizer_from_json(tokenizer_json)


def predict_caption(image_feature, model, tokenizer, maxlen, index_word):
    """
    Generates a caption for a given image feature vector.
    """
    in_text = 'startseq'
    # Reshape image_feature to (1, num_features) for model.predict
    image_feature_reshaped = image_feature.reshape(1, len(image_feature))

    for iword in range(maxlen):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=maxlen, padding='post')
        
        yhat = model.predict([image_feature_reshaped, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        
        # Handle cases where predicted index might be out of vocab_size
        # (Though unlikely if model is trained well and index_word is correct)
        newword = index_word.get(yhat_index, "") # Use .get() for safer access

        in_text += " " + newword

        if newword == "endseq":
            break
    return in_text.strip() # .strip() to clean up leading/trailing spaces


def display_predictions(fnm_test, di_test, model, tokenizer, maxlen, index_word, dir_Flickr_jpg, num_examples=5):
    """
    Displays sample images and their predicted captions.
    """
    target_size = (224, 224, 3) # Consistent with VGG16 input
    npic = num_examples

    print(f"\nGenerating and displaying {num_examples} sample captions from test set:")
    for i in range(num_examples):
        jpgfnm = fnm_test[i]
        image_feature = di_test[i]

        filename = os.path.join(dir_Flickr_jpg, jpgfnm)
        try:
            image_load = load_img(filename, target_size=target_size)
        except Exception as e:
            print(f"Error loading image {filename} for display: {e}")
            continue
        
        caption = predict_caption(image_feature, model, tokenizer, maxlen, index_word)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6)) # Create a figure with 2 subplots
        
        # Plot image
        axes[0].imshow(image_load)
        axes[0].set_title(f"Image: {jpgfnm}", fontsize=10)
        axes[0].axis('off')

        # Plot caption
        axes[1].text(0.05, 0.5, caption, fontsize=12, wrap=True, va='center')
        axes[1].set_title("Predicted Caption", fontsize=10)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt # Import here for display_predictions

    # Paths to saved model and tokenizer (adjust if different)
    model_path = os.path.join("models", "image_captioning_model.keras")
    tokenizer_path = os.path.join("models", "tokenizer.json")

    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        print(f"Model or tokenizer not found. Please run scripts/train_model.py first.")
        sys.exit(1)

    print("Loading trained model and tokenizer...")
    loaded_model = load_model(model_path)
    loaded_tokenizer = load_tokenizer_from_json(tokenizer_path)

    # Reconstruct index_word mapping
    index_word = {idx: word for word, idx in loaded_tokenizer.word_index.items()}

    # You need maxlen from the training phase. If not saved, you might need to re-derive it
    # from your tokenized test captions, or save it during training.
    # For simplicity, let's assume a dummy maxlen or get it from a configuration if available.
    # In a real application, you'd save/load this with your model/tokenizer.
    # For now, we'll try to infer it from the test data from `train_model.py`
    # or use a fixed value if it's constant. Let's assume the maxlen was 30 as in the notebook.
    maxlen_from_training = 30 # This should ideally be loaded or passed from training

    # Re-load test data for prediction, similar to how it was loaded in train_model.py
    # This is a bit redundant but ensures the prediction script is self-contained for test data.
    # In a production setup, you'd likely load a single image and its feature.

    # Assuming dataset is in the default KaggleHub cache path
    default_dataset_path = os.path.expanduser("~/.cache/kagglehub/datasets/shadabhussain/flickr8k/versions/3")
    if not os.path.exists(default_dataset_path):
        print(f"Dataset not found at {default_dataset_path}. Cannot load test data for prediction display.")
        sys.exit(1)

    dir_Flickr_jpg = os.path.join(default_dataset_path, 'flickr_data', 'Flickr_Data', 'Images')
    dir_Flickr_text = os.path.join(default_dataset_path, 'flickr_data', 'Flickr_Data', 'Flickr_TextData')

    # Load captions to get full df_txt for filtering
    datatxt_full = []
    with open(os.path.join(dir_Flickr_text, 'Flickr8k.token.txt'), 'r', encoding='utf8') as file:
        text_full = file.read()
    for line in text_full.split('\n'):
        col = line.split('\t')
        if len(col) == 1:
            continue
        w = col[0].split("#")
        datatxt_full.append(w + [col[1].lower()])
    df_txt_full = pd.DataFrame(datatxt_full, columns=["filename", "index", "caption"])

    # Load VGG16 to extract features for _all_ images (or just test images if preferred)
    from models.vgg16_model import load_vgg16_feature_extractor
    modelvgg_for_prediction = load_vgg16_feature_extractor()
    all_images_features_dict = load_and_extract_image_features(dir_Flickr_jpg, modelvgg_for_prediction)

    # Filter df_txt to get unique first captions that have features
    df_txt_filtered_for_test = df_txt_full[df_txt_full['filename'].isin(all_images_features_dict.keys())].copy()
    df_txt_filtered_for_test = df_txt_filtered_for_test.loc[df_txt_filtered_for_test["index"] == "0", :]

    # Clean and add start/end tokens
    from utils.text_cleaning import text_clean, add_start_end_seq_token
    for i, caption in enumerate(df_txt_filtered_for_test.caption.values):
        cleaned_caption = text_clean(caption)
        df_txt_filtered_for_test.at[df_txt_filtered_for_test.index[i], "caption"] = cleaned_caption
    df_txt_filtered_for_test["caption"] = add_start_end_seq_token(df_txt_filtered_for_test["caption"])
    
    # Tokenize captions for test set (need dtexts to split)
    from utils.data_preprocessing import create_tokenizer_and_sequences, split_test_val_train
    # Re-use the loaded_tokenizer for consistency if possible, otherwise re-create
    _, _, dtexts_full, _ = create_tokenizer_and_sequences(df_txt_filtered_for_test, nb_words=6000) # Re-tokenize to get dtexts for splitting

    # Re-get the split numbers from the total data size
    N_full = len(dtexts_full)
    prop_test, prop_val = 0.2, 0.2
    Ntest_full, Nval_full = int(N_full*prop_test), int(N_full*prop_val)

    # Get the test set data (this needs to be consistent with train_model.py's split)
    dt_test_for_pred, _, _ = split_test_val_train(dtexts_full, Ntest_full, Nval_full)
    di_test_for_pred, _, _ = split_test_val_train(np.array([all_images_features_dict[fnm] for fnm in df_txt_filtered_for_test['filename'].values]), Ntest_full, Nval_full)
    fnm_test_for_pred, _, _ = split_test_val_train(df_txt_filtered_for_test['filename'].values, Ntest_full, Nval_full)


    display_predictions(fnm_test_for_pred, di_test_for_pred, loaded_model, loaded_tokenizer, maxlen_from_training, index_word, dir_Flickr_jpg, num_examples=5)