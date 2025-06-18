import os
import sys
import warnings
import tensorflow as tf
import keras # Keras is now part of TensorFlow 2.x
import numpy as np
import pandas as pd
from collections import Counter
from time import time
from keras.callbacks import TensorBoard

# Import custom modules
from utils.text_cleaning import df_word, text_clean, add_start_end_seq_token
from utils.data_preprocessing import load_and_extract_image_features, create_tokenizer_and_sequences, split_test_val_train, preprocess_sequence_data
from utils.visualization import plot_training_history
from models.vgg16_model import load_vgg16_feature_extractor
from models.captioning_model import build_captioning_model

warnings.filterwarnings("ignore")

def train(dataset_base_path):
    print("Python version:", sys.version)
    print("TensorFlow version:", tf.__version__)
    # Keras version check is less relevant since it's integrated with TF2
    print("Keras version (from TensorFlow):", tf.keras.__version__) 

    ## Define data paths
    dir_Flickr_jpg = os.path.join(dataset_base_path, 'flickr_data', 'Flickr_Data', 'Images')
    dir_Flickr_text = os.path.join(dataset_base_path, 'flickr_data', 'Flickr_Data', 'Flickr_TextData')

    jpgs = os.listdir(dir_Flickr_jpg)
    print("The number of jpg files in Flicker8k:", len(jpgs))

    ## Load captions
    captions_file_path = os.path.join(dir_Flickr_text, 'Flickr8k.token.txt')
    datatxt = []
    with open(captions_file_path, 'r', encoding='utf8') as file:
        text = file.read()
    for line in text.split('\n'):
        col = line.split('\t')
        if len(col) == 1:
            continue
        w = col[0].split("#")
        datatxt.append(w + [col[1].lower()])
    df_txt = pd.DataFrame(datatxt, columns=["filename", "index", "caption"])
    
    uni_filenames = np.unique(df_txt.filename.values)
    print("The number of unique file names:", len(uni_filenames))
    print("The distribution of the number of captions for each image:")
    print(Counter(Counter(df_txt.filename.values).values()))

    # Clean captions
    print("\nCleaning captions...")
    for i, caption in enumerate(df_txt.caption.values):
        newcaption = text_clean(caption)
        df_txt.at[i, "caption"] = newcaption # Use .at for setting single values

    # Add start/end tokens
    df_txt0 = df_txt.copy() # Use .copy() to avoid SettingWithCopyWarning
    df_txt0["caption"] = add_start_end_seq_token(df_txt["caption"])
    print("\nCaptions after adding start/end tokens (first 5 rows):")
    print(df_txt0.head(5))

    # Load VGG16 feature extractor
    modelvgg = load_vgg16_feature_extractor()
    
    # Extract image features
    # Ensure dir_Flickr_jpg is the path where image files actually reside
    # The original notebook had a slightly complex path structure for images
    # Let's verify and correct if necessary.
    # From previous `os.walk` output, images are in `.../flickr_data/Flickr_Data/Images`
    images_features_dict = load_and_extract_image_features(dir_Flickr_jpg, modelvgg)

    # Filter df_txt0 to include only images for which features were extracted
    df_txt_filtered = df_txt0[df_txt0['filename'].isin(images_features_dict.keys())].copy()
    df_txt_filtered = df_txt_filtered.loc[df_txt_filtered["index"] == "0", :] # Only take the first caption for each image

    fnames = df_txt_filtered["filename"].values
    dcaptions = df_txt_filtered["caption"].values
    dimages = np.array([images_features_dict[fnm] for fnm in fnames])
    
    print("\nFiltered DataFrame for training (first 5 rows):")
    print(df_txt_filtered.head(5))
    print(f"Number of filtered images/captions: {len(fnames)}")
    print(f"Shape of extracted image features array: {dimages.shape}")

    # Create tokenizer and sequences
    nb_words = 6000
    tokenizer, vocab_size, dtexts, maxlen = create_tokenizer_and_sequences(df_txt_filtered, nb_words=nb_words)
    print("\nSample tokenized captions (first 5):")
    print(dtexts[:5])

    # Split data
    prop_test, prop_val = 0.2, 0.2
    N = len(dtexts)
    Ntest, Nval = int(N*prop_test), int(N*prop_val)

    dt_test, dt_val, dt_train = split_test_val_train(dtexts, Ntest, Nval)
    di_test, di_val, di_train = split_test_val_train(dimages, Ntest, Nval)
    fnm_test, fnm_val, fnm_train = split_test_val_train(fnames, Ntest, Nval) # Keep fnm_test for evaluation later

    # Preprocess data for model
    print("\nPreprocessing training data:")
    Xtext_train, Ximage_train, ytext_train = preprocess_sequence_data(dt_train, di_train, maxlen, vocab_size)
    print("\nPreprocessing validation data:")
    Xtext_val, Ximage_val, ytext_val = preprocess_sequence_data(dt_val, di_val, maxlen, vocab_size)

    # Build captioning model
    image_feature_shape = Ximage_train.shape[1] # This should be 1000 from VGG16
    caption_model = build_captioning_model(image_feature_shape, maxlen, vocab_size)
    caption_model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Train the model
    print("\nStarting model training...")
    log_dir = os.path.join("logs", str(int(time())))
    tensorboard_callback = TensorBoard(log_dir=log_dir)

    hist = caption_model.fit(
        [Ximage_train, Xtext_train],
        ytext_train,
        epochs=6,
        verbose=2,
        batch_size=32,
        validation_data=([Ximage_val, Xtext_val], ytext_val),
        callbacks=[tensorboard_callback]
    )
    print("Model training finished.")

    # Plot training history
    plot_training_history(hist)

    # Save the trained model and tokenizer
    model_save_path = os.path.join("models", "image_captioning_model.keras")
    tokenizer_save_path = os.path.join("models", "tokenizer.json")
    
    caption_model.save(model_save_path)
    import json
    tokenizer_json = tokenizer.to_json()
    with open(tokenizer_save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))
    
    print(f"\nModel saved to: {model_save_path}")
    print(f"Tokenizer saved to: {tokenizer_save_path}")
    
    # Return necessary components for evaluation/prediction
    return caption_model, tokenizer, maxlen, vocab_size, fnm_test, di_test, dt_test, dir_Flickr_jpg


if __name__ == "__main__":
    # This assumes `download_data.py` has been run and dataset_base_path is known.
    # For a full workflow, you might call download_data() here or pass the path.
    # For now, let's assume the default KaggleHub cache path for `flickr8k`.
    # You might need to adjust this if your download path is different.
    default_dataset_path = os.path.expanduser("~/.cache/kagglehub/datasets/shadabhussain/flickr8k/versions/3")
    
    if not os.path.exists(default_dataset_path):
        print(f"Dataset not found at {default_dataset_path}. Please run scripts/download_data.py first.")
        sys.exit(1)

    trained_model, trained_tokenizer, max_len, vocab_s, \
    test_filenames, test_image_features, test_tokenized_texts, images_directory = \
        train(default_dataset_path)