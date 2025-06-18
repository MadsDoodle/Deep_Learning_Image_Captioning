import numpy as np
import os
import sys
import tensorflow as tf
from keras.models import load_model
from nltk.translate.bleu_score import sentence_bleu
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Import custom modules
from scripts.predict_caption import predict_caption, load_tokenizer_from_json # Re-use predict_caption
from utils.visualization import plot_predicted_captions, create_caption_string
from utils.text_cleaning import text_clean, add_start_end_seq_token
from utils.data_preprocessing import load_and_extract_image_features, create_tokenizer_and_sequences, split_test_val_train

def evaluate(model_path, tokenizer_path, dataset_base_path):
    print("Loading trained model and tokenizer for evaluation...")
    loaded_model = load_model(model_path)
    loaded_tokenizer = load_tokenizer_from_json(tokenizer_path)

    index_word = {idx: word for word, idx in loaded_tokenizer.word_index.items()}
    
    # maxlen needs to be consistent with training. If not saved, re-derive.
    # Assuming maxlen was 30 as in the notebook.
    maxlen = 30 

    # Re-load necessary data similar to train_model.py to ensure consistency
    dir_Flickr_jpg = os.path.join(dataset_base_path, 'flickr_data', 'Flickr_Data', 'Images')
    dir_Flickr_text = os.path.join(dataset_base_path, 'flickr_data', 'Flickr_Data', 'Flickr_TextData')

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
    modelvgg_for_evaluation = load_vgg16_feature_extractor()
    all_images_features_dict = load_and_extract_image_features(dir_Flickr_jpg, modelvgg_for_evaluation)

    # Filter df_txt to get unique first captions that have features
    df_txt_filtered_for_eval = df_txt_full[df_txt_full['filename'].isin(all_images_features_dict.keys())].copy()
    df_txt_filtered_for_eval = df_txt_filtered_for_eval.loc[df_txt_filtered_for_eval["index"] == "0", :]

    # Clean and add start/end tokens
    for i, caption in enumerate(df_txt_filtered_for_eval.caption.values):
        cleaned_caption = text_clean(caption)
        df_txt_filtered_for_eval.at[df_txt_filtered_for_eval.index[i], "caption"] = cleaned_caption
    df_txt_filtered_for_eval["caption"] = add_start_end_seq_token(df_txt_filtered_for_eval["caption"])
    
    # Tokenize captions for test set (need dtexts to split)
    # Re-use the loaded_tokenizer for consistency if possible, otherwise re-create
    _, _, dtexts_full, _ = create_tokenizer_and_sequences(df_txt_filtered_for_eval, nb_words=6000) # Re-tokenize to get dtexts for splitting

    # Get the split numbers from the total data size
    N_full = len(dtexts_full)
    prop_test, prop_val = 0.2, 0.2
    Ntest_full, Nval_full = int(N_full*prop_test), int(N_full*prop_val)

    # Get the test set data (this needs to be consistent with train_model.py's split)
    dt_test, _, _ = split_test_val_train(dtexts_full, Ntest_full, Nval_full)
    di_test, _, _ = split_test_val_train(np.array([all_images_features_dict[fnm] for fnm in df_txt_filtered_for_eval['filename'].values]), Ntest_full, Nval_full)
    fnm_test, _, _ = split_test_val_train(df_txt_filtered_for_eval['filename'].values, Ntest_full, Nval_full)

    print("\nStarting BLEU score evaluation...")
    pred_good, pred_bad, bleus = [], [], []
    nkeep = 5 # Number of good/bad examples to keep

    for count, (jpgfnm, image_feature, tokenized_text_true) in enumerate(zip(fnm_test, di_test, dt_test)):
        if (count + 1) % 200 == 0:
            print(f"{(count + 1) / len(fnm_test) * 100:.2f}% is done..")

        # True caption (remove startseq/endseq)
        caption_true_list = [index_word.get(i, "") for i in tokenized_text_true if i in index_word] # Safely get words
        if caption_true_list[0] == 'startseq':
            caption_true_list = caption_true_list[1:]
        if caption_true_list[-1] == 'endseq':
            caption_true_list = caption_true_list[:-1]

        # Predicted caption
        caption_pred_full = predict_caption(image_feature, loaded_model, loaded_tokenizer, maxlen, index_word)
        caption_pred_list = caption_pred_full.split()
        if caption_pred_list and caption_pred_list[0] == 'startseq':
            caption_pred_list = caption_pred_list[1:]
        if caption_pred_list and caption_pred_list[-1] == 'endseq':
            caption_pred_list = caption_pred_list[:-1]
        
        # Calculate BLEU score
        # sentence_bleu expects list of reference sentences and a candidate sentence
        # Each reference sentence is a list of tokens.
        bleu = sentence_bleu([caption_true_list], caption_pred_list)
        bleus.append(bleu)

        if bleu > 0.7 and len(pred_good) < nkeep:
            pred_good.append((bleu, jpgfnm, caption_true_list, caption_pred_list))
        elif bleu < 0.3 and len(pred_bad) < nkeep:
            pred_bad.append((bleu, jpgfnm, caption_true_list, caption_pred_list))
    
    mean_bleu = np.mean(bleus)
    print(f"\nMean BLEU: {mean_bleu:.3f}")

    print("\n--- Bad Captions Examples ---")
    plot_predicted_captions(pred_bad, dir_Flickr_jpg, title="Bad Captions")

    print("\n--- Good Captions Examples ---")
    if pred_good:
        plot_predicted_captions(pred_good, dir_Flickr_jpg, title="Good Captions")
    else:
        print("No good captions (BLEU > 0.7) found among the first few checked examples.")
        print("This is common with low overall BLEU scores.")


if __name__ == "__main__":
    import matplotlib.pyplot as plt # Import here for plotting functions

    model_file = os.path.join("models", "image_captioning_model.keras")
    tokenizer_file = os.path.join("models", "tokenizer.json")
    
    default_dataset_path = os.path.expanduser("~/.cache/kagglehub/datasets/shadabhussain/flickr8k/versions/3")

    if not os.path.exists(model_file) or not os.path.exists(tokenizer_file) or not os.path.exists(default_dataset_path):
        print("Required model, tokenizer, or dataset not found.")
        print("Please ensure you have run 'python scripts/train_model.py' and 'python scripts/download_data.py' first.")
        sys.exit(1)

    evaluate(model_file, tokenizer_file, default_dataset_path)