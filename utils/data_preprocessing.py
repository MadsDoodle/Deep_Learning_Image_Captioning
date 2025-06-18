import numpy as np
import os
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from collections import OrderedDict
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer


def load_and_extract_image_features(image_dir, modelvgg, target_size=(224, 224, 3)):
    """
    Loads images, preprocesses them, and extracts features using the VGG16 model.
    """
    images = OrderedDict()
    jpgs = os.listdir(image_dir)
    print(f"Loading and extracting features from {len(jpgs)} images...")
    
    # Sort jpgs for consistent processing order if needed, but not critical for feature extraction
    # jpgs.sort() 

    for i, name in enumerate(jpgs):
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i}/{len(jpgs)} images...")
        filename = os.path.join(image_dir, name)
        try:
            image = load_img(filename, target_size=target_size)
            nimage = img_to_array(image)
            nimage = preprocess_input(nimage)
            y_pred = modelvgg.predict(nimage.reshape((1,) + nimage.shape[:3]), verbose=0)
            images[name] = y_pred.flatten()
        except Exception as e:
            print(f"Error processing image {name}: {e}")
            continue
    print(f"Finished processing {len(images)} images.")
    return images


def create_tokenizer_and_sequences(captions_df, nb_words=6000):
    """
    Creates a Keras Tokenizer and converts captions to sequences.
    """
    tokenizer = Tokenizer(num_words=nb_words)
    tokenizer.fit_on_texts(captions_df["caption"])
    vocab_size = len(tokenizer.word_index) + 1
    print("Vocabulary size: {}".format(vocab_size))
    dtexts = tokenizer.texts_to_sequences(captions_df["caption"])
    maxlen = np.max([len(text) for text in dtexts])
    print(f"Max caption length: {maxlen}")
    return tokenizer, vocab_size, dtexts, maxlen


def split_test_val_train(data_list, Ntest, Nval):
    """Splits a list of data into test, validation, and training sets."""
    return(data_list[:Ntest], data_list[Ntest: Ntest+Nval], data_list[Ntest+Nval:])


def preprocess_sequence_data(dtexts, dimages, maxlen, vocab_size):
    """
    Preprocesses text and image data for the captioning model.
    Generates input-output pairs for sequence prediction.
    """
    N = len(dtexts)
    print("# captions/images = {}".format(N))
    assert (N==len(dimages)), "Lengths of texts and images must be similar."

    Xtext, Ximage, ytext = [], [], []

    for text, image in zip(dtexts, dimages):
        for i in range(1, len(text)):
            in_text = text[:i]
            out_text = text[i]

            in_text = pad_sequences([in_text], maxlen=maxlen, padding='post').flatten()
            out_text = to_categorical(out_text, num_classes=vocab_size)

            Xtext.append(in_text)
            Ximage.append(image)
            ytext.append(out_text)

    Xtext = np.array(Xtext)
    Ximage = np.array(Ximage)
    ytext = np.array(ytext)

    print("Shapes: Xtext={}, Ximage={}, ytext={}".format(Xtext.shape, Ximage.shape, ytext.shape))
    return (Xtext, Ximage, ytext)