from keras import layers
from keras import models
from keras.layers import Input, Flatten, Dropout, Activation, LeakyReLU, PReLU # PReLU is not used in original code, but imported.

def build_captioning_model(image_feature_shape, maxlen, vocab_size, dim_embedding=64):
    """
    Builds the image captioning model architecture.
    It combines image features (from VGG16) and text embeddings (LSTM).
    """
    print("Building Image Captioning Model...")

    ## Image Feature Input
    input_image = layers.Input(shape=(image_feature_shape,))
    fimage = layers.Dense(256, activation='relu', name="ImageFeature")(input_image)

    ## Sequence Model (Text) Input
    input_txt = layers.Input(shape=(maxlen,))
    ftxt = layers.Embedding(vocab_size, dim_embedding, mask_zero=True)(input_txt)
    ftxt = layers.LSTM(256, name="CaptionFeature", return_sequences=True)(ftxt)
    se2 = Dropout(0.04)(ftxt)
    ftxt = layers.LSTM(256, name="CaptionFeature2")(se2)

    ## Combined Decoder Model
    decoder = layers.add([ftxt, fimage])
    decoder = layers.Dense(256, activation='relu')(decoder)
    output = layers.Dense(vocab_size, activation='softmax')(decoder)

    model = models.Model(inputs=[input_image, input_txt], outputs=output)
    
    print("Image Captioning Model Summary:")
    model.summary()
    return model