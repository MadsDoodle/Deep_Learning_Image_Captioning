from keras.applications import VGG16
from keras import models

def load_vgg16_feature_extractor():
    """
    Loads a pre-trained VGG16 model from Keras with ImageNet weights
    and removes its final classification layer to use as a feature extractor.
    """
    print("Loading VGG16 model with ImageNet weights...")
    modelvgg = VGG16(weights='imagenet')
    print("Original VGG16 Model Summary:")
    modelvgg.summary()

    # Remove the last layer (predictions)
    modelvgg.layers.pop()
    modelvgg = models.Model(inputs=modelvgg.inputs, outputs=modelvgg.layers[-1].output)
    print("\nModified VGG16 Feature Extractor Model Summary:")
    modelvgg.summary()
    return modelvgg