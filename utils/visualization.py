import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import load_img # Assuming Keras is available for image loading
import os

def plthist(dfsub, title="The top 50 most frequently appearing words", topn=50):
    """Plots a histogram of word frequencies."""
    plt.figure(figsize=(20,3))
    plt.bar(dfsub.index, dfsub["count"])
    plt.yticks(fontsize=20)
    plt.xticks(dfsub.index, dfsub["word"], rotation=90,fontsize=20)
    plt.title(title, fontsize=20)
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()

def plot_pca_embeddings(y_pca, picked_pic):
    """Plots 2D PCA embeddings with highlighted points."""
    fig, ax = plt.subplots(figsize=(15,15))
    ax.scatter(y_pca[:,0], y_pca[:, 1], c="white")
    
    # Annotate all points (optional, can be slow for large datasets)
    # for irow in range(y_pca.shape[0]):
    #     ax.annotate(irow, y_pca[irow,:], color="black", alpha=0.5) 
    
    for color, irows in picked_pic.items():
        for irow in irows:
            ax.annotate(irow, y_pca[irow,:], color=color, fontsize=12, weight='bold')

    ax.set_xlabel("PCA Embedding 1", fontsize=20) # Corrected label text
    ax.set_ylabel("PCA Embedding 2", fontsize=20) # Corrected label text
    ax.tick_params(axis='both', which='major', labelsize=15) # Adjust tick font size
    plt.title("PCA Embeddings of Image Features", fontsize=25)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_selected_images(jpgs, picked_pic, image_dir, target_size=(224, 224, 3)):
    """Plots selected images based on PCA clustering."""
    fig = plt.figure(figsize=(16,20))
    count = 1
    
    # Determine the total number of subplots needed
    total_images_to_plot = sum(len(irows) for irows in picked_pic.values())
    
    for color, irows in picked_pic.items():
        for ivec in irows:
            name = jpgs[ivec]
            filename = os.path.join(image_dir, name)
            try:
                image = load_img(filename, target_size=target_size)
                # Create subplots dynamically, assuming 5 columns for images within each color group
                ax = fig.add_subplot(len(picked_pic), 5, count, xticks=[], yticks=[]) 
                ax.imshow(image)
                plt.title(f"{ivec} ({color})", fontsize=10) # Smaller title for subplot
                count += 1
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
                continue
    plt.tight_layout()
    plt.show()

def plot_training_history(hist):
    """Plots the training and validation loss from model history."""
    plt.figure(figsize=(10, 6))
    for label in ["loss", "val_loss"]:
        plt.plot(hist.history[label], label=label)
    plt.legend(fontsize=12)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Model Loss During Training", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def create_caption_string(caption_parts):
    """Helper to join list of words into a string."""
    return " ".join(caption_parts)

def plot_predicted_captions(pred_list, image_dir, target_size=(224, 224, 3), title="Captions"):
    """
    Plots images with their true and predicted captions, and BLEU score.
    pred_list should contain tuples of (bleu_score, filename, true_caption_list, predicted_caption_list)
    """
    npic = len(pred_list) # Number of rows for subplots
    fig = plt.figure(figsize=(10, npic * 4)) # Adjust figure height based on number of captions
    
    count = 1
    for pb in pred_list:
        bleu, jpgfnm, caption_true_list, caption_pred_list = pb

        # Image subplot
        filename = os.path.join(image_dir, jpgfnm)
        try:
            image_load = load_img(filename, target_size=target_size)
            ax = fig.add_subplot(npic, 2, count, xticks=[], yticks=[])
            ax.imshow(image_load)
            count += 1
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
            # If image fails to load, still create a placeholder for caption to maintain layout
            ax = fig.add_subplot(npic, 2, count)
            ax.text(0.5, 0.5, "Image Load Error", horizontalalignment='center', verticalalignment='center', fontsize=15, color='red')
            count += 1
            continue

        # Text subplot
        caption_true_str = create_caption_string(caption_true_list)
        caption_pred_str = create_caption_string(caption_pred_list)
        
        ax = fig.add_subplot(npic, 2, count)
        plt.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0, 0.7, "true: " + caption_true_str, fontsize=12, wrap=True)
        ax.text(0, 0.4, "pred: " + caption_pred_str, fontsize=12, wrap=True)
        ax.text(0, 0.1, "BLEU: {:.3f}".format(bleu), fontsize=12) # Formatted to 3 decimal places
        count += 1
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02) # Add a main title for the plot
    plt.show()