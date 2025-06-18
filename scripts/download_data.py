import os
import kagglehub
import warnings

def setup_kaggle_credentials():
    """
    Instructions to manually upload kaggle.json in a Colab-like environment.
    For VS Code, you'd typically place kaggle.json directly in ~/.kaggle/
    or ensure it's provided securely in your environment.
    """
    print("Please ensure your kaggle.json is in '~/.kaggle/' for local execution.")
    print("If running in a different environment, you might need to adjust paths or upload.")

    # For local VS Code setup, assume kaggle.json is already correctly placed
    # or handle its secure placement. The Colab-specific file upload is omitted.

    kaggle_dir = os.path.expanduser('~/.kaggle')
    if not os.path.exists(kaggle_dir):
        os.makedirs(kaggle_dir)
    
    # This part is highly dependent on how kaggle.json is made available locally.
    # In a typical VS Code setup, you would have already manually placed it or
    # configured your environment.
    # We will assume ~/.kaggle/kaggle.json exists for simplicity here.
    kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')
    if not os.path.exists(kaggle_json_path):
        warnings.warn("kaggle.json not found in ~/.kaggle/. Please place it there for Kaggle downloads to work.")
    else:
        os.chmod(kaggle_json_path, 0o600) # Ensure correct permissions


def download_and_list_data(dataset_name="shadabhussain/flickr8k"):
    """
    Downloads the specified Kaggle dataset and lists its contents.
    """
    print("Downloading dataset...")
    path = kagglehub.dataset_download(dataset_name)
    print("Path to dataset files:", path)

    dataset_path = path # The downloaded path is the base path

    print("\nListing files and folders in the dataset directory:")
    for root, dirs, files in os.walk(dataset_path):
        print(f"Directory: {root}")
        print(f"Subdirectories: {dirs}")
        print(f"Files: {files}")
        print("-" * 50)
    
    csv_files = [file for file in os.listdir(dataset_path) if file.endswith('.csv')]
    print("CSV Files:", csv_files)
    
    return dataset_path

if __name__ == "__main__":
    setup_kaggle_credentials()
    dataset_base_path = download_and_list_data()
    print(f"\nDataset available at: {dataset_base_path}")