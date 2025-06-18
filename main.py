import subprocess
import sys
import os

# Define paths to your scripts
DOWNLOAD_SCRIPT = "scripts/download_data.py"
TRAIN_SCRIPT = "scripts/train_model.py"
PREDICT_SCRIPT = "scripts/predict_caption.py"
EVALUATE_SCRIPT = "scripts/evaluate_model.py"

def run_script(script_path):
    """Helper function to run a Python script."""
    print(f"\n--- Running {script_path} ---")
    try:
        # Use sys.executable to ensure the correct python interpreter is used
        result = subprocess.run([sys.executable, script_path], check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Errors/Warnings from {script_path}:\n{result.stderr}")
        print(f"--- Finished {script_path} ---")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}:")
        print(e.stdout)
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"Error: Script not found at {script_path}. Please check the path.")
        return False

def main():
    print("Starting Image Captioning Workflow...")

    # Step 1: Download Data
    if not run_script(DOWNLOAD_SCRIPT):
        print("Data download failed. Exiting.")
        return

    # Step 2: Train Model
    # The train_model.py will handle loading and preprocessing
    if not run_script(TRAIN_SCRIPT):
        print("Model training failed. Exiting.")
        return
    
    # Step 3: Evaluate Model
    # Requires model and tokenizer saved from training
    if not run_script(EVALUATE_SCRIPT):
        print("Model evaluation failed. Exiting.")
        return

    # Step 4: Predict Captions (Display examples)
    if not run_script(PREDICT_SCRIPT):
        print("Caption prediction display failed. Exiting.")
        return

    print("\nImage Captioning Workflow Completed Successfully!")

if __name__ == "__main__":
    # Ensure the models directory exists before training/saving
    os.makedirs('models', exist_ok=True)
    main()