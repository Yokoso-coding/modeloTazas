from roboflow import Roboflow
import os

def download_dataset():
    try:
        # Initialize Roboflow
        rf = Roboflow(api_key="0I6D2S0neOm1xZQjylg4")

        # Download the mug dataset
        project = rf.workspace("seluk-niversitesi").project("coffee-mug-detection-lzd4w")
        version = project.version(1)
        dataset = version.download("yolov8")

        # Print and return the path where the dataset was downloaded
        print(f"Dataset downloaded to: {dataset.location}")
        return dataset.location
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

if __name__ == "__main__":
    # Get the current working directory
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")

    dataset_path = download_dataset()
    if dataset_path:
        print(f"Dataset path: {dataset_path}")
        
        # Full path for the text file
        file_path = os.path.join(current_dir, "dataset_path.txt")
        
        try:
            # Save the path to a file
            with open(file_path, "w") as f:
                f.write(dataset_path)
            print(f"Path saved to: {file_path}")
        except Exception as e:
            print(f"Error writing to file: {e}")
        
        # Verify that the file was created and read its contents
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    saved_path = f.read()
                print(f"Saved path: {saved_path}")
            except Exception as e:
                print(f"Error reading file: {e}")
        else:
            print(f"Error: {file_path} was not created")
    else:
        print("Dataset download failed. Cannot proceed.")

    # List directory contents
    print("\nDirectory contents:")
    for item in os.listdir(current_dir):
        print(item)