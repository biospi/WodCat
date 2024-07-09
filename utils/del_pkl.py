import os
import glob

# Define the directory path
directory = r"E:\Cats\paper_debug_regularisation_36"

# Construct the search pattern
pattern = os.path.join(directory, "*.pkl")

# Get the list of all .pkl files in the directory
pkl_files = glob.glob(pattern)

# Delete each .pkl file
for file_path in pkl_files:
    try:
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")
