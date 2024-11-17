import os
import argparse

def print_folder_structure(root_folder, level=0, prefix=""):
    # Print the folder name at the current level
    folder_name = os.path.basename(root_folder)
    if level == 0:
        print(f"{folder_name}/")  # Print root folder without prefix
    else:
        print(f"{prefix}├── {folder_name}/")

    # Get all subdirectories and files in the current directory
    entries = sorted(os.listdir(root_folder))
    files = [f for f in entries if os.path.isfile(os.path.join(root_folder, f))]
    dirs = [d for d in entries if os.path.isdir(os.path.join(root_folder, d))]

    # Adjust indentation for files and subdirectories
    file_prefix = "    " * level + "├── "
    folder_prefix = "    " * level + "│   "

    # Print up to the first three files and add "..." if more files exist
    for i, file in enumerate(files[:3]):
        print(f"{file_prefix}{file}")
    if len(files) > 3:
        print(f"{file_prefix}...")

    # Print each subdirectory, updating the level and prefix
    for i, directory in enumerate(dirs):
        last_dir = i == len(dirs) - 1
        sub_prefix = "    " * level + ("└── " if last_dir else "├── ")
        next_level = level + 1
        print_folder_structure(os.path.join(root_folder, directory), next_level, sub_prefix)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Print folder structure with limited file listing")
    parser.add_argument("folder", type=str, help="Path to the folder to display")
    args = parser.parse_args()

    # Call the function with the provided folder path
    print_folder_structure(args.folder)
