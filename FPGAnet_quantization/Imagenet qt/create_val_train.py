import os
import shutil
import random

def create_train_val_dirs(input_dir, output_dir, val_split=0.2):
    """
    Create train and val directories for an image dataset.

    Args:
    - input_dir: Path to the directory containing the original dataset.
    - output_dir: Path to the directory where train and val directories will be created.
    - val_split: Fraction of data to be used for validation. Should be between 0 and 1.

    Returns:
    - None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # Get list of all class folders
    class_folders = [folder for folder in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, folder))]

    # Iterate over each class folder
    for folder in class_folders:
        class_files = os.listdir(os.path.join(input_dir, folder))
        num_val = int(val_split * len(class_files))

        # Split files into train and val sets
        train_files = class_files[num_val:]
        val_files = class_files[:num_val]

        # Create class directories in train and val directories
        train_class_dir = os.path.join(train_dir, folder)
        val_class_dir = os.path.join(val_dir, folder)

        if not os.path.exists(train_class_dir):
            os.makedirs(train_class_dir)
        if not os.path.exists(val_class_dir):
            os.makedirs(val_class_dir)

        # Copy train files to train directory
        for file in train_files:
            src = os.path.join(input_dir, folder, file)
            dst = os.path.join(train_class_dir, file)
            shutil.copyfile(src, dst)

        # Copy val files to val directory
        for file in val_files:
            src = os.path.join(input_dir, folder, file)
            dst = os.path.join(val_class_dir, file)
            shutil.copyfile(src, dst)
# Example usage:
input_directory = "./data/imagenet_subset/imagenet_subval"
output_directory = "./data/imagenet_subset"
create_train_val_dirs(input_directory, output_directory)
