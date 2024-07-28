import glob
import numpy as np
import os
import random
import shutil
import yaml
from PIL import Image

from .fomat_converters import normalized_coordinates

def extract_file_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

def get_file_name():
    # Get list of annotation and image files
    annots_path = glob.glob("data/raw/*.txt")
    annots_path = [os.path.normpath(file).replace("\\", "/") for file in annots_path]
    imgs_path = glob.glob("data/raw/*.jpg")
    imgs_path = [os.path.normpath(file).replace("\\", "/") for file in imgs_path]

    # Extract base file names (without extensions)
    annots_file_names = [extract_file_name(label_path) for label_path in annots_path]
    imgs_file_names = [extract_file_name(img_path) for img_path in imgs_path]
    return annots_file_names, imgs_file_names

def get_valid_file_name(annots_file_names, imgs_file_names):
    valid_file_names = list(set(annots_file_names).intersection(imgs_file_names))
    return valid_file_names

def get_file_path(valid_file_names):
    annots_path = ["data/raw/" + file + ".txt" for file in valid_file_names]
    imgs_path = ["data/raw/" + file + ".jpg" for file in valid_file_names]
    return annots_path, imgs_path

def apply_label_formatter(annot_files, img_files):
    output_dir = "data/formatted"
    os.makedirs(output_dir, exist_ok=True)

    for annot_file, img_file in zip(annot_files, img_files):
        # Get image dimensions
        with Image.open(img_file) as img:
            width, height = img.size
        
        with open(annot_file, 'r') as file:
            lines = file.readlines()
        
        new_lines = []
        for line in lines:
            values = line.split()
            # Remove the first value (file name)
            values = values[1:-1]
            # Convert to floats
            values = list(map(float, values))
            
            # Original format: [class, xmin, ymin, xmax, ymax]
            class_id = 1
            xmin = values[0]
            ymin = values[1]
            box_width = values[2]
            box_height = values[3]
            
            # Calculate YOLO format values
            x_center, y_center, box_width, box_height = normalized_coordinates(width, height, xmin, ymin, box_width, box_height)
#            x_center, y_center, box_width, box_height = xmin, ymin, xmax, ymax
            new_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
        
        # Write the modified annotation to a new file in the output directory
        new_annot_file = os.path.join(output_dir, os.path.basename(annot_file))
        with open(new_annot_file, 'w') as file:
            file.write('\n'.join(new_lines))

    return glob.glob(output_dir + "/*.txt")

def apply_img_formatter(img_files):
    ## Resize the images to 640x640, return the new file paths
    output_dir = "data/formatted"
    os.makedirs(output_dir, exist_ok=True)

    for img_file in img_files:
        with Image.open(img_file) as img:
            img = img.resize((640, 640))
            img.save(os.path.join(output_dir, os.path.basename(img_file)))

    return glob.glob(output_dir + "/*.jpg")

def make_splits(annots_path, imgs_path, train_ratio = 0.7, val_ratio = 0.15, log=True):
    # Shuffle the indices
    indices = np.arange(len(imgs_path))
    random.shuffle(indices)

    # Calculate the number of samples for each set
    num_samples = len(imgs_path)
    num_train = int(train_ratio * num_samples)
    num_val = int(val_ratio * num_samples)

    # Split the indices
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]

    # Separate the files into different sets
    train_annots_path = [annots_path[i] for i in train_indices]
    train_imgs_path = [imgs_path[i] for i in train_indices]

    val_annots_path = [annots_path[i] for i in val_indices]
    val_imgs_path = [imgs_path[i] for i in val_indices]

    test_annots_path = [annots_path[i] for i in test_indices]
    test_imgs_path = [imgs_path[i] for i in test_indices]

    if log:
        # Verify the splits
        print(f"Number of training samples: {len(train_annots_path)}")
        print(f"Number of validation samples: {len(val_annots_path)}")
        print(f"Number of testing samples: {len(test_annots_path)}")
    
    return train_annots_path, train_imgs_path, val_annots_path, val_imgs_path, test_annots_path, test_imgs_path

def copy_files(file_list, dest_dir):
    for file in file_list:
        shutil.copy(file, dest_dir)

def copy_to_directories(train_annots_path, train_imgs_path, val_annots_path, val_imgs_path, test_annots_path, test_imgs_path, log=True):
    # Define the directory structure
    base_dir = "data/preprocessed"
    dirs = ['train', 'val', 'test']
    sub_dirs = ['images', 'labels']

    # Create the directories
    for dir_ in dirs:
        for sub_dir in sub_dirs:
            os.makedirs(os.path.join(base_dir, dir_, sub_dir), exist_ok=True)

    # Copy the training files
    copy_files(train_imgs_path, os.path.join(base_dir, 'train/images'))
    copy_files(train_annots_path, os.path.join(base_dir, 'train/labels'))

    # Copy the validation files
    copy_files(val_imgs_path, os.path.join(base_dir, 'val/images'))
    copy_files(val_annots_path, os.path.join(base_dir, 'val/labels'))

    # Copy the testing files
    copy_files(test_imgs_path, os.path.join(base_dir, 'test/images'))
    copy_files(test_annots_path, os.path.join(base_dir, 'test/labels'))

    if log:
        # Verify the splits
        print(f"Number of training images: {len(os.listdir(os.path.join(base_dir, 'train/images')))}")
        print(f"Number of training labels: {len(os.listdir(os.path.join(base_dir, 'train/labels')))}")
        print(f"Number of validation images: {len(os.listdir(os.path.join(base_dir, 'val/images')))}")
        print(f"Number of validation labels: {len(os.listdir(os.path.join(base_dir, 'val/labels')))}")
        print(f"Number of testing images: {len(os.listdir(os.path.join(base_dir, 'test/images')))}")
        print(f"Number of testing labels: {len(os.listdir(os.path.join(base_dir, 'test/labels')))}")


# Function to create a data.yaml file
def create_data_yaml(log=True):
    # Define the base directory and class names
    dir_path = "data/preprocessed"
    class_names = ['background', 'license plate']

    data = {
        'path' : os.getcwd(),
        'train': os.path.join(dir_path, 'train'),
        'val': os.path.join(dir_path, 'val'),
        'test': os.path.join(dir_path, 'test'),
        'nc': len(class_names),
        'names': class_names
    }
        
    with open(os.path.join(dir_path, 'data.yaml'), 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    if log:
        # Verify the data.yaml file content
        with open(os.path.join(dir_path, 'data.yaml'), 'r') as file:
            data_yaml_content = yaml.safe_load(file)
            print(data_yaml_content)


def pipeline():
    # Get the file names
    annots_file_names, imgs_file_names = get_file_name()

    # Get the valid file names
    valid_file_names = get_valid_file_name(annots_file_names, imgs_file_names)

    # Get the file paths
    annots_path, imgs_path = get_file_path(valid_file_names)

    # Apply the label formatter
    annot_files = apply_label_formatter(annots_path, imgs_path)
    
    ## Resize the images to 640x640
    imgs_path = apply_img_formatter(imgs_path)

    # Make the splits
    train_annots_path, train_imgs_path, val_annots_path, val_imgs_path, test_annots_path, test_imgs_path = make_splits(annot_files, imgs_path)

    # Copy the files to the directories
    copy_to_directories(train_annots_path, train_imgs_path, val_annots_path, val_imgs_path, test_annots_path, test_imgs_path)

    # Create the data.yaml file
    create_data_yaml()

    return "Data preparation pipeline completed successfully!"