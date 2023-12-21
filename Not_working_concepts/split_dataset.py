import os
import random
import shutil
from collections import defaultdict

def split_dataset(images_folder, output_folder, train_ratio=0.49, validation_ratio=0.21):
    # Create output folders if they don't exist
    for folder in ['train', 'validation', 'test']:
        folder_path = os.path.join(output_folder, folder)
        os.makedirs(folder_path, exist_ok=True)

    # Get list of image filenames and captions from the text file
    with open(os.path.join(images_folder, 'captions.txt'), 'r') as file:
        lines = file.readlines()

    # Shuffle the data
    random.shuffle(lines)

    # Create a mapping from image filenames to lists of captions
    image_caption_mapping = defaultdict(list)
    for line in lines:
        parts = line.strip().split(',', 1)
        image_filename = parts[0]
        caption = parts[1] if len(parts) > 1 else ""
        image_caption_mapping[image_filename].append(caption)

    total_images = len(image_caption_mapping)
    train_size = int(total_images * train_ratio)
    validation_size = int(total_images * validation_ratio)

    # Split the data
    train_data = list(image_caption_mapping.items())[:train_size]
    validation_data = list(image_caption_mapping.items())[train_size:train_size + validation_size]
    test_data = list(image_caption_mapping.items())[train_size + validation_size:]

    # Write data to txt files for each split
    splits = [('train', train_data), ('validation', validation_data), ('test', test_data)]

    for split_name, split_data in splits:
        with open(os.path.join(output_folder, split_name, f'{split_name}_images.txt'), 'w') as file:
            for image_filename, captions in split_data:
                for caption in captions:
                    file.write(f'{image_filename},{caption}\n')

        # Copy images to respective folders
        split_folder = os.path.join(output_folder, split_name)
        for image_filename, _ in split_data:
            source_path = os.path.join(images_folder, 'Images', image_filename)
            destination_path = os.path.join(split_folder, image_filename)

            # Use shutil.copy instead of os.rename to keep the original images
            shutil.copy(source_path, destination_path)



def check_indexes(file1_path, file2_path, file3_path):
    # Read data from input files
    data1 = set()
    with open(file1_path, 'r') as file1:
        for line in file1:
            index, caption = line.strip().split(',', 1)
            data1.add(index)

    data2 = set()
    with open(file2_path, 'r') as file2:
        for line in file2:
            index, caption = line.strip().split(',', 1)
            data2.add(index)

    data3 = set()
    with open(file3_path, 'r') as file3:
        for line in file3:
            index, caption = line.strip().split(',', 1)
            data3.add(index)

    # Check for common indexes
    common_indexes = data1.intersection(data2).union(data2.intersection(data3)).union(data3.intersection(data1))

    # Return True if any common indexes are found, else return False
    return bool(common_indexes)

if __name__ == "__main__":
    file1_path = "/fhome/gia03/Images_split/test/test_images.txt"
    file2_path = "/fhome/gia03/Images_split/train/train_images.txt"
    file3_path = "/fhome/gia03/Images_split/validation/validation_images.txt"

    has_common_indexes = check_indexes(file1_path, file2_path, file3_path)

    if has_common_indexes:
        print("Error: Some indexes are repeated between files.")
    else:
        print("All indexes are exclusive to their respective files.")
'''
images_folder = '/fhome/gia03/'
output_folder = '/fhome/gia03/Images_split'


split_dataset(images_folder, output_folder)
'''
