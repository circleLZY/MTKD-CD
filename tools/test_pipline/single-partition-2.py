import os
import shutil
import numpy as np
from PIL import Image
from subprocess import call
import importlib.util

# Root directories
root_image_dir = '/nas/datasets/lzy/RS-ChangeDetection/Figures/TEST/Single'
root_checkpoint_dir = '/nas/datasets/lzy/RS-ChangeDetection/Best_ckpt'
root_output_dir = '/nas/datasets/lzy/RS-ChangeDetection/Figures/TEST/Single-Partition'
gt_folder = '/nas/datasets/lzy/RS-ChangeDetection/CGWX/test/label'
test_dir = '/nas/datasets/lzy/RS-ChangeDetection/CGWX/test'
test_l_dir = '/nas/datasets/lzy/RS-ChangeDetection/CGWX/test_l'
test_s_dir = '/nas/datasets/lzy/RS-ChangeDetection/CGWX/test_s'

# Set CUDA device
CUDA_DEVICE = "2"

# Clean and recreate test directories
def reset_directories():
    for dir_path in [test_l_dir, test_s_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

# Calculate the percentage of changed pixels
def calculate_change_percentage(image):
    image_array = np.array(image)
    changed_pixels = np.sum(image_array == 255)
    total_pixels = image_array.size
    return changed_pixels / total_pixels

# Split test dataset based on change percentage
def split_test_set(label_image_dir):
    image_list = sorted(os.listdir(label_image_dir))
    for image_name in image_list:
        image_path = os.path.join(label_image_dir, image_name)
        image = Image.open(image_path).convert('L')
        change_percentage = calculate_change_percentage(image)
        
        # Determine target folder (small or large)
        target_dir = test_s_dir if change_percentage <= 0.1 else test_l_dir
        
        # Ensure each subfolder exists
        for subfolder in ['A', 'B', 'label']:
            src_path = os.path.join(test_dir, subfolder, image_name)
            dst_subfolder = os.path.join(target_dir, subfolder)
            os.makedirs(dst_subfolder, exist_ok=True)
            dst_path = os.path.join(dst_subfolder, image_name)
            shutil.copy(src_path, dst_path)

# Function to dynamically load a config file
def load_config(config_path):
    if os.path.exists(config_path):
        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        print(f"Loaded config from {config_path}")
        return config
    else:
        print(f"No config file found at {config_path}")
        return None

# Run small and large models for each subset
def run_models_on_subsets(config_small, config_large, checkpoint_small, checkpoint_large, small_output_dir, large_output_dir):
    # Small model on test_s
    command = f"CUDA_VISIBLE_DEVICES={CUDA_DEVICE} python tools/test.py {config_small} {checkpoint_small} --show-dir {small_output_dir}"
    call(command, shell=True)
    
    # Large model on test_l
    command = f"CUDA_VISIBLE_DEVICES={CUDA_DEVICE} python tools/test.py {config_large} {checkpoint_large} --show-dir {large_output_dir}"
    call(command, shell=True)

# Combine small and large result images
def combine_images(small_image_dir, large_image_dir, final_image_dir):
    os.makedirs(final_image_dir, exist_ok=True)
    
    small_images = sorted(os.listdir(small_image_dir))
    large_images = sorted(os.listdir(large_image_dir))
    
    for image_name in small_images:
        small_image_path = os.path.join(small_image_dir, image_name)
        final_image_path = os.path.join(final_image_dir, image_name)
        if os.path.exists(small_image_path):
            image = Image.open(small_image_path)
            image.save(final_image_path)
    
    for image_name in large_images:
        large_image_path = os.path.join(large_image_dir, image_name)
        final_image_path = os.path.join(final_image_dir, image_name)
        if os.path.exists(large_image_path):
            image = Image.open(large_image_path)
            image.save(final_image_path)

# Calculate the final mIOU
def calculate_final_mIOU(gt_folder, result_folder):
    command = f"python /nas/datasets/lzy/RS-ChangeDetection/tools/cal_mIOU.py --gt_folder {gt_folder} --result_folder {result_folder}"
    call(command, shell=True)


# Process each model
def process_model(model_name, model_checkpoint_path):
    label_image_dir = os.path.join(root_image_dir, model_name, 'vis_data/vis_image')
    
    # Identify small and large model checkpoints based on .pth files
    small_checkpoint_dir = os.path.join(model_checkpoint_path, 'small')
    large_checkpoint_dir = os.path.join(model_checkpoint_path, 'large')

    small_checkpoint = None
    large_checkpoint = None

    # Find the .pth file in the small model directory
    for file_name in os.listdir(small_checkpoint_dir):
        if file_name.endswith('.pth'):
            small_checkpoint = os.path.join(small_checkpoint_dir, file_name)
            break

    # Find the .pth file in the large model directory
    for file_name in os.listdir(large_checkpoint_dir):
        if file_name.endswith('.pth'):
            large_checkpoint = os.path.join(large_checkpoint_dir, file_name)
            break

    if small_checkpoint is None or large_checkpoint is None:
        print(f"Error: Checkpoints not found for {model_name}")
        return

    # Define output directories for this model
    small_output_dir = os.path.join(root_output_dir, model_name, 'small')
    large_output_dir = os.path.join(root_output_dir, model_name, 'large')
    final_output_dir = os.path.join(root_output_dir, model_name, 'final')
    small_image_dir = os.path.join(small_output_dir, 'vis_data/vis_image')
    large_image_dir = os.path.join(large_output_dir, 'vis_data/vis_image')
    final_image_dir = os.path.join(final_output_dir, 'vis_data/vis_image')

    # Reset directories
    reset_directories()

    # Split the test set based on change percentage
    split_test_set(label_image_dir)

    # Load configs for small and large models
    config_small_path = os.path.join(small_checkpoint_dir, 'config.py')
    config_large_path = os.path.join(large_checkpoint_dir, 'config.py')
    config_small = load_config(config_small_path)
    config_large = load_config(config_large_path)

    # Run small and large models on their respective subsets
    run_models_on_subsets(config_small=config_small_path, 
                          config_large=config_large_path, 
                          checkpoint_small=small_checkpoint, 
                          checkpoint_large=large_checkpoint, 
                          small_output_dir=small_output_dir, 
                          large_output_dir=large_output_dir)

    # Combine images from small and large model results
    combine_images(small_image_dir, large_image_dir, final_image_dir)

    # Calculate mIOU
    calculate_final_mIOU(gt_folder, final_image_dir)


# Main function to process all models from a given list
def main(models_to_run):
    for model_name in models_to_run:
        model_checkpoint_path = os.path.join(root_checkpoint_dir, model_name)
        if os.path.exists(model_checkpoint_path):
            print(f"Processing {model_name}...")
            process_model(model_name, model_checkpoint_path)
        else:
            print(f"Model checkpoint path not found for {model_name}.")

if __name__ == '__main__':
    models_to_run = ["Changer-mit-b0", "Changer-mit-b1", "CGNet"]  # List of models to process
    # models_to_run = [ "TTP", "Changer-mit-b0", "Changer-mit-b1", "TinyCD", "CGNet", "BAN-vit-b16-int21k-mit-b2"]  # List of models to process
    main(models_to_run)
