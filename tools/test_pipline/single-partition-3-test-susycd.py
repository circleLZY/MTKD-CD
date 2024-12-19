import os
import shutil
import numpy as np
from PIL import Image
from subprocess import call
import importlib.util

# Root directories
root_image_dir = '/nas/datasets/lzy/RS-ChangeDetection/Figures-SYSUCD/TEST/Three-Teachers'
root_checkpoint_dir = '/nas/datasets/lzy/RS-ChangeDetection/Best_ckpt-SYSU-CD'
root_output_dir = '/nas/datasets/lzy/RS-ChangeDetection/Figures-SYSUCD/TEST/Three-Teachers'
gt_folder = '/nas/datasets/lzy/RS-ChangeDetection/Benchmarks/SYSU-CD/test/label'
test_dir = '/nas/datasets/lzy/RS-ChangeDetection/Benchmarks/SYSU-CD/test'
test_s_dir = '/nas/datasets/lzy/RS-ChangeDetection/Benchmarks/SYSU-CD/test_s'
test_m_dir = '/nas/datasets/lzy/RS-ChangeDetection/Benchmarks/SYSU-CD/test_m'
test_l_dir = '/nas/datasets/lzy/RS-ChangeDetection/Benchmarks/SYSU-CD/test_l'

# Set CUDA device
CUDA_DEVICE = "2"

# Clean and recreate test directories
def reset_directories():
    for dir_path in [test_s_dir, test_m_dir, test_l_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

# Calculate the percentage of changed pixels
def calculate_change_percentage(image):
    image_array = np.array(image)
    changed_pixels = np.sum(image_array == 255)
    total_pixels = image_array.size
    return changed_pixels / total_pixels

# Split test dataset based on change percentage into small, medium, large
def split_test_set(label_image_dir):
    image_list = sorted(os.listdir(label_image_dir))
    for image_name in image_list:
        image_path = os.path.join(label_image_dir, image_name)
        image = Image.open(image_path).convert('L')
        change_percentage = calculate_change_percentage(image)

        # Determine target folder (small, medium, or large)
        if change_percentage <= 0.05:
            target_dir = test_s_dir
        elif change_percentage <= 0.2:
            target_dir = test_m_dir
        else:
            target_dir = test_l_dir
        
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

# Run small, medium, and large models for each subset
def run_models_on_subsets(config_small, config_medium, config_large, checkpoint_small, checkpoint_medium, checkpoint_large, small_output_dir, medium_output_dir, large_output_dir):
    # Small model on test_s
    command = f"CUDA_VISIBLE_DEVICES={CUDA_DEVICE} python tools/test.py {config_small} {checkpoint_small} --show-dir {small_output_dir}"
    call(command, shell=True)

    # Medium model on test_m
    command = f"CUDA_VISIBLE_DEVICES={CUDA_DEVICE} python tools/test.py {config_medium} {checkpoint_medium} --show-dir {medium_output_dir}"
    call(command, shell=True)

    # Large model on test_l
    command = f"CUDA_VISIBLE_DEVICES={CUDA_DEVICE} python tools/test.py {config_large} {checkpoint_large} --show-dir {large_output_dir}"
    call(command, shell=True)

# Combine small, medium, and large result images
def combine_images(small_image_dir, medium_image_dir, large_image_dir, final_image_dir):
    os.makedirs(final_image_dir, exist_ok=True)

    for image_dir in [small_image_dir, medium_image_dir, large_image_dir]:
        images = sorted(os.listdir(image_dir))
        for image_name in images:
            image_path = os.path.join(image_dir, image_name)
            final_image_path = os.path.join(final_image_dir, image_name)
            if os.path.exists(image_path):
                image = Image.open(image_path)
                image.save(final_image_path)

# Calculate the final mIOU
def calculate_final_mIOU(gt_folder, result_folder):
    command = f"python /nas/datasets/lzy/RS-ChangeDetection/tools/cal_mIOU.py --gt_folder {gt_folder} --result_folder {result_folder}"
    call(command, shell=True)

# Process each model
def process_model(model_name, model_checkpoint_path):
    label_image_dir = os.path.join(root_image_dir, model_name, 'initial/vis_data/vis_image')
    
    # Identify small, medium, and large model checkpoints based on .pth files
    small_checkpoint_dir = os.path.join(model_checkpoint_path, 'small')
    medium_checkpoint_dir = os.path.join(model_checkpoint_path, 'medium')
    large_checkpoint_dir = os.path.join(model_checkpoint_path, 'large')

    small_checkpoint = None
    medium_checkpoint = None
    large_checkpoint = None

    # Find the .pth file in the small model directory
    for file_name in os.listdir(small_checkpoint_dir):
        if file_name.endswith('.pth'):
            small_checkpoint = os.path.join(small_checkpoint_dir, file_name)
            break

    # Find the .pth file in the medium model directory
    for file_name in os.listdir(medium_checkpoint_dir):
        if file_name.endswith('.pth'):
            medium_checkpoint = os.path.join(medium_checkpoint_dir, file_name)
            break

    # Find the .pth file in the large model directory
    for file_name in os.listdir(large_checkpoint_dir):
        if file_name.endswith('.pth'):
            large_checkpoint = os.path.join(large_checkpoint_dir, file_name)
            break

    if small_checkpoint is None or medium_checkpoint is None or large_checkpoint is None:
        print(f"Error: Checkpoints not found for {model_name}")
        return

    # Define output directories for this model
    small_output_dir = os.path.join(root_output_dir, model_name, 'small')
    medium_output_dir = os.path.join(root_output_dir, model_name, 'medium')
    large_output_dir = os.path.join(root_output_dir, model_name, 'large')
    final_output_dir = os.path.join(root_output_dir, model_name, 'final')
    small_image_dir = os.path.join(small_output_dir, 'vis_data/vis_image')
    medium_image_dir = os.path.join(medium_output_dir, 'vis_data/vis_image')
    large_image_dir = os.path.join(large_output_dir, 'vis_data/vis_image')
    final_image_dir = os.path.join(final_output_dir, 'vis_data/vis_image')

    # Reset directories
    reset_directories()

    # Split the test set based on change percentage
    split_test_set(label_image_dir)

    # Load configs for small, medium, and large models
    config_small_path = os.path.join(small_checkpoint_dir, 'config.py')
    config_medium_path = os.path.join(medium_checkpoint_dir, 'config.py')
    config_large_path = os.path.join(large_checkpoint_dir, 'config.py')
    config_small = load_config(config_small_path)
    config_medium = load_config(config_medium_path)
    config_large = load_config(config_large_path)

    # Run small, medium, and large models on their respective subsets
    run_models_on_subsets(config_small=config_small_path, 
                          config_medium=config_medium_path,
                          config_large=config_large_path, 
                          checkpoint_small=small_checkpoint, 
                          checkpoint_medium=medium_checkpoint, 
                          checkpoint_large=large_checkpoint, 
                          small_output_dir=small_output_dir, 
                          medium_output_dir=medium_output_dir,
                          large_output_dir=large_output_dir)

    # Combine images from small, medium, and large model results
    combine_images(small_image_dir, medium_image_dir, large_image_dir, final_image_dir)

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
    models_to_run = ["CGNet"] 
    main(models_to_run)
