import os
from subprocess import call
import shutil

# Root directories
root_checkpoint_dir = '/nas/datasets/lzy/RS-ChangeDetection/Best_ckpt-KD'
root_output_dir = '/nas/datasets/lzy/RS-ChangeDetection/Figures-KD/TEST/Three-Teachers'
test_dir = '/nas/datasets/lzy/RS-ChangeDetection/CGWX/test'

# Set CUDA device
CUDA_DEVICE = "2"

# Function to reset output directories for initial models
# def reset_initial_directories(model_name):
#     output_dir = os.path.join(root_output_dir, model_name)
#     if os.path.exists(output_dir):
#         shutil.rmtree(output_dir)
#     os.makedirs(output_dir)

# Run initial model to generate images
def run_initial_model(model_name, config_path, checkpoint_path, output_dir):
    command = f"CUDA_VISIBLE_DEVICES={CUDA_DEVICE} python tools/test.py {config_path} {checkpoint_path} --show-dir {output_dir}"
    call(command, shell=True)

# Process each model to generate initial images
def process_initial_model(model_name, model_checkpoint_path):
    initial_checkpoint_dir = os.path.join(model_checkpoint_path, 'initial')

    initial_checkpoint = None

    # Find the .pth file in the initial model directory
    for file_name in os.listdir(initial_checkpoint_dir):
        if file_name.endswith('.pth'):
            initial_checkpoint = os.path.join(initial_checkpoint_dir, file_name)
            break

    if initial_checkpoint is None:
        print(f"Error: Initial checkpoint not found for {model_name}")
        return

    # Define output directory for this model
    output_dir = os.path.join(root_output_dir, model_name, 'initial')

    # Reset output directory
    # reset_initial_directories(model_name)

    # Load the config file for the initial model
    config_path = os.path.join(initial_checkpoint_dir, 'config.py')

    # Run the initial model to generate images
    run_initial_model(model_name=model_name, 
                      config_path=config_path, 
                      checkpoint_path=initial_checkpoint, 
                      output_dir=output_dir)

# Main function to process all models from a given list
def main(models_to_run):
    for model_name in models_to_run:
        model_checkpoint_path = os.path.join(root_checkpoint_dir, model_name)
        if os.path.exists(model_checkpoint_path):
            print(f"Processing initial model for {model_name}...")
            process_initial_model(model_name, model_checkpoint_path)
        else:
            print(f"Model checkpoint path not found for {model_name}.")

if __name__ == '__main__':
    models_to_run = ["BAN-vit-b16-clip-mit-b2", "BAN-vit-b16-in21k-mit-b2", "BAN-vit-l14-clip-mit-b2", 
                     "BIT", "CGNet", "ChangeFormer-mit-b0", "ChangeFormer-mit-b1", 
                     "Changer-mit-b0", "Changer-mit-b1", "Changer-r18", "Changer-s50", 
                     "ChangeStar-farseg", "ChangeStar-upernet", "FC-EF", "FC-Siam-Conc", "FC-Siam-Diff", 
                     "HANet", "IFN", "LightCDNet", "SNUNet", "STANet", "TinyCD", "TTP"]  # List of models to process
    main(models_to_run)
