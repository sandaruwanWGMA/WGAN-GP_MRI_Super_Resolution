import os
import numpy as np
import tensorflow as tf
import nibabel as nib
from models.wgan_3d_low import generator
from utils.super_res_data_generator import SuperResDataGenerator

def save_nifti(img_data, output_path, reference_file=None):
    """
    Save a numpy array as a NIfTI file
    
    Args:
        img_data (np.ndarray): Image data to save
        output_path (str): Path where to save the file
        reference_file (nibabel.Nifti1Image, optional): Reference NIfTI file to copy header from
    """
    # If we have a reference file, copy its header
    if reference_file is not None:
        # Create new NIfTI object with the data and the header from reference
        new_img = nib.Nifti1Image(img_data, reference_file.affine, reference_file.header)
    else:
        # Create new NIfTI object with identity affine
        new_img = nib.Nifti1Image(img_data, np.eye(4))
    
    # Save the image
    nib.save(new_img, output_path)
    print(f"Saved NIfTI file to {output_path}")

def process_patches(patches, original_shape):
    """
    Reconstruct full image from patches
    
    Args:
        patches (np.ndarray): Array of patches with shape (n_patches, patch_size, patch_size, patch_size, 1)
        original_shape (tuple): Original shape of the image (depth, height, width)
        
    Returns:
        np.ndarray: Reconstructed full image
    """
    patch_size = patches.shape[1]  # Get patch size (assuming cube patches)
    
    # Calculate how many patches fit in each dimension
    n_patches_d = original_shape[0] // patch_size
    n_patches_h = original_shape[1] // patch_size
    n_patches_w = original_shape[2] // patch_size
    
    # Calculate final shape that will contain complete patches
    d_trunc = n_patches_d * patch_size
    h_trunc = n_patches_h * patch_size
    w_trunc = n_patches_w * patch_size
    
    # Initialize result array with zeros
    result = np.zeros((d_trunc, h_trunc, w_trunc))
    
    # Reshape patches to prepare for reconstruction
    patches = patches.reshape(n_patches_d, n_patches_h, n_patches_w, patch_size, patch_size, patch_size, 1)
    patches = patches.squeeze(-1)  # Remove channel dimension
    
    # Fill the result array with patches
    for d in range(n_patches_d):
        for h in range(n_patches_h):
            for w in range(n_patches_w):
                result[d*patch_size:(d+1)*patch_size, 
                       h*patch_size:(h+1)*patch_size, 
                       w*patch_size:(w+1)*patch_size] = patches[d, h, w]
                
    return result

def denormalize(img):
    """
    Convert from normalized [-1, 1] range back to original range
    
    Args:
        img (np.ndarray): Normalized image
        
    Returns:
        np.ndarray: Denormalized image
    """
    return (img + 1) / 2

def generate_high_res(low_res_file, output_file):
    """
    Generate high-resolution MRI from a low-resolution input file
    
    Args:
        low_res_file (str): Path to low-resolution NIfTI file
        output_file (str): Path to save the high-resolution output
    """
    print(f"Processing {low_res_file}...")
    
    # Load the low-res file
    lr_nifti = nib.load(low_res_file)
    lr_data = lr_nifti.get_fdata().astype('float32')
    original_shape = lr_data.shape
    
    # Normalize the data
    lr_data = lr_data - np.min(lr_data)
    lr_data = (lr_data / np.max(lr_data)) * 2 - 1
    
    # Create data generator to extract patches
    data_processor = SuperResDataGenerator("", "", patch_size=16)
    lr_patches = data_processor._extract_patches(lr_data, 16)
    
    print(f"Extracted {len(lr_patches)} patches of size 16x16x16")
    
    # Load the generator model
    generator.load_weights("models/weights/dc_wgan_low/generator_dc_wgan_low_sr.h5")
    
    # Process patches in batches to avoid memory issues
    batch_size = 8
    n_batches = (len(lr_patches) + batch_size - 1) // batch_size
    
    hr_patches = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(lr_patches))
        
        # Generate random noise vectors as input to generator
        z_vectors = tf.random.normal((end_idx - start_idx, 10))
        
        # Generate high-res patches
        hr_batch = generator(z_vectors).numpy()
        hr_patches.append(hr_batch)
        
        print(f"Processed batch {i+1}/{n_batches}")
    
    # Combine all patches
    hr_patches = np.vstack(hr_patches)
    
    # Reconstruct the full image
    hr_image = process_patches(hr_patches, original_shape)
    
    # Denormalize
    hr_image = denormalize(hr_image)
    
    # Save as NIfTI file
    save_nifti(hr_image, output_file, lr_nifti)
    
    print(f"Successfully generated high-resolution MRI: {output_file}")

if __name__ == "__main__":
    # Path to low-res file
    low_res_file = "lowres_CC0001_philips_15_55_M.nii"
    
    # Output file
    output_file = "highres_CC0001_philips_15_55_M.nii"
    
    # Generate high-res MRI
    generate_high_res(low_res_file, output_file) 