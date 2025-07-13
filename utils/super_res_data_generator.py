from os import path, listdir
import numpy as np
from nibabel import load as load_nib
from scipy.ndimage import zoom
import re

class SuperResDataGenerator:
    """
    Python Generator that takes pairs of high-res and low-res MRI files,
    processes them and returns batches for super-resolution training.
    Uses non-overlapping patches for training.
    """

    def __init__(self, high_res_path, low_res_path, patch_size=16, normalize=True):
        """
        Initialize the SuperResDataGenerator
        
        Args:
            high_res_path (str): Path to high-resolution images directory
            low_res_path (str): Path to low-resolution images directory
            patch_size (int): Size of non-overlapping patches
            normalize (bool): Whether to normalize images to [-1, 1]
        """
        self.high_res_path = high_res_path
        self.low_res_path = low_res_path
        self.patch_size = patch_size
        self.normalize = normalize
        
        # Get list of files (excluding hidden files)
        self.hr_files = [f for f in listdir(high_res_path) if not f.startswith('.')]
        self.lr_files = [f for f in listdir(low_res_path) if not f.startswith('.')]
        
        # Match low-res files to high-res files
        self.paired_files = self._pair_files()
        self.data_length = len(self.paired_files)
        self.data_taken = 0
        
        # Initialize generator
        self.reset_generator()
        
    def _pair_files(self):
        """
        Match low-res files with their high-res counterparts
        
        Returns:
            list: List of tuples (high_res_file, low_res_file)
        """
        paired_files = []
        
        # Create pattern to match the prefix
        prefix_pattern = re.compile(r'^lowres_(.*)')
        
        # Create mapping of base names to full filenames
        hr_dict = {file: file for file in self.hr_files}
        
        for lr_file in self.lr_files:
            # Extract base name without 'lowres_' prefix
            match = prefix_pattern.match(lr_file)
            if match:
                base_name = match.group(1)
                # Find matching high-res file
                if base_name in hr_dict:
                    paired_files.append((hr_dict[base_name], lr_file))
        
        return paired_files
    
    def _load_and_process_image(self, file_path):
        """
        Load and process a NIfTI image
        
        Args:
            file_path (str): Path to the NIfTI file
            
        Returns:
            np.ndarray: Processed 3D image
        """
        img = load_nib(file_path).get_fdata()
        img = img.astype('float32')
        
        # Normalize to [-1, 1] if required
        if self.normalize:
            img = self._normalize(img)
            
        return img
    
    def _normalize(self, img):
        """
        Normalize image to range [-1, 1]
        
        Args:
            img (np.ndarray): Input image
            
        Returns:
            np.ndarray: Normalized image
        """
        img = img - np.min(img)
        img = (img / np.max(img)) * 2 - 1
        return img
    
    def _extract_patches(self, image, patch_size):
        """
        Extract non-overlapping patches from 3D image
        
        Args:
            image (np.ndarray): 3D input image
            patch_size (int): Size of patches to extract
            
        Returns:
            np.ndarray: Array of patches with shape (n_patches, patch_size, patch_size, patch_size, 1)
        """
        # Get dimensions
        depth, height, width = image.shape
        
        # Calculate number of patches in each dimension
        n_patches_d = depth // patch_size
        n_patches_h = height // patch_size
        n_patches_w = width // patch_size
        
        # Truncate image to fit complete patches
        d_trunc = n_patches_d * patch_size
        h_trunc = n_patches_h * patch_size
        w_trunc = n_patches_w * patch_size
        
        # Reshape to extract patches
        reshaped = image[:d_trunc, :h_trunc, :w_trunc]
        patches = reshaped.reshape(n_patches_d, patch_size, n_patches_h, patch_size, n_patches_w, patch_size)
        patches = patches.transpose(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(-1, patch_size, patch_size, patch_size)
        
        # Add channel dimension
        patches = np.expand_dims(patches, axis=-1)
        
        return patches

    def get_next_pair(self):
        """
        Get next pair of high-res and low-res images
        
        Returns:
            tuple: (high_res_patches, low_res_patches)
        """
        if self.data_taken >= self.data_length:
            self.reset_generator()
            
        # Get file pair
        hr_file, lr_file = self.paired_files[self.data_taken]
        
        # Load images
        hr_img = self._load_and_process_image(path.join(self.high_res_path, hr_file))
        lr_img = self._load_and_process_image(path.join(self.low_res_path, lr_file))
        
        # Extract patches
        hr_patches = self._extract_patches(hr_img, self.patch_size)
        lr_patches = self._extract_patches(lr_img, self.patch_size)
        
        self.data_taken += 1
        
        return hr_patches, lr_patches
    
    def get_batch(self, batch_size):
        """
        Get batch of high-res and low-res patch pairs
        
        Args:
            batch_size (int): Number of patch pairs to include in batch
            
        Returns:
            tuple: (high_res_batch, low_res_batch) with shapes (batch_size, patch_size, patch_size, patch_size, 1)
        """
        hr_batch = []
        lr_batch = []
        
        patches_needed = batch_size
        while patches_needed > 0:
            hr_patches, lr_patches = self.get_next_pair()
            n_patches = len(hr_patches)
            
            if n_patches <= patches_needed:
                hr_batch.append(hr_patches)
                lr_batch.append(lr_patches)
                patches_needed -= n_patches
            else:
                hr_batch.append(hr_patches[:patches_needed])
                lr_batch.append(lr_patches[:patches_needed])
                patches_needed = 0
        
        # Concatenate all patches
        hr_batch = np.concatenate(hr_batch)[:batch_size]
        lr_batch = np.concatenate(lr_batch)[:batch_size]
        
        return hr_batch, lr_batch
    
    def reset_generator(self):
        """
        Reset the generator state
        """
        self.data_taken = 0 