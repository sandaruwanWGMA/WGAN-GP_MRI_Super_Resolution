# 3D MRI Imaging Super-Resolution using GANs

Super resolution, or SR, is the process by which a low resolution (LR) image is transformed into another high resolution (HR) version of itself. The super resolution has captured the attention of many scientific fields, but most especially, those seeking its application in the field of medicine. Obtaining quality images during diagnosis is vital for the correct detection and treatment of many pathologies. Specifically, images of a three-dimensional nature obtained through magnetic resonance imaging (MRI) are characterized by a relative low quality due to factors inherent to their acquisition, such as variations in the penetration of magnetic waves in different tissues, strong signal / noise ratios and the patient's own movement.

![MRI](https://lh3.googleusercontent.com/proxy/w6WPAW4AUqUPNVT601N3CS2lKnBgtaE6L-1Oqj3T4dif1D1p-AE7rCDuq_FUEx-IsWGB8NIG3j1lkPswRVwLsogvO-l1zligrLYhMHmcZBHk6UhyNR-3qth_r6NWJ1lVtRp031LG8n_V6v9B5rA4k-gxeitMM12l-U8tTWJgKxQG71_8yeA)

The great advance of Deep Learning in recent years has allowed the application of artificial neural networks, especially those with dense and convolutional architectures, in super-resolution techniques. These networks have been the reference in obtaining high resolution images from lower quality images. However, the appearance in 2014 of antagonistic generative networks, or GANs for its acronym in English, has been a before and after in the field of generative models. In many cases reaching to show the state of the art in a wide range of applications.
For all the aforementioned, it is necessary to investigate how GAN can be applied in super resolution imaging techniques, and more specifically, 3D medical images obtained with MRI, and the possible benefits that this can bring on the detection and treatment of pathologies. neurological.

## Simple GAN

Animation of how Generator NN learns to estimate Real Data Distribution.

![GAN_Animation](resources/animations/GAN_animation.gif)

## Generating High-Resolution MRI from NIFTI Files

This repository now includes a script to generate high-resolution MRI volumes from arbitrary-sized NIFTI inputs using a non-overlapping patch-based approach.

### Prerequisites

- Python 3.6+
- TensorFlow 2.x
- nibabel
- numpy
- tqdm

### Usage

```bash
python generate_high_res.py --input input.nii --output output.nii --factor 2 --batch_size 8
```

### Arguments

- `--input`: Path to input NIFTI file (required)
- `--output`: Path to output high-resolution NIFTI file (required)
- `--model`: Path to generator model (default: models/weights/dc_wgan_low/generator_dc_wgan_low.h5)
- `--factor`: Upscaling factor (default: 2)
- `--batch_size`: Number of patches to process at once (default: 8)

### How It Works

1. The script loads the input NIFTI volume and divides it into non-overlapping 16x16x16 patches
2. Each patch is processed through the pre-trained generator model
3. The high-resolution patches are reassembled into the final volume
4. The output is saved as a NIFTI file with adjusted voxel spacing to reflect the upscaling

This approach allows for processing MRI volumes of any size, not just those with dimensions that are multiples of 16.