"""
Base utility functions for ComfyUI-SAM-Body4D

Tensor conversion utilities for bridging ComfyUI and SAM-Body4D formats.
"""

import numpy as np
import torch
from PIL import Image


def comfy_image_to_numpy(image_tensor):
    """
    Convert ComfyUI image tensor to NumPy array (BGR format for OpenCV).

    ComfyUI format: [B, H, W, C] float32 [0, 1]
    Output format: [H, W, C] uint8 [0, 255] BGR

    Args:
        image_tensor: ComfyUI image tensor [B, H, W, C]

    Returns:
        numpy array in BGR format for OpenCV
    """
    # Extract first image from batch
    img = image_tensor[0].cpu().numpy()

    # Scale to [0, 255]
    img = (img * 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    img_bgr = img[:, :, ::-1].copy()

    return img_bgr


def comfy_image_to_pil(image_tensor):
    """
    Convert ComfyUI image tensor to PIL Image.

    ComfyUI format: [B, H, W, C] float32 [0, 1]
    Output format: PIL Image RGB

    Args:
        image_tensor: ComfyUI image tensor [B, H, W, C]

    Returns:
        PIL Image in RGB format
    """
    # Extract first image from batch
    img = image_tensor[0].cpu().numpy()

    # Scale to [0, 255]
    img = (img * 255).astype(np.uint8)

    # Create PIL image
    pil_image = Image.fromarray(img)

    return pil_image


def numpy_to_comfy_image(img_np):
    """
    Convert NumPy array to ComfyUI image tensor.

    Input format: [H, W, C] uint8 [0, 255] (BGR or RGB)
    ComfyUI format: [1, H, W, C] float32 [0, 1] RGB

    Args:
        img_np: NumPy array (BGR or RGB)

    Returns:
        ComfyUI image tensor
    """
    # Ensure RGB (assume BGR if 3 channels)
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        # Convert BGR to RGB
        img_rgb = img_np[:, :, ::-1].copy()
    else:
        img_rgb = img_np

    # Scale to [0, 1]
    img_float = img_rgb.astype(np.float32) / 255.0

    # Add batch dimension
    img_tensor = torch.from_numpy(img_float).unsqueeze(0)

    return img_tensor


def comfy_batch_to_list(image_tensor):
    """
    Convert ComfyUI image batch to list of NumPy arrays (BGR).

    ComfyUI format: [B, H, W, C] float32 [0, 1]
    Output format: List of [H, W, C] uint8 [0, 255] BGR

    Args:
        image_tensor: ComfyUI image tensor [B, H, W, C]

    Returns:
        List of NumPy arrays in BGR format
    """
    images = []
    batch_size = image_tensor.shape[0]

    for i in range(batch_size):
        img = image_tensor[i].cpu().numpy()
        # Scale to [0, 255]
        img = (img * 255).astype(np.uint8)
        # Convert RGB to BGR
        img_bgr = img[:, :, ::-1].copy()
        images.append(img_bgr)

    return images


def list_to_comfy_batch(images_list):
    """
    Convert list of NumPy arrays to ComfyUI image batch.

    Input format: List of [H, W, C] uint8 [0, 255] (BGR or RGB)
    ComfyUI format: [B, H, W, C] float32 [0, 1] RGB

    Args:
        images_list: List of NumPy arrays

    Returns:
        ComfyUI image tensor
    """
    tensors = []

    for img_np in images_list:
        # Ensure RGB (assume BGR if 3 channels)
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_rgb = img_np[:, :, ::-1].copy()
        else:
            img_rgb = img_np

        # Scale to [0, 1]
        img_float = img_rgb.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_float)
        tensors.append(img_tensor)

    # Stack into batch
    batch_tensor = torch.stack(tensors, dim=0)

    return batch_tensor
