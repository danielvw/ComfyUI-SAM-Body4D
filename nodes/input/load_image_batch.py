"""
LoadImageBatch Node

Loads a sequence of images from a directory for Body4D processing.
"""

import os
import folder_paths
from PIL import Image
import numpy as np
import torch


class LoadImageBatch:
    """
    Load a batch of images from a directory as an image sequence.

    Supports common image formats (PNG, JPG, etc.) and returns them
    as a ComfyUI IMAGE tensor suitable for Body4D processing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get available directories
        input_dir = folder_paths.get_input_directory()

        return {
            "required": {
                "directory": ("STRING", {
                    "default": "",
                    "tooltip": "Directory path containing image sequence (relative to ComfyUI/input or absolute path)"
                }),
                "pattern": ("STRING", {
                    "default": "*.png",
                    "tooltip": "File pattern (e.g., *.png, frame_*.jpg, image_####.png)"
                }),
            },
            "optional": {
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "First image index to load"
                }),
                "max_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Maximum number of frames to load (0 = all)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "frame_count")
    FUNCTION = "load_images"
    CATEGORY = "SAM-Body4D/Input"

    def load_images(self, directory, pattern="*.png", start_index=0, max_frames=0):
        """
        Load images from directory matching pattern.

        Args:
            directory: Path to directory (relative to input folder or absolute)
            pattern: Glob pattern for matching files
            start_index: Skip first N images
            max_frames: Limit number of frames (0 = no limit)

        Returns:
            Tuple of (images tensor, frame_count)
        """
        import glob

        # Resolve directory path
        if not os.path.isabs(directory):
            # Try relative to ComfyUI input directory
            input_dir = folder_paths.get_input_directory()
            directory = os.path.join(input_dir, directory)

        if not os.path.exists(directory):
            raise ValueError(f"Directory not found: {directory}")

        # Find all matching files
        search_pattern = os.path.join(directory, pattern)
        image_files = sorted(glob.glob(search_pattern))

        if not image_files:
            raise ValueError(f"No images found matching pattern: {search_pattern}")

        # Apply start_index
        if start_index > 0:
            image_files = image_files[start_index:]

        # Apply max_frames limit
        if max_frames > 0:
            image_files = image_files[:max_frames]

        print(f"[Body4D] Loading {len(image_files)} images from {directory}")

        # Load all images
        images = []
        for img_path in image_files:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img).astype(np.float32) / 255.0
            images.append(img_np)

        # Stack into batch tensor [B, H, W, C]
        images_tensor = torch.from_numpy(np.stack(images, axis=0))

        print(f"[Body4D] Loaded image sequence: {images_tensor.shape}")

        return (images_tensor, len(images))


# Register the node
NODE_CLASS_MAPPINGS = {
    "LoadImageBatch": LoadImageBatch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageBatch": "Load Image Batch (Body4D)"
}
