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
            pattern: Glob pattern for matching files (supports #### for digit placeholders)
            start_index: Skip first N images
            max_frames: Limit number of frames (0 = no limit)

        Returns:
            Tuple of (images tensor, frame_count)
        """
        import glob
        import re

        # Resolve directory path
        if not os.path.isabs(directory):
            # Try relative to ComfyUI input directory
            input_dir = folder_paths.get_input_directory()
            directory = os.path.join(input_dir, directory)

        if not os.path.exists(directory):
            raise ValueError(f"Directory not found: {directory}")

        # Convert #### pattern to glob pattern
        # e.g., "frame_####.png" -> "frame_*.png"
        glob_pattern = re.sub(r'#+', '*', pattern)

        # Find all matching files
        search_pattern = os.path.join(directory, glob_pattern)
        image_files = sorted(glob.glob(search_pattern))

        print(f"[Body4D] Debug: directory={directory}")
        print(f"[Body4D] Debug: pattern={pattern}")
        print(f"[Body4D] Debug: glob_pattern={glob_pattern}")
        print(f"[Body4D] Debug: search_pattern={search_pattern}")
        print(f"[Body4D] Debug: found {len(image_files)} files with glob")

        # If original pattern had ####, filter results with regex
        if '#' in pattern:
            # Convert pattern to regex: "frame_####.png" -> "frame_\d{4}\.png"
            regex_pattern = pattern.replace('.', r'\.')
            regex_pattern = re.sub(r'(#+)', lambda m: r'\d{' + str(len(m.group(1))) + '}', regex_pattern)
            regex_pattern = '^' + regex_pattern + '$'

            print(f"[Body4D] Debug: regex_pattern={regex_pattern}")

            filtered_files = []
            for f in image_files:
                basename = os.path.basename(f)
                matches = re.match(regex_pattern, basename)
                print(f"[Body4D] Debug: {basename} -> {'MATCH' if matches else 'NO MATCH'}")
                if matches:
                    filtered_files.append(f)

            image_files = filtered_files

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
        failed_images = []
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img_np = np.array(img).astype(np.float32) / 255.0
                images.append(img_np)
            except Exception as e:
                failed_images.append((img_path, str(e)))
                print(f"[Body4D] Warning: Failed to load {img_path}: {e}")

        if not images:
            error_msg = f"No images could be loaded. Found {len(image_files)} files but all failed to load."
            if failed_images:
                error_msg += f"\nFirst error: {failed_images[0][1]}"
            raise ValueError(error_msg)

        if failed_images:
            print(f"[Body4D] Warning: {len(failed_images)} images failed to load")

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
