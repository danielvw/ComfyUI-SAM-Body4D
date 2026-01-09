"""
Body4DImageSequence Node

Validates and prepares image batch for Body4D processing.
"""

import torch


class Body4DImageSequence:
    """
    Validate and prepare image sequence for Body4D.

    Ensures ComfyUI image tensor is in correct format [B, H, W, C]
    and provides frame slicing capabilities.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # ComfyUI [B,H,W,C] Tensor
            },
            "optional": {
                "fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 120.0,
                    "tooltip": "Frames per second for output animation"
                }),
                "start_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "First frame to process (0-indexed)"
                }),
                "end_frame": ("INT", {
                    "default": -1,
                    "tooltip": "Last frame to process (-1 for all frames)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "FLOAT")
    RETURN_NAMES = ("images", "frame_count", "fps")
    FUNCTION = "prepare_sequence"
    CATEGORY = "SAM-Body4D/Input"

    def prepare_sequence(self, images, fps=30.0, start_frame=0, end_frame=-1):
        """
        Prepare image sequence for Body4D processing.

        Args:
            images: ComfyUI image tensor [B, H, W, C]
            fps: Frames per second
            start_frame: First frame to process
            end_frame: Last frame to process (-1 for all)

        Returns:
            Tuple of (images, frame_count, fps)
        """
        # Validate tensor shape
        if len(images.shape) != 4:
            raise ValueError(
                f"Invalid image tensor shape: {images.shape}. "
                f"Expected [B, H, W, C] format."
            )

        batch_size, height, width, channels = images.shape

        # Validate channels
        if channels != 3:
            raise ValueError(
                f"Invalid number of channels: {channels}. "
                f"Expected 3 (RGB)."
            )

        # Apply frame slicing
        if end_frame == -1:
            end_frame = batch_size

        if start_frame < 0 or start_frame >= batch_size:
            raise ValueError(
                f"Invalid start_frame: {start_frame}. "
                f"Must be in range [0, {batch_size-1}]."
            )

        if end_frame <= start_frame or end_frame > batch_size:
            raise ValueError(
                f"Invalid end_frame: {end_frame}. "
                f"Must be in range ({start_frame}, {batch_size}]."
            )

        # Slice the tensor
        images_sliced = images[start_frame:end_frame]
        frame_count = images_sliced.shape[0]

        print(f"[Body4D] Image sequence prepared:")
        print(f"  - Resolution: {width}x{height}")
        print(f"  - Frames: {frame_count} (from {start_frame} to {end_frame-1})")
        print(f"  - FPS: {fps}")
        print(f"  - Duration: {frame_count/fps:.2f}s")

        return (images_sliced, frame_count, fps)


# Register the node
NODE_CLASS_MAPPINGS = {
    "Body4DImageSequence": Body4DImageSequence
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Body4DImageSequence": "Body4D Image Sequence"
}
