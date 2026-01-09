"""
ComfyUI-SAM-Body4D Nodes

Aggregates all node mappings from processing modules.
"""

import os
import sys
from pathlib import Path

# CRITICAL: Setup sam-body4d paths BEFORE importing any nodes
# This ensures all imports can find sam-body4d modules correctly
# sam-body4d is a sibling directory in custom_nodes/
# Path: nodes/__init__.py -> nodes/ -> ComfyUI-SAM-Body4D/ -> custom_nodes/ -> sam-body4d/
SAM_BODY4D_PATH = Path(__file__).parent.parent.parent / "sam-body4d"
if SAM_BODY4D_PATH.exists():
    # Package structure: models/sam_3d_body/sam_3d_body (inner package)
    # Outer __init__.py is empty, so we add parent directory
    paths_to_add = [
        str(SAM_BODY4D_PATH),
        str(SAM_BODY4D_PATH / "models" / "sam_3d_body"),  # Parent of inner package
        str(SAM_BODY4D_PATH / "models" / "diffusion_vas"),
    ]

    # Remove conflicting sam_3d_body paths from other custom nodes
    paths_to_remove = []
    for existing_path in sys.path:
        if 'sam_3d_body' in existing_path.lower() or 'sam3d' in existing_path.lower():
            if existing_path not in paths_to_add:
                paths_to_remove.append(existing_path)

    for path in paths_to_remove:
        sys.path.remove(path)

    # Add our paths at the beginning
    for path in reversed(paths_to_add):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

# Now import the nodes
from .input.load_image_batch import NODE_CLASS_MAPPINGS as INPUT_MAPPINGS
from .input.load_image_batch import NODE_DISPLAY_NAME_MAPPINGS as INPUT_DISPLAY_MAPPINGS
from .processing.load_model import NODE_CLASS_MAPPINGS as LOAD_MODEL_MAPPINGS
from .processing.load_model import NODE_DISPLAY_NAME_MAPPINGS as LOAD_MODEL_DISPLAY_MAPPINGS
from .processing.image_sequence import NODE_CLASS_MAPPINGS as IMAGE_SEQ_MAPPINGS
from .processing.image_sequence import NODE_DISPLAY_NAME_MAPPINGS as IMAGE_SEQ_DISPLAY_MAPPINGS
from .processing.body4d_process import NODE_CLASS_MAPPINGS as PROCESS_MAPPINGS
from .processing.body4d_process import NODE_DISPLAY_NAME_MAPPINGS as PROCESS_DISPLAY_MAPPINGS
from .processing.skeleton_extract import NODE_CLASS_MAPPINGS as SKELETON_MAPPINGS
from .processing.skeleton_extract import NODE_DISPLAY_NAME_MAPPINGS as SKELETON_DISPLAY_MAPPINGS
from .processing.fbx_export import NODE_CLASS_MAPPINGS as EXPORT_MAPPINGS
from .processing.fbx_export import NODE_DISPLAY_NAME_MAPPINGS as EXPORT_DISPLAY_MAPPINGS

# Combine all node mappings
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(INPUT_MAPPINGS)
NODE_CLASS_MAPPINGS.update(LOAD_MODEL_MAPPINGS)
NODE_CLASS_MAPPINGS.update(IMAGE_SEQ_MAPPINGS)
NODE_CLASS_MAPPINGS.update(PROCESS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(SKELETON_MAPPINGS)
NODE_CLASS_MAPPINGS.update(EXPORT_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(INPUT_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(LOAD_MODEL_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(IMAGE_SEQ_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(PROCESS_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(SKELETON_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(EXPORT_DISPLAY_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
