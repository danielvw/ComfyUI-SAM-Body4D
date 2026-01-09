"""
ComfyUI-SAM-Body4D Nodes

Aggregates all node mappings from processing modules.
"""

import os
import sys
from pathlib import Path

# CRITICAL: Setup sam-body4d paths BEFORE importing any nodes
# This ensures all imports can find sam-body4d modules correctly
SAM_BODY4D_PATH = Path(__file__).parent.parent / "sam-body4d"
if SAM_BODY4D_PATH.exists():
    # IMPORTANT: sam_3d_body package structure is models/sam_3d_body/sam_3d_body/
    # We need to add the INNER sam_3d_body directory to sys.path
    paths_to_add = [
        str(SAM_BODY4D_PATH),
        str(SAM_BODY4D_PATH / "models"),
        str(SAM_BODY4D_PATH / "models" / "sam_3d_body" / "sam_3d_body"),  # INNER package
        str(SAM_BODY4D_PATH / "models" / "diffusion_vas"),
    ]
    for path in reversed(paths_to_add):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

# Now import the nodes
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
NODE_CLASS_MAPPINGS.update(LOAD_MODEL_MAPPINGS)
NODE_CLASS_MAPPINGS.update(IMAGE_SEQ_MAPPINGS)
NODE_CLASS_MAPPINGS.update(PROCESS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(SKELETON_MAPPINGS)
NODE_CLASS_MAPPINGS.update(EXPORT_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(LOAD_MODEL_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(IMAGE_SEQ_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(PROCESS_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(SKELETON_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(EXPORT_DISPLAY_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
