"""
ComfyUI-SAM-Body4D: Image Sequence to Animated FBX

Custom node suite for converting image sequences to animated FBX files
with skeleton rigging using SAM-Body4D pipeline.

Main Nodes:
- LoadBody4DModel: Load SAM-Body4D pipeline components
- Body4DImageSequence: Validate and prepare image sequence
- Body4DProcess: Run full Body4D pipeline (tracking + 3D mesh)
- Body4DSkeletonExtract: Extract skeleton animation from sequence
- Body4DExportFBX: Export animated FBX with Blender integration

Author: ComfyUI-SAM-Body4D
Version: 1.0.0
License: MIT
"""

import os
import sys
import traceback

# Version info
__version__ = "1.0.0"

# Track initialization status
INIT_SUCCESS = False
INIT_ERRORS = []

# Detect if running under pytest
force_init = os.environ.get('BODY4D_FORCE_INIT') == '1'
is_pytest = 'PYTEST_CURRENT_TEST' in os.environ
skip_init = is_pytest and not force_init

if not skip_init:
    print(f"[SAM-Body4D] ComfyUI-SAM-Body4D v{__version__} initializing...")

    # Step 0: Register body4d model folder with ComfyUI
    try:
        import folder_paths
        body4d_model_dir = os.path.join(folder_paths.models_dir, "body4d")
        os.makedirs(body4d_model_dir, exist_ok=True)
        folder_paths.add_model_folder_path("body4d", body4d_model_dir)
        print(f"[SAM-Body4D] [OK] Registered model folder: {body4d_model_dir}")
    except Exception as e:
        error_msg = f"Failed to register model folder: {str(e)}"
        INIT_ERRORS.append(error_msg)
        print(f"[SAM-Body4D] [WARNING] {error_msg}")

    # Step 1: Import node classes
    try:
        from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        print("[SAM-Body4D] [OK] Node classes imported successfully")
        INIT_SUCCESS = True
    except Exception as e:
        error_msg = f"Failed to import node classes: {str(e)}"
        INIT_ERRORS.append(error_msg)
        print(f"[SAM-Body4D] [WARNING] {error_msg}")
        print(f"[SAM-Body4D] Traceback:\n{traceback.format_exc()}")

        # Set empty mappings if import failed
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}

    # Report final status
    if INIT_SUCCESS:
        print(f"[SAM-Body4D] [OK] Loaded successfully!")
        print(f"[SAM-Body4D] Available nodes: {', '.join(NODE_CLASS_MAPPINGS.keys())}")
    else:
        print(f"[SAM-Body4D] [ERROR] Failed to load ({len(INIT_ERRORS)} error(s)):")
        for error in INIT_ERRORS:
            print(f"  - {error}")
        print("[SAM-Body4D] Please check the errors above and your installation.")

else:
    # During testing, skip initialization to prevent import errors
    print(f"[SAM-Body4D] ComfyUI-SAM-Body4D v{__version__} running in pytest mode - skipping initialization")
    print(f"[SAM-Body4D] Reason: PYTEST_CURRENT_TEST={os.environ.get('PYTEST_CURRENT_TEST')}")
    print(f"[SAM-Body4D] If this is a false positive, set environment variable: BODY4D_FORCE_INIT=1")

    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Web directory for custom UI (future)
WEB_DIRECTORY = "./web"

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
