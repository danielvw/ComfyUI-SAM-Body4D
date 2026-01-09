#!/usr/bin/env python3
"""
Test script to debug import issues without restarting ComfyUI.

Usage:
    python test_imports.py
"""

import os
import sys
from pathlib import Path

print("=" * 70)
print("SAM-Body4D Import Test")
print("=" * 70)

# Show current working directory
print(f"\nCurrent working directory: {os.getcwd()}")

# Setup paths (simulate what nodes/__init__.py does)
print("\n1. Setting up sys.path...")
# sam-body4d is a sibling directory in custom_nodes/
# Path: test_imports.py -> ComfyUI-SAM-Body4D/ -> custom_nodes/ -> sam-body4d/
SAM_BODY4D_PATH = Path(__file__).parent.parent / "sam-body4d"
print(f"   SAM_BODY4D_PATH: {SAM_BODY4D_PATH}")
print(f"   Exists: {SAM_BODY4D_PATH.exists()}")

if SAM_BODY4D_PATH.exists():
    # FIXED: The package structure is: models/sam_3d_body/sam_3d_body (inner package)
    # The outer models/sam_3d_body/__init__.py is empty, so we add models/sam_3d_body to path
    # This makes 'sam_3d_body' resolve to the inner package at models/sam_3d_body/sam_3d_body/
    paths_to_add = [
        str(SAM_BODY4D_PATH),
        str(SAM_BODY4D_PATH / "models" / "sam_3d_body"),  # Parent of inner package
        str(SAM_BODY4D_PATH / "models" / "diffusion_vas"),
    ]

    print("\n   Adding paths to sys.path:")
    for path in paths_to_add:
        exists = Path(path).exists()
        print(f"   - {path} [{'EXISTS' if exists else 'MISSING'}]")

    # CRITICAL: Remove ALL existing sam_3d_body paths from other custom nodes (like ComfyUI-SAM3DBody)
    print("\n   Cleaning up conflicting paths...")
    paths_to_remove = []
    for existing_path in sys.path:
        if 'sam_3d_body' in existing_path.lower() or 'sam3d' in existing_path.lower():
            if existing_path not in paths_to_add:
                paths_to_remove.append(existing_path)
                print(f"   - Removing: {existing_path}")

    for path in paths_to_remove:
        sys.path.remove(path)

    # Now add our paths at the very beginning
    for path in reversed(paths_to_add):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    print("\n   sys.path (first 10 entries):")
    for i, path in enumerate(sys.path[:10]):
        print(f"   [{i}] {path}")
else:
    print(f"   ERROR: SAM_BODY4D_PATH does not exist!")
    sys.exit(1)

# Test 1: Import utils
print("\n2. Testing 'utils' import...")
try:
    import utils
    print(f"   ✓ SUCCESS: utils imported from {utils.__file__}")

    # Check if it has the expected functions
    if hasattr(utils, 'kalman_smooth_mhr_params_per_obj_id_adaptive'):
        print(f"   ✓ Found kalman_smooth_mhr_params_per_obj_id_adaptive")
    else:
        print(f"   ✗ WARNING: kalman_smooth_mhr_params_per_obj_id_adaptive not found")
        print(f"   Available: {dir(utils)}")
except ImportError as e:
    print(f"   ✗ FAILED: {e}")

# Test 2: Import sam_3d_body package
print("\n3. Testing 'sam_3d_body' package import...")
try:
    import sam_3d_body
    print(f"   ✓ SUCCESS: sam_3d_body imported from {sam_3d_body.__file__}")
except ImportError as e:
    print(f"   ✗ FAILED: {e}")

# Test 3: Import sam_3d_body.data.transforms
print("\n4. Testing 'sam_3d_body.data.transforms' import...")
try:
    from sam_3d_body.data.transforms import (
        ResizeToMaxSize,
        PadToSquare,
        ToTensor,
        Normalize,
    )
    print(f"   ✓ SUCCESS: sam_3d_body.data.transforms imported")
    print(f"   ✓ ResizeToMaxSize available")
except ImportError as e:
    print(f"   ✗ FAILED: {e}")

    # Debug: try to find where the issue is
    print("\n   Debugging:")
    try:
        import sam_3d_body.data
        print(f"   - sam_3d_body.data OK: {sam_3d_body.data.__file__}")
    except ImportError as e2:
        print(f"   - sam_3d_body.data FAILED: {e2}")

# Test 4: Import models.sam_3d_body.sam_3d_body
print("\n5. Testing 'models.sam_3d_body.sam_3d_body' import...")
try:
    from models.sam_3d_body.sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    print(f"   ✓ SUCCESS: models.sam_3d_body.sam_3d_body imported")
except ImportError as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    print("\n   Full traceback:")
    traceback.print_exc()

# Test 5: Check specific files exist
print("\n6. Checking critical files exist...")
critical_files = [
    SAM_BODY4D_PATH / "utils" / "__init__.py",
    SAM_BODY4D_PATH / "utils" / "kalman.py",
    SAM_BODY4D_PATH / "models" / "sam_3d_body" / "sam_3d_body" / "__init__.py",
    SAM_BODY4D_PATH / "models" / "sam_3d_body" / "sam_3d_body" / "data" / "__init__.py",
    SAM_BODY4D_PATH / "models" / "sam_3d_body" / "sam_3d_body" / "data" / "transforms" / "__init__.py",
]

for file_path in critical_files:
    exists = file_path.exists()
    status = "✓" if exists else "✗"
    print(f"   {status} {file_path.relative_to(SAM_BODY4D_PATH)}")

# Test 6: Check config path resolution
print("\n7. Testing config path resolution...")
try:
    # Simulate what load_model.py does for default config
    config_base_path = Path(__file__).parent / "nodes" / "processing"
    default_config = config_base_path.parent.parent / "configs" / "body4d.yaml"
    print(f"   Default config path: {default_config}")
    print(f"   Exists: {default_config.exists()}")

    if default_config.exists():
        print(f"   ✓ Config file found at correct location")
    else:
        print(f"   ✗ Config file NOT found")
        print(f"   Expected at: {default_config.absolute()}")

        # Check alternative locations
        alt_locations = [
            Path(__file__).parent / "configs" / "body4d.yaml",
            SAM_BODY4D_PATH / "configs" / "body4d.yaml",
        ]
        print(f"\n   Checking alternative locations:")
        for loc in alt_locations:
            exists = loc.exists()
            status = "✓" if exists else "✗"
            print(f"   {status} {loc}")
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Test complete!")
print("=" * 70)
