"""
ComfyUI-SAM-Body4D Constants

Global constants for model parameters, timeouts, and joint counts.
"""

# Operational timeouts (seconds)
BLENDER_TIMEOUT = 300  # 5 minutes for FBX export with animation
SAM3_TIMEOUT = 1800    # 30 minutes for SAM-3 propagation
BODY4D_TIMEOUT = 3600  # 60 minutes for full 4D generation

# MHR Model Parameters
NUM_JOINTS_MHR70 = 70  # MHR70 subset (backwards compatible)
NUM_JOINTS_MHR127 = 127  # Full MHR skeleton
NUM_JOINTS = NUM_JOINTS_MHR70  # Default to MHR70 for backwards compatibility

BODY_JOINTS = 17
FOOT_JOINTS = 6
HAND_JOINTS = 40  # 20 per hand
EXTRA_JOINTS = 7

# Joint subsets for export
JOINT_SUBSETS = {
    "full_127": list(range(127)),  # Full MHR skeleton (127 joints)
    "full_70": list(range(70)),    # MHR70 subset (70 joints)
    "body_17": list(range(17)),    # Body only (17 joints)
    "body_hands": list(range(17)) + list(range(21, 62)),  # Body + both hands
}

# Export formats
SUPPORTED_EXPORT_FORMATS = ["fbx"]

# Default values
DEFAULT_FPS = 30.0
DEFAULT_BATCH_SIZE = 64
DEFAULT_DETECTION_THRESHOLD = 0.5
DEFAULT_MAX_PERSONS = 5
