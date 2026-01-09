"""
Skeleton Utilities

Defines MHR70 bone hierarchy and skeleton structure for Blender export.
"""

# MHR70 Bone Hierarchy (70 joints)
# Based on sam-body4d/models/sam_3d_body/sam_3d_body/metadata/mhr70.py

MHR70_BONE_NAMES = [
    "nose",                          # 0
    "left_eye",                      # 1
    "right_eye",                     # 2
    "left_ear",                      # 3
    "right_ear",                     # 4
    "left_shoulder",                 # 5
    "right_shoulder",                # 6
    "left_elbow",                    # 7
    "right_elbow",                   # 8
    "left_hip",                      # 9
    "right_hip",                     # 10
    "left_knee",                     # 11
    "right_knee",                    # 12
    "left_ankle",                    # 13
    "right_ankle",                   # 14
    "left_big_toe",                  # 15
    "left_small_toe",                # 16
    "left_heel",                     # 17
    "right_big_toe",                 # 18
    "right_small_toe",               # 19
    "right_heel",                    # 20
    "right_thumb4",                  # 21
    "right_thumb3",                  # 22
    "right_thumb2",                  # 23
    "right_thumb_third_joint",       # 24
    "right_forefinger4",             # 25
    "right_forefinger3",             # 26
    "right_forefinger2",             # 27
    "right_forefinger_third_joint",  # 28
    "right_middle_finger4",          # 29
    "right_middle_finger3",          # 30
    "right_middle_finger2",          # 31
    "right_middle_finger_third_joint", # 32
    "right_ring_finger4",            # 33
    "right_ring_finger3",            # 34
    "right_ring_finger2",            # 35
    "right_ring_finger_third_joint", # 36
    "right_pinky_finger4",           # 37
    "right_pinky_finger3",           # 38
    "right_pinky_finger2",           # 39
    "right_pinky_finger_third_joint", # 40
    "right_wrist",                   # 41
    "left_thumb4",                   # 42
    "left_thumb3",                   # 43
    "left_thumb2",                   # 44
    "left_thumb_third_joint",        # 45
    "left_forefinger4",              # 46
    "left_forefinger3",              # 47
    "left_forefinger2",              # 48
    "left_forefinger_third_joint",   # 49
    "left_middle_finger4",           # 50
    "left_middle_finger3",           # 51
    "left_middle_finger2",           # 52
    "left_middle_finger_third_joint", # 53
    "left_ring_finger4",             # 54
    "left_ring_finger3",             # 55
    "left_ring_finger2",             # 56
    "left_ring_finger_third_joint",  # 57
    "left_pinky_finger4",            # 58
    "left_pinky_finger3",            # 59
    "left_pinky_finger2",            # 60
    "left_pinky_finger_third_joint", # 61
    "left_wrist",                    # 62
    "left_olecranon",                # 63
    "right_olecranon",               # 64
    "left_cubital_fossa",            # 65
    "right_cubital_fossa",           # 66
    "left_acromion",                 # 67
    "right_acromion",                # 68
    "neck",                          # 69
]

# Parent relationships (child -> parent index)
# -1 means root (no parent)
MHR70_BONE_PARENTS = {
    0: 69,    # nose -> neck
    1: 0,     # left_eye -> nose
    2: 0,     # right_eye -> nose
    3: 1,     # left_ear -> left_eye
    4: 2,     # right_ear -> right_eye
    5: 67,    # left_shoulder -> left_acromion
    6: 68,    # right_shoulder -> right_acromion
    7: 5,     # left_elbow -> left_shoulder
    8: 6,     # right_elbow -> right_shoulder
    9: -1,    # left_hip -> root
    10: -1,   # right_hip -> root
    11: 9,    # left_knee -> left_hip
    12: 10,   # right_knee -> right_hip
    13: 11,   # left_ankle -> left_knee
    14: 12,   # right_ankle -> right_knee
    15: 13,   # left_big_toe -> left_ankle
    16: 13,   # left_small_toe -> left_ankle
    17: 13,   # left_heel -> left_ankle
    18: 14,   # right_big_toe -> right_ankle
    19: 14,   # right_small_toe -> right_ankle
    20: 14,   # right_heel -> right_ankle
    21: 22,   # right_thumb4 -> right_thumb3
    22: 23,   # right_thumb3 -> right_thumb2
    23: 24,   # right_thumb2 -> right_thumb_third_joint
    24: 41,   # right_thumb_third_joint -> right_wrist
    25: 26,   # right_forefinger4 -> right_forefinger3
    26: 27,   # right_forefinger3 -> right_forefinger2
    27: 28,   # right_forefinger2 -> right_forefinger_third_joint
    28: 41,   # right_forefinger_third_joint -> right_wrist
    29: 30,   # right_middle_finger4 -> right_middle_finger3
    30: 31,   # right_middle_finger3 -> right_middle_finger2
    31: 32,   # right_middle_finger2 -> right_middle_finger_third_joint
    32: 41,   # right_middle_finger_third_joint -> right_wrist
    33: 34,   # right_ring_finger4 -> right_ring_finger3
    34: 35,   # right_ring_finger3 -> right_ring_finger2
    35: 36,   # right_ring_finger2 -> right_ring_finger_third_joint
    36: 41,   # right_ring_finger_third_joint -> right_wrist
    37: 38,   # right_pinky_finger4 -> right_pinky_finger3
    38: 39,   # right_pinky_finger3 -> right_pinky_finger2
    39: 40,   # right_pinky_finger2 -> right_pinky_finger_third_joint
    40: 41,   # right_pinky_finger_third_joint -> right_wrist
    41: 8,    # right_wrist -> right_elbow
    42: 43,   # left_thumb4 -> left_thumb3
    43: 44,   # left_thumb3 -> left_thumb2
    44: 45,   # left_thumb2 -> left_thumb_third_joint
    45: 62,   # left_thumb_third_joint -> left_wrist
    46: 47,   # left_forefinger4 -> left_forefinger3
    47: 48,   # left_forefinger3 -> left_forefinger2
    48: 49,   # left_forefinger2 -> left_forefinger_third_joint
    49: 62,   # left_forefinger_third_joint -> left_wrist
    50: 51,   # left_middle_finger4 -> left_middle_finger3
    51: 52,   # left_middle_finger3 -> left_middle_finger2
    52: 53,   # left_middle_finger2 -> left_middle_finger_third_joint
    53: 62,   # left_middle_finger_third_joint -> left_wrist
    54: 55,   # left_ring_finger4 -> left_ring_finger3
    55: 56,   # left_ring_finger3 -> left_ring_finger2
    56: 57,   # left_ring_finger2 -> left_ring_finger_third_joint
    57: 62,   # left_ring_finger_third_joint -> left_wrist
    58: 59,   # left_pinky_finger4 -> left_pinky_finger3
    59: 60,   # left_pinky_finger3 -> left_pinky_finger2
    60: 61,   # left_pinky_finger2 -> left_pinky_finger_third_joint
    61: 62,   # left_pinky_finger_third_joint -> left_wrist
    62: 7,    # left_wrist -> left_elbow
    63: 7,    # left_olecranon -> left_elbow
    64: 8,    # right_olecranon -> right_elbow
    65: 7,    # left_cubital_fossa -> left_elbow
    66: 8,    # right_cubital_fossa -> right_elbow
    67: 69,   # left_acromion -> neck
    68: 69,   # right_acromion -> neck
    69: -1,   # neck -> root (spine)
}


def get_bone_hierarchy():
    """
    Get bone hierarchy as dict: bone_name -> parent_bone_name.

    Returns:
        dict: {bone_name: parent_bone_name or None}
    """
    hierarchy = {}
    for idx, bone_name in enumerate(MHR70_BONE_NAMES):
        parent_idx = MHR70_BONE_PARENTS[idx]
        if parent_idx == -1:
            hierarchy[bone_name] = None
        else:
            hierarchy[bone_name] = MHR70_BONE_NAMES[parent_idx]
    return hierarchy
