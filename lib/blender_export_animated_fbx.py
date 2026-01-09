#!/usr/bin/env python3
"""
Blender Script for Animated FBX Export

This script creates a Blender armature from MHR70 skeleton data,
applies animation keyframes, and exports to FBX format.

Usage:
    blender --background --python blender_export_animated_fbx.py -- \
        skeleton.json output.fbx fps [mesh.obj]
"""

import sys
import json
import bpy
from mathutils import Vector, Quaternion, Matrix


# MHR70 Bone Hierarchy (same as skeleton_utils.py)
MHR70_BONE_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
    "right_thumb4", "right_thumb3", "right_thumb2", "right_thumb_third_joint",
    "right_forefinger4", "right_forefinger3", "right_forefinger2", "right_forefinger_third_joint",
    "right_middle_finger4", "right_middle_finger3", "right_middle_finger2", "right_middle_finger_third_joint",
    "right_ring_finger4", "right_ring_finger3", "right_ring_finger2", "right_ring_finger_third_joint",
    "right_pinky_finger4", "right_pinky_finger3", "right_pinky_finger2", "right_pinky_finger_third_joint",
    "right_wrist",
    "left_thumb4", "left_thumb3", "left_thumb2", "left_thumb_third_joint",
    "left_forefinger4", "left_forefinger3", "left_forefinger2", "left_forefinger_third_joint",
    "left_middle_finger4", "left_middle_finger3", "left_middle_finger2", "left_middle_finger_third_joint",
    "left_ring_finger4", "left_ring_finger3", "left_ring_finger2", "left_ring_finger_third_joint",
    "left_pinky_finger4", "left_pinky_finger3", "left_pinky_finger2", "left_pinky_finger_third_joint",
    "left_wrist",
    "left_olecranon", "right_olecranon", "left_cubital_fossa", "right_cubital_fossa",
    "left_acromion", "right_acromion", "neck",
]

MHR70_BONE_PARENTS = {
    0: 69, 1: 0, 2: 0, 3: 1, 4: 2, 5: 67, 6: 68, 7: 5, 8: 6,
    9: -1, 10: -1, 11: 9, 12: 10, 13: 11, 14: 12,
    15: 13, 16: 13, 17: 13, 18: 14, 19: 14, 20: 14,
    21: 22, 22: 23, 23: 24, 24: 41,
    25: 26, 26: 27, 27: 28, 28: 41,
    29: 30, 30: 31, 31: 32, 32: 41,
    33: 34, 34: 35, 35: 36, 36: 41,
    37: 38, 38: 39, 39: 40, 40: 41, 41: 8,
    42: 43, 43: 44, 44: 45, 45: 62,
    46: 47, 47: 48, 48: 49, 49: 62,
    50: 51, 51: 52, 52: 53, 53: 62,
    54: 55, 55: 56, 56: 57, 57: 62,
    58: 59, 59: 60, 60: 61, 61: 62, 62: 7,
    63: 7, 64: 8, 65: 7, 66: 8, 67: 69, 68: 69, 69: -1,
}


def clear_scene():
    """Clear all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def create_armature(skeleton_data):
    """
    Create armature from skeleton data.

    Args:
        skeleton_data: Dict with 'frames' key containing animation data

    Returns:
        armature: Blender armature object
    """
    print("[Blender] Creating armature...")

    # Create armature
    bpy.ops.object.armature_add()
    armature = bpy.context.object
    armature.name = "Body4D_Armature"

    # Remove default bone
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.armature.select_all(action='SELECT')
    bpy.ops.armature.delete()

    # Get first frame for bone positions
    first_frame = skeleton_data['frames'][0]
    joint_positions = first_frame['joint_positions']

    # Get joint indices (which MHR70 bones are included)
    # If not specified, assume all 70 bones
    joint_indices = skeleton_data.get('joint_indices', list(range(70)))
    joint_count = skeleton_data.get('joint_count', len(joint_positions))

    print(f"[Blender] Joint subset: {skeleton_data.get('joint_subset', 'full_70')}")
    print(f"[Blender] Joint count: {joint_count}")

    # Create bones with hierarchy
    bones_created = {}

    # First pass: create bones for the selected joints
    for local_idx, mhr70_idx in enumerate(joint_indices):
        bone_name = MHR70_BONE_NAMES[mhr70_idx]

        # Create bone
        bone = armature.data.edit_bones.new(bone_name)

        # Get position from first frame and convert coordinate system
        # After flip in Python: (-X, -Y, -Z)
        # Blender transform: X stays, Y <- -Z, Z <- Y
        # This matches ComfyUI-SAM3DBody reference implementation
        src_pos = joint_positions[local_idx]
        pos = Vector((src_pos[0], -src_pos[2], src_pos[1]))

        bone.head = pos

        # Set tail (slightly offset from head in bone's direction)
        # For better visualization, offset along Z (up in Blender)
        bone.tail = pos + Vector((0, 0, 0.03))

        bones_created[bone_name] = bone

    # Second pass: set parent relationships (only if both bones exist)
    for local_idx, mhr70_idx in enumerate(joint_indices):
        bone_name = MHR70_BONE_NAMES[mhr70_idx]
        parent_mhr70_idx = MHR70_BONE_PARENTS[mhr70_idx]

        if parent_mhr70_idx != -1:
            parent_name = MHR70_BONE_NAMES[parent_mhr70_idx]
            # Only connect if parent is also in our subset
            if parent_name in bones_created and bone_name in armature.data.edit_bones:
                armature.data.edit_bones[bone_name].parent = armature.data.edit_bones[parent_name]

                # Connect tail of parent to head of child for cleaner hierarchy
                parent_bone = armature.data.edit_bones[parent_name]
                child_bone = armature.data.edit_bones[bone_name]
                parent_bone.tail = child_bone.head

    bpy.ops.object.mode_set(mode='OBJECT')

    print(f"[Blender] Created {len(bones_created)} bones")
    return armature, joint_indices


def apply_animation(armature, skeleton_data, fps, joint_indices):
    """
    Apply animation keyframes to armature.

    The joint positions from SAM-3D-Body are in world space.
    We compute the delta from rest pose to animate the bones.

    Args:
        armature: Blender armature object
        skeleton_data: Dict with 'frames' key
        fps: Frames per second
        joint_indices: List of MHR70 indices that are included
    """
    print("[Blender] Applying animation...")

    # Set scene settings
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = len(skeleton_data['frames']) - 1
    bpy.context.scene.render.fps = int(fps)

    # Get rest pose positions (from first frame - used to create armature)
    first_frame = skeleton_data['frames'][0]
    rest_positions = first_frame['joint_positions']

    # Convert rest positions to Blender coordinate system
    # After flip in Python: (-X, -Y, -Z)
    # Blender transform: X stays, Y <- -Z, Z <- Y
    rest_positions_blender = []
    for pos in rest_positions:
        rest_positions_blender.append(Vector((pos[0], -pos[2], pos[1])))

    # Set armature as active
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')

    # Create keyframes for each frame
    for frame_idx, frame_data in enumerate(skeleton_data['frames']):
        bpy.context.scene.frame_set(frame_idx)

        joint_positions = frame_data['joint_positions']
        joint_rotations = frame_data['joint_rotations']

        # Iterate over the joints we have (using local index)
        for local_idx, mhr70_idx in enumerate(joint_indices):
            bone_name = MHR70_BONE_NAMES[mhr70_idx]

            if bone_name not in armature.pose.bones:
                continue

            pose_bone = armature.pose.bones[bone_name]

            # Get world position for this frame (using local index)
            src_pos = joint_positions[local_idx]

            # Convert from flipped coords to Blender
            # After flip in Python: (-X, -Y, -Z)
            # Blender transform: X stays, Y <- -Z, Z <- Y
            blender_pos = Vector((src_pos[0], -src_pos[2], src_pos[1]))

            # Compute delta from rest pose
            rest_pos = rest_positions_blender[local_idx]
            delta = blender_pos - rest_pos

            # Apply delta as location offset (relative to rest pose)
            pose_bone.location = delta
            pose_bone.keyframe_insert(data_path="location", frame=frame_idx)

            # Apply rotation (quaternion)
            # Convert quaternion to match coordinate swap
            # Coordinate transform: X stays, Y <- -Z, Z <- Y
            # Quaternion: swap Y/Z components to match
            quat_src = joint_rotations[local_idx]  # (w, x, y, z)
            quat = Quaternion((quat_src[0], quat_src[1], quat_src[3], quat_src[2]))
            pose_bone.rotation_mode = 'QUATERNION'
            pose_bone.rotation_quaternion = quat
            pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[Blender] Applied {len(skeleton_data['frames'])} keyframes")


def import_mesh(mesh_path, armature):
    """
    Import OBJ mesh and parent to armature.

    Args:
        mesh_path: Path to OBJ file
        armature: Armature object to parent to

    Returns:
        mesh_obj: Imported mesh object
    """
    if mesh_path == "none":
        return None

    print(f"[Blender] Importing mesh from {mesh_path}...")

    # Import OBJ
    bpy.ops.wm.obj_import(filepath=mesh_path)
    mesh_obj = bpy.context.selected_objects[0]

    # Add armature modifier
    modifier = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
    modifier.object = armature

    # Parent mesh to armature
    mesh_obj.parent = armature

    print("[Blender] Mesh imported and parented to armature")
    return mesh_obj


def export_fbx(output_path):
    """Export scene as FBX."""
    print(f"[Blender] Exporting to {output_path}...")

    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=False,
        bake_anim=True,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        add_leaf_bones=False,
        apply_scale_options='FBX_SCALE_ALL',
        path_mode='COPY',
        embed_textures=False,
    )

    print("[Blender] Export complete!")


def main():
    """Main entry point."""
    # Parse arguments
    argv = sys.argv
    if "--" not in argv:
        print("Error: No arguments provided")
        print("Usage: blender --background --python script.py -- skeleton.json output.fbx fps [mesh.obj]")
        sys.exit(1)

    argv = argv[argv.index("--") + 1:]

    if len(argv) < 3:
        print("Error: Insufficient arguments")
        print("Required: skeleton.json output.fbx fps")
        sys.exit(1)

    skeleton_json = argv[0]
    output_fbx = argv[1]
    fps = float(argv[2])
    mesh_obj = argv[3] if len(argv) > 3 else "none"

    print(f"[Blender] Starting export...")
    print(f"  - Skeleton: {skeleton_json}")
    print(f"  - Output: {output_fbx}")
    print(f"  - FPS: {fps}")
    print(f"  - Mesh: {mesh_obj}")

    # Load skeleton data
    with open(skeleton_json) as f:
        skeleton_data = json.load(f)

    # Clear scene
    clear_scene()

    # Create armature and apply animation
    armature, joint_indices = create_armature(skeleton_data)
    apply_animation(armature, skeleton_data, fps, joint_indices)

    # Import mesh if provided
    if mesh_obj != "none":
        import_mesh(mesh_obj, armature)

    # Export FBX
    export_fbx(output_fbx)

    print("[Blender] Done!")


if __name__ == "__main__":
    main()
