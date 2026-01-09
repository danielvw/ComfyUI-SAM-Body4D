"""
Body4DSkeletonExtract Node

Extracts skeleton animation data from Body4D sequence.
"""

import numpy as np
import torch
from scipy.signal import butter, filtfilt

# Import constants
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from constants import JOINT_SUBSETS


class Body4DSkeletonExtract:
    """
    Extract skeleton animation from Body4D sequence.

    Converts per-frame 3D joint positions and rotations into
    a structured animation data format suitable for FBX export.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sequence_data": ("BODY4D_SEQUENCE",),
            },
            "optional": {
                "person_id": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Person ID for multi-person sequences (0-indexed)"
                }),
                "joint_subset": (["full_70", "body_17", "body_hands"], {
                    "default": "full_70",
                    "tooltip": "Subset of joints to extract"
                }),
                "smooth_factor": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "tooltip": "Temporal smoothing (0=off, 1=max smoothing)"
                }),
            }
        }

    RETURN_TYPES = ("BODY4D_ANIMATION",)
    RETURN_NAMES = ("animation",)
    FUNCTION = "extract"
    CATEGORY = "SAM-Body4D"

    def extract(self, sequence_data, person_id=0, joint_subset="full_70", smooth_factor=0.0):
        """
        Extract skeleton animation from sequence data.

        Args:
            sequence_data: Output from Body4DProcess
            person_id: Person ID to extract (0-indexed)
            joint_subset: Which joints to include
            smooth_factor: Temporal smoothing amount

        Returns:
            Tuple containing animation dict
        """
        print(f"[Body4D] Extracting skeleton animation...")
        print(f"  - Person ID: {person_id}")
        print(f"  - Joint subset: {joint_subset}")
        print(f"  - Smooth factor: {smooth_factor}")

        # Get person outputs
        if person_id not in sequence_data['person_outputs']:
            available_ids = list(sequence_data['person_outputs'].keys())
            raise ValueError(
                f"Person ID {person_id} not found in sequence. "
                f"Available IDs: {available_ids}"
            )

        person_outputs = sequence_data['person_outputs'][person_id]
        n_frames = len(person_outputs)

        # Extract joint data for all frames
        joint_positions = []  # [n_frames, n_joints, 3]
        joint_rotations = []  # [n_frames, n_joints, 4] (quaternion)
        vertices_list = []    # [n_frames, n_vertices, 3]

        for frame_output in person_outputs:
            # Joint coordinates (3D positions)
            joints_3d = frame_output['pred_joint_coords']  # [70, 3]
            joint_positions.append(joints_3d)

            # Joint rotations (global rotations as quaternions)
            rotations = frame_output['pred_global_rots']  # [70, 4] or [70, 3, 3]
            # Convert rotation matrices to quaternions if needed
            if len(rotations.shape) == 3:
                # Assume rotation matrices, convert to quaternions
                rotations = self._rotation_matrix_to_quaternion(rotations)
            joint_rotations.append(rotations)

            # Mesh vertices
            vertices = frame_output['pred_vertices']  # [n_vertices, 3]
            vertices_list.append(vertices)

        # Stack into arrays
        joint_positions = np.stack(joint_positions, axis=0)  # [n_frames, 70, 3]
        joint_rotations = np.stack(joint_rotations, axis=0)  # [n_frames, 70, 4]
        vertices_array = np.stack(vertices_list, axis=0)     # [n_frames, n_vertices, 3]

        # Apply joint subset filtering
        joint_indices = JOINT_SUBSETS[joint_subset]
        joint_positions = joint_positions[:, joint_indices, :]
        joint_rotations = joint_rotations[:, joint_indices, :]

        # Apply temporal smoothing if requested
        if smooth_factor > 0.0:
            print(f"[Body4D] Applying temporal smoothing...")
            joint_positions = self._smooth_temporal(joint_positions, smooth_factor)
            joint_rotations = self._smooth_temporal(joint_rotations, smooth_factor)

        # Get first frame data for reference
        first_frame = person_outputs[0]
        faces = first_frame.get('faces', None)

        # Build animation data structure
        animation_data = {
            'joint_positions': joint_positions,  # [n_frames, n_joints, 3]
            'joint_rotations': joint_rotations,  # [n_frames, n_joints, 4]
            'vertices': vertices_array,          # [n_frames, n_vertices, 3]
            'faces': faces,                      # [n_faces, 3] or None
            'joint_subset': joint_subset,
            'joint_indices': joint_indices,
            'fps': sequence_data['fps'],
            'frame_count': n_frames,
            'person_id': person_id,
        }

        print(f"[Body4D] Animation extracted:")
        print(f"  - Frames: {n_frames}")
        print(f"  - Joints: {len(joint_indices)}")
        print(f"  - Vertices: {vertices_array.shape[1]}")

        return (animation_data,)

    def _rotation_matrix_to_quaternion(self, rot_matrices):
        """
        Convert rotation matrices to quaternions.

        Args:
            rot_matrices: [N, 3, 3] rotation matrices

        Returns:
            quaternions: [N, 4] quaternions (w, x, y, z)
        """
        # Simple conversion (can be optimized)
        n = rot_matrices.shape[0]
        quaternions = np.zeros((n, 4))

        for i in range(n):
            R = rot_matrices[i]
            trace = np.trace(R)

            if trace > 0:
                s = 0.5 / np.sqrt(trace + 1.0)
                w = 0.25 / s
                x = (R[2, 1] - R[1, 2]) * s
                y = (R[0, 2] - R[2, 0]) * s
                z = (R[1, 0] - R[0, 1]) * s
            else:
                if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                    s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                    w = (R[2, 1] - R[1, 2]) / s
                    x = 0.25 * s
                    y = (R[0, 1] + R[1, 0]) / s
                    z = (R[0, 2] + R[2, 0]) / s
                elif R[1, 1] > R[2, 2]:
                    s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                    w = (R[0, 2] - R[2, 0]) / s
                    x = (R[0, 1] + R[1, 0]) / s
                    y = 0.25 * s
                    z = (R[1, 2] + R[2, 1]) / s
                else:
                    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                    w = (R[1, 0] - R[0, 1]) / s
                    x = (R[0, 2] + R[2, 0]) / s
                    y = (R[1, 2] + R[2, 1]) / s
                    z = 0.25 * s

            quaternions[i] = [w, x, y, z]

        return quaternions

    def _smooth_temporal(self, data, smooth_factor):
        """
        Apply Butterworth low-pass filter for temporal smoothing.

        Args:
            data: [n_frames, ...] array to smooth
            smooth_factor: Smoothing strength (0-1)

        Returns:
            smoothed: Smoothed array
        """
        if smooth_factor <= 0:
            return data

        # Determine cutoff frequency based on smooth_factor
        # Higher smooth_factor = lower cutoff = more smoothing
        fps = 30.0  # Assume 30 FPS for filter design
        nyquist = fps / 2.0
        cutoff = nyquist * (1.0 - smooth_factor * 0.8)  # Keep some high freq

        # Design Butterworth filter
        order = 2
        b, a = butter(order, cutoff / nyquist, btype='low')

        # Reshape data for filtering
        original_shape = data.shape
        n_frames = original_shape[0]

        # Flatten all dimensions except time
        data_flat = data.reshape(n_frames, -1)

        # Filter each dimension
        filtered = np.zeros_like(data_flat)
        for i in range(data_flat.shape[1]):
            filtered[:, i] = filtfilt(b, a, data_flat[:, i])

        # Reshape back
        smoothed = filtered.reshape(original_shape)

        return smoothed


# Register the node
NODE_CLASS_MAPPINGS = {
    "Body4DSkeletonExtract": Body4DSkeletonExtract
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Body4DSkeletonExtract": "Body4D Skeleton Extract"
}
