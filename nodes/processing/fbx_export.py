"""
Body4DExportFBX Node

Exports animated FBX file using Blender subprocess.
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
from pathlib import Path

# Import constants
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from constants import BLENDER_TIMEOUT

# Import skeleton utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lib"))
from skeleton_utils import MHR70_BONE_NAMES


class Body4DExportFBX:
    """
    Export animated FBX with skeleton animation using Blender.

    This node:
    1. Prepares skeleton animation JSON
    2. Optionally saves reference mesh as OBJ
    3. Calls Blender subprocess to create animated FBX
    4. Returns path to exported FBX
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "animation": ("BODY4D_ANIMATION",),
                "output_filename": ("STRING", {
                    "default": "body4d_animation",
                    "tooltip": "Output filename (without .fbx extension)"
                }),
            },
            "optional": {
                "export_mesh": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Export mesh with skinning (experimental)"
                }),
                "reference_frame": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Frame to use for bind-pose mesh"
                }),
                "blender_path": ("STRING", {
                    "default": "blender",
                    "tooltip": "Path to Blender executable"
                }),
                "output_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Custom output directory (empty = ComfyUI output folder)"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fbx_path",)
    OUTPUT_NODE = True
    FUNCTION = "export"
    CATEGORY = "SAM-Body4D/Export"

    def export(self, animation, output_filename, export_mesh=False, reference_frame=0,
               blender_path="blender", output_dir=""):
        """
        Export animated FBX via Blender subprocess.

        Args:
            animation: Animation data from Body4DSkeletonExtract
            output_filename: Output filename (without extension)
            export_mesh: Whether to export mesh with skinning
            reference_frame: Frame to use for mesh
            blender_path: Path to Blender executable
            output_dir: Custom output directory

        Returns:
            Tuple containing path to exported FBX
        """
        print(f"[Body4D] Exporting animated FBX...")
        print(f"  - Frames: {animation['frame_count']}")
        print(f"  - FPS: {animation['fps']}")
        print(f"  - Joints: {len(animation['joint_indices'])}")

        # Determine output directory
        if output_dir:
            out_dir = Path(output_dir)
        else:
            try:
                import folder_paths
                out_dir = Path(folder_paths.get_output_directory())
            except:
                out_dir = Path("/tmp")

        out_dir.mkdir(parents=True, exist_ok=True)

        # Prepare output path
        fbx_path = out_dir / f"{output_filename}.fbx"

        # Create temporary directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            # Prepare skeleton animation JSON
            skeleton_json = temp_dir / "skeleton_animation.json"
            self._save_skeleton_json(animation, skeleton_json)

            # Prepare mesh OBJ if requested
            mesh_obj_path = "none"
            if export_mesh and animation['faces'] is not None:
                mesh_obj_path = temp_dir / "reference_mesh.obj"
                self._save_mesh_obj(animation, reference_frame, mesh_obj_path)

            # Get path to Blender export script
            blender_script = Path(__file__).parent.parent.parent / "lib" / "blender_export_animated_fbx.py"

            if not blender_script.exists():
                raise FileNotFoundError(f"Blender script not found: {blender_script}")

            # Resolve Blender executable path
            blender_exec = self._resolve_blender_path(blender_path)
            print(f"[Body4D] Using Blender: {blender_exec}")

            # Check if we have execute permission
            if not os.access(blender_exec, os.X_OK):
                raise PermissionError(
                    f"No execute permission for Blender: {blender_exec}\n"
                    f"Try: sudo chmod +x {blender_exec}\n"
                    f"Or copy Blender to a user-accessible location (e.g., ~/blender/)"
                )

            # Call Blender subprocess
            print(f"[Body4D] Calling Blender subprocess...")
            try:
                result = subprocess.run(
                    [
                        blender_exec,
                        "--background",
                        "--python", str(blender_script),
                        "--",
                        str(skeleton_json),
                        str(fbx_path),
                        str(animation['fps']),
                        str(mesh_obj_path),
                    ],
                    timeout=BLENDER_TIMEOUT,
                    capture_output=True,
                    text=True,
                    check=True,
                )

                print("[Body4D] Blender output:")
                print(result.stdout)

                if result.stderr:
                    print("[Body4D] Blender warnings:")
                    print(result.stderr)

            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"Blender export timed out after {BLENDER_TIMEOUT} seconds. "
                    f"Try reducing the number of frames or increasing timeout in constants.py"
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Blender export failed with error code {e.returncode}:\n"
                    f"stdout: {e.stdout}\n"
                    f"stderr: {e.stderr}"
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Blender executable not found: {blender_exec}\n"
                    f"Please install Blender or specify correct path."
                )

        # Verify FBX was created
        if not fbx_path.exists():
            raise RuntimeError(f"FBX file was not created: {fbx_path}")

        print(f"[Body4D] FBX exported successfully:")
        print(f"  - Path: {fbx_path}")
        print(f"  - Size: {fbx_path.stat().st_size / 1024:.1f} KB")

        return (str(fbx_path),)

    def _resolve_blender_path(self, blender_path):
        """
        Resolve Blender executable path.

        If blender_path is a directory, try to find the blender executable inside.
        Common patterns: blender, blender.exe, Blender.app/Contents/MacOS/Blender
        """
        blender_path = Path(blender_path)

        # If it's already an executable file, return it
        if blender_path.is_file() and os.access(blender_path, os.X_OK):
            return str(blender_path)

        # If it's a directory, search for blender executable
        if blender_path.is_dir():
            # Try common locations
            candidates = [
                blender_path / "blender",               # Linux
                blender_path / "blender.exe",           # Windows
                blender_path / "Blender.app" / "Contents" / "MacOS" / "Blender",  # macOS
                blender_path / "Contents" / "MacOS" / "Blender",  # macOS (if path is to .app)
            ]

            for candidate in candidates:
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return str(candidate)

            raise FileNotFoundError(
                f"Could not find Blender executable in directory: {blender_path}\n"
                f"Tried: {[str(c) for c in candidates]}"
            )

        # If it's a command name (like "blender"), let subprocess find it
        if not str(blender_path).startswith('/') and not str(blender_path).startswith('.'):
            # Try to find it in PATH using shutil.which
            found = shutil.which(str(blender_path))
            if found:
                return found

        # Otherwise return as-is and let subprocess fail with helpful error
        return str(blender_path)

    def _save_skeleton_json(self, animation, output_path):
        """Save skeleton animation data as JSON for Blender."""
        # Prepare frames data
        frames = []
        n_frames = animation['frame_count']

        for frame_idx in range(n_frames):
            # Get positions and rotations for this frame
            positions = animation['joint_positions'][frame_idx]  # [n_joints, 3]
            rotations = animation['joint_rotations'][frame_idx]  # [n_joints, 4]

            frame_data = {
                'joint_positions': positions.tolist(),
                'joint_rotations': rotations.tolist(),
            }
            frames.append(frame_data)

        # Build complete skeleton data
        skeleton_data = {
            'frames': frames,
            'fps': animation['fps'],
            'joint_count': len(animation['joint_indices']),
            'joint_indices': animation['joint_indices'],  # Which MHR70 joints to use
            'joint_subset': animation['joint_subset'],    # Name of subset (full_70, body_17, etc.)
        }

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(skeleton_data, f, indent=2)

        print(f"[Body4D] Saved skeleton JSON: {output_path}")

    def _save_mesh_obj(self, animation, frame_idx, output_path):
        """Save reference mesh as OBJ file."""
        # Get vertices and faces for reference frame
        if frame_idx >= animation['frame_count']:
            frame_idx = 0

        vertices = animation['vertices'][frame_idx]  # [n_vertices, 3]
        faces = animation['faces']  # [n_faces, 3]

        # Write OBJ file
        with open(output_path, 'w') as f:
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write faces (1-indexed)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        print(f"[Body4D] Saved mesh OBJ: {output_path}")


# Register the node
NODE_CLASS_MAPPINGS = {
    "Body4DExportFBX": Body4DExportFBX
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Body4DExportFBX": "Body4D Export FBX"
}
