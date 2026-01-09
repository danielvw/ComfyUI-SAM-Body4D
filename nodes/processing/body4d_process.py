"""
Body4DProcess Node

Executes the full SAM-Body4D pipeline: video tracking + 3D mesh estimation.
"""

import os
import sys
import time
import tempfile
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

# Import base utilities
from ..base import comfy_batch_to_list, list_to_comfy_batch


class Body4DProcess:
    """
    Execute full Body4D pipeline: Tracking + 3D Mesh generation.

    This node:
    1. Converts ComfyUI images to temporary frames
    2. Runs SAM-3 video tracking for temporal consistency
    3. Runs SAM-3D-Body for 3D mesh estimation
    4. Returns processed sequence data
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BODY4D_MODEL",),
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 30.0}),
            },
            "optional": {
                "detection_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "tooltip": "Minimum confidence for human detection"
                }),
                "max_persons": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Maximum number of persons to track"
                }),
                "inference_type": (["full", "body", "hand"], {
                    "default": "full",
                    "tooltip": "full=body+hands, body=body only, hand=hands only"
                }),
            }
        }

    RETURN_TYPES = ("BODY4D_SEQUENCE", "IMAGE")
    RETURN_NAMES = ("sequence_data", "preview")
    FUNCTION = "process"
    CATEGORY = "SAM-Body4D"

    def process(self, model, images, fps, detection_threshold=0.5, max_persons=5, inference_type="full"):
        """
        Execute Body4D pipeline on image sequence.

        Args:
            model: Model bundle from LoadBody4DModel
            images: ComfyUI image tensor [B, H, W, C]
            fps: Frames per second
            detection_threshold: Detection confidence threshold
            max_persons: Maximum persons to track
            inference_type: Type of inference (full/body/hand)

        Returns:
            Tuple of (sequence_data dict, preview images)
        """
        print(f"[Body4D] Starting processing...")
        print(f"[Body4D] Frames: {images.shape[0]}, FPS: {fps}")

        # Extract model components
        predictor = model['predictor']
        estimator = model['estimator']
        config = model['config']
        batch_size = model['batch_size']

        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            image_dir = temp_dir / "images"
            mask_dir = temp_dir / "masks"
            image_dir.mkdir()
            mask_dir.mkdir()

            # Save frames to temporary directory
            print(f"[Body4D] Saving frames to temporary directory...")
            frame_paths = self._save_frames(images, image_dir)

            # Phase 1: Mask Generation (SAM-3 tracking)
            print(f"[Body4D] Phase 1: SAM-3 video tracking...")
            out_obj_ids, masks_dict = self._run_sam3_tracking(
                predictor, temp_dir, frame_paths, mask_dir, max_persons
            )

            if len(out_obj_ids) == 0:
                raise RuntimeError("No persons detected in the video sequence")

            print(f"[Body4D] Detected {len(out_obj_ids)} person(s)")

            # Phase 2: 4D Generation (SAM-3D-Body)
            print(f"[Body4D] Phase 2: 4D mesh generation...")
            person_outputs = self._run_4d_generation(
                estimator, frame_paths, mask_dir, out_obj_ids,
                batch_size, inference_type
            )

            # Create preview visualization
            print(f"[Body4D] Creating preview...")
            preview_images = self._create_preview(frame_paths, person_outputs, estimator.faces)

        # Build sequence data structure
        sequence_data = {
            'fps': fps,
            'frame_count': len(frame_paths),
            'person_ids': out_obj_ids,
            'person_outputs': person_outputs,  # {person_id: [frame_outputs]}
        }

        print(f"[Body4D] Processing complete!")

        return (sequence_data, preview_images)

    def _save_frames(self, images, output_dir):
        """Save ComfyUI images to temporary directory."""
        frame_paths = []
        images_np = comfy_batch_to_list(images)

        for idx, img_np in enumerate(images_np):
            # Convert BGR to RGB for saving
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            frame_path = output_dir / f"{idx:08d}.jpg"
            cv2.imwrite(str(frame_path), img_rgb)
            frame_paths.append(str(frame_path))

        return frame_paths

    def _run_sam3_tracking(self, predictor, temp_dir, frame_paths, mask_dir, max_persons):
        """Run SAM-3 video tracking for temporal consistency."""
        # Initialize video state
        inference_state = predictor.init_state(video_path=str(temp_dir / "images"))

        # Auto-detect humans in first frame using SAM-3's backbone
        # We'll use a simple approach: add points in a grid pattern to detect all objects
        first_frame = cv2.imread(frame_paths[0])
        h, w = first_frame.shape[:2]

        # Add grid of points to detect objects automatically
        # This is a simple auto-detection approach
        grid_points = []
        grid_labels = []

        # Create a 3x3 grid of points
        for y in np.linspace(h * 0.2, h * 0.8, 3):
            for x in np.linspace(w * 0.2, w * 0.8, 3):
                grid_points.append([x, y])
                grid_labels.append(1)  # 1 = foreground point

        # Add points to first frame
        points = np.array(grid_points, dtype=np.float32)
        labels = np.array(grid_labels, dtype=np.int32)

        # Use SAM-3's add_new_points_or_box to detect objects
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=None,  # Auto-assign object IDs
            points=points,
            labels=labels,
        )

        print(f"[Body4D] Auto-detected {len(out_obj_ids)} object(s) in first frame")

        # Limit to max_persons
        out_obj_ids = out_obj_ids[:max_persons]

        # Propagate masks through video
        video_segments = {}

        for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores, iou_scores in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=1800,
            reverse=False,
            propagate_preflight=True,
        ):
            video_segments[frame_idx] = {
                out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Save masks to disk
        out_h = inference_state['video_height']
        out_w = inference_state['video_width']
        masks_dict = {}

        for frame_idx in video_segments.keys():
            img = inference_state['images'][frame_idx].detach().float().cpu()
            img = (img + 1) / 2
            img = img.clamp(0, 1)
            img = F.interpolate(
                img.unsqueeze(0),
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            img = img.permute(1, 2, 0)
            img = (img.numpy() * 255).astype("uint8")

            # Create combined mask image
            mask_combined = np.zeros((out_h, out_w), dtype=np.uint8)
            for out_obj_id, out_mask in video_segments[frame_idx].items():
                mask_combined[out_mask[0] > 0] = out_obj_id

            # Save mask
            mask_pil = Image.fromarray(mask_combined).convert('P')
            # Use simple palette
            mask_pil.save(str(mask_dir / f"{frame_idx:08d}.png"))

            masks_dict[frame_idx] = mask_combined

        return out_obj_ids, masks_dict

    def _run_4d_generation(self, estimator, frame_paths, mask_dir, out_obj_ids, batch_size, inference_type):
        """Run SAM-3D-Body for 4D mesh estimation."""
        n_frames = len(frame_paths)
        person_outputs = {obj_id: [] for obj_id in out_obj_ids}

        # Process in batches
        for i in range(0, n_frames, batch_size):
            batch_frames = frame_paths[i:i + batch_size]
            batch_masks = [str(mask_dir / f"{j:08d}.png") for j in range(i, min(i + batch_size, n_frames))]

            # Prepare bboxes and masks for each frame
            bboxes_batch = []
            masks_batch = []
            id_batch = []

            for frame_idx, (frame_path, mask_path) in enumerate(zip(batch_frames, batch_masks)):
                # Load mask
                mask = np.array(Image.open(mask_path).convert('P'))
                img = cv2.imread(frame_path)
                H, W = img.shape[:2]

                bbox_list = []
                mask_list = []
                id_list = []

                for obj_id in out_obj_ids:
                    # Extract object mask
                    obj_mask = (mask == obj_id).astype(np.uint8) * 255

                    if obj_mask.sum() == 0:
                        continue

                    # Compute bbox from mask
                    coords = cv2.findNonZero(obj_mask)
                    if coords is not None:
                        x, y, w, h = cv2.boundingRect(coords)
                        bbox = np.array([[x, y, x + w, y + h]], dtype=np.float32)
                        bbox_list.append(bbox)
                        mask_list.append(obj_mask)
                        id_list.append(obj_id)

                if len(bbox_list) > 0:
                    bboxes_batch.append(np.concatenate(bbox_list, axis=0))
                    masks_batch.append(np.stack(mask_list, axis=0))
                    id_batch.append(id_list)

            if len(bboxes_batch) == 0:
                continue

            # Run SAM-3D-Body inference
            outputs_batch = estimator.process_frames(
                batch_frames,
                bboxes=bboxes_batch,
                masks=masks_batch,
                use_mask=True,
                inference_type=inference_type,
                id_batch=id_batch,
                idx_path={},
                idx_dict={},
                mhr_shape_scale_dict={},
                occ_dict={obj_id: [1] * len(batch_frames) for obj_id in out_obj_ids},
            )

            # Organize outputs by person
            for frame_outputs, ids in zip(outputs_batch, id_batch):
                for person_output, person_id in zip(frame_outputs, ids):
                    person_outputs[person_id].append(person_output)

        return person_outputs

    def _create_preview(self, frame_paths, person_outputs, faces):
        """Create preview visualization with overlaid meshes."""
        from pathlib import Path
        import sys

        # Import visualization utilities from SAM-Body4D
        sam_body4d_path = Path(__file__).parent.parent.parent.parent / "sam-body4d"
        sys.path.insert(0, str(sam_body4d_path / "models" / "sam_3d_body"))

        from tools.vis_utils import visualize_sample_together

        preview_images = []
        for frame_idx, frame_path in enumerate(frame_paths):
            img = cv2.imread(frame_path)

            # Collect outputs for this frame
            frame_outputs = []
            frame_ids = []
            for person_id, outputs in person_outputs.items():
                if frame_idx < len(outputs):
                    frame_outputs.append(outputs[frame_idx])
                    frame_ids.append(person_id)

            # Visualize
            if len(frame_outputs) > 0:
                vis_img = visualize_sample_together(img, frame_outputs, faces, frame_ids)
            else:
                vis_img = img

            preview_images.append(vis_img)

        # Convert to ComfyUI format
        preview_tensor = list_to_comfy_batch(preview_images)

        return preview_tensor


# Register the node
NODE_CLASS_MAPPINGS = {
    "Body4DProcess": Body4DProcess
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Body4DProcess": "Body4D Process"
}
