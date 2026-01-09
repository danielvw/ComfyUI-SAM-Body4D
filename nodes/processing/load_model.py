"""
LoadBody4DModel Node

Loads SAM-Body4D pipeline components with model caching.
"""

import os
import sys
from pathlib import Path

# CRITICAL: Add sam-body4d to path BEFORE any other imports
# This must happen before torch, omegaconf, or any sam-body4d modules are imported
# sam-body4d is a sibling directory in custom_nodes/
# Path: nodes/processing/load_model.py -> nodes/ -> ComfyUI-SAM-Body4D/ -> custom_nodes/ -> sam-body4d/
SAM_BODY4D_PATH = Path(__file__).parent.parent.parent.parent / "sam-body4d"
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

# Now import other modules
import torch
from omegaconf import OmegaConf


class LoadBody4DModel:
    """
    Load SAM-Body4D pipeline with all components.

    This node initializes:
    - SAM-3 for video tracking
    - SAM-3D-Body for 3D mesh estimation
    - Optional: Diffusion-VAS for occlusion handling
    """

    # Class-level model cache
    _model_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        # Default to the configs/ directory in the ComfyUI-SAM-Body4D repository
        default_config = str(Path(__file__).parent.parent / "configs" / "body4d.yaml")
        return {
            "required": {
                "config_path": ("STRING", {
                    "default": default_config,
                    "tooltip": "Path to SAM-Body4D configuration YAML"
                }),
                "enable_occlusion": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable Diffusion-VAS for occlusion handling (requires 40GB+ VRAM)"
                }),
            },
            "optional": {
                "batch_size": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 128,
                    "tooltip": "Batch size for SAM-3D-Body inference"
                }),
            }
        }

    RETURN_TYPES = ("BODY4D_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "SAM-Body4D"

    def load_model(self, config_path, enable_occlusion, batch_size=64):
        """
        Load SAM-Body4D model bundle with caching.

        Args:
            config_path: Path to body4d.yaml configuration
            enable_occlusion: Enable occlusion completion pipeline
            batch_size: Batch size for SAM-3D-Body

        Returns:
            Tuple containing model bundle dict
        """
        # CRITICAL: Re-apply path fixes at runtime in case other nodes modified sys.path
        # This must happen EVERY time, not just at module import
        if SAM_BODY4D_PATH.exists():
            paths_to_add = [
                str(SAM_BODY4D_PATH),
                str(SAM_BODY4D_PATH / "models" / "sam_3d_body"),
                str(SAM_BODY4D_PATH / "models" / "diffusion_vas"),
            ]

            # Remove conflicting paths
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

            # CRITICAL: Remove cached modules that might be from wrong paths
            # Python caches modules, so we need to clear conflicting ones
            modules_to_remove = [
                key for key in sys.modules.keys()
                if key == 'utils' or key.startswith('utils.')
                or key == 'sam_3d_body' or key.startswith('sam_3d_body.')
            ]
            for mod in modules_to_remove:
                del sys.modules[mod]

        # Cache key
        cache_key = f"{config_path}_{enable_occlusion}_{batch_size}"

        # Check cache
        if cache_key in self._model_cache:
            print(f"[Body4D] Using cached model: {cache_key}")
            model_bundle = self._model_cache[cache_key]

            # Check if MHR is already wrapped for CPU execution
            estimator = model_bundle.get('estimator')
            if estimator and hasattr(estimator, 'model'):
                if hasattr(estimator.model, 'head_pose'):
                    head_pose = estimator.model.head_pose
                    if hasattr(head_pose, '_mhr_cpu_wrapped') and head_pose._mhr_cpu_wrapped:
                        print("[Body4D] Cached MHR already wrapped for CPU")
                    else:
                        print("[Body4D] Warning: Cached MHR not wrapped - may cause BFloat16 errors")

            return (model_bundle,)

        print(f"[Body4D] Loading models from: {config_path}")

        # Load configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please ensure SAM-Body4D is installed at: {SAM_BODY4D_PATH}"
            )

        config = OmegaConf.load(config_path)

        # Override batch size
        config.sam_3d_body.batch_size = batch_size

        # Override occlusion setting
        config.completion.enable = enable_occlusion

        # Initialize models
        try:
            from models.sam3.sam3.model_builder import build_sam3_video_model
            from models.sam_3d_body.sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
            from models.sam_3d_body.tools.build_fov_estimator import FOVEstimator

            # 1. SAM-3 for video tracking
            print("[Body4D] Loading SAM-3...")
            sam3_model = build_sam3_video_model(checkpoint_path=config.sam3['ckpt_path'])
            predictor = sam3_model.tracker
            predictor.backbone = sam3_model.detector.backbone

            # 2. SAM-3D-Body for mesh estimation
            print("[Body4D] Loading SAM-3D-Body...")
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            model, model_cfg = load_sam_3d_body(
                config.sam_3d_body['ckpt_path'],
                device=device,
                mhr_path=config.sam_3d_body['mhr_path']
            )

            # FOV estimator
            fov_estimator = FOVEstimator(
                name='moge2',
                device=device,
                path=config.sam_3d_body['fov_path']
            )

            estimator = SAM3DBodyEstimator(
                sam_3d_body_model=model,
                model_cfg=model_cfg,
                human_detector=None,  # Will use SAM-3 masks
                human_segmentor=None,
                fov_estimator=fov_estimator,
            )

            # CRITICAL: Wrap self.mhr to run on CPU
            # PyTorch CUDA doesn't support "addmm_sparse_cuda" with BFloat16
            # TorchScript constants are baked in and can't be converted
            # Solution: Run MHR model on CPU (where BFloat16 sparse ops work)
            print("[Body4D] Wrapping MHR to run on CPU (BFloat16 sparse workaround)...")
            if hasattr(estimator.model, 'head_pose') and hasattr(estimator.model.head_pose, 'mhr'):
                head_pose = estimator.model.head_pose
                try:
                    # Move MHR model to CPU permanently
                    original_mhr = head_pose.mhr
                    original_mhr.cpu()
                    original_mhr.float()

                    class MHRCPUWrapper:
                        """Wrapper that runs MHR on CPU and moves results back to GPU."""
                        def __init__(self, mhr_model):
                            self.mhr_model = mhr_model

                        def __call__(self, shape_params, model_params, expr_params):
                            # Remember original device
                            original_device = shape_params.device

                            # Move inputs to CPU and convert to float32
                            shape_cpu = shape_params.detach().cpu().float()
                            model_cpu = model_params.detach().cpu().float()
                            expr_cpu = expr_params.detach().cpu().float() if expr_params is not None else None

                            # Run on CPU
                            verts, skel_state = self.mhr_model(shape_cpu, model_cpu, expr_cpu)

                            # Move results back to original device
                            return verts.to(original_device), skel_state.to(original_device)

                        def __getattr__(self, name):
                            # Forward all other attribute access to the original model
                            return getattr(self.mhr_model, name)

                    head_pose.mhr = MHRCPUWrapper(original_mhr)
                    head_pose._mhr_cpu_wrapped = True
                    print("[Body4D] MHR wrapped to run on CPU")
                except Exception as e:
                    print(f"[Body4D] Warning: Could not wrap MHR: {e}")
                    import traceback
                    traceback.print_exc()

            # 3. Optional: Diffusion-VAS for occlusion
            pipeline_mask = None
            pipeline_rgb = None
            depth_model = None
            max_occ_len = None
            generator = None

            if enable_occlusion:
                print("[Body4D] Loading Diffusion-VAS for occlusion handling...")
                from models.diffusion_vas.demo import (
                    init_amodal_segmentation_model,
                    init_rgb_model,
                    init_depth_model
                )

                pipeline_mask = init_amodal_segmentation_model(config.completion['model_path_mask'])
                pipeline_rgb = init_rgb_model(config.completion['model_path_rgb'])
                depth_model = init_depth_model(
                    config.completion['model_path_depth'],
                    config.completion['depth_encoder']
                )
                max_occ_len = min(config.completion['max_occ_len'], batch_size)
                generator = torch.manual_seed(23)

            # Bundle all models
            model_bundle = {
                'sam3_model': sam3_model,
                'predictor': predictor,
                'estimator': estimator,
                'pipeline_mask': pipeline_mask,
                'pipeline_rgb': pipeline_rgb,
                'depth_model': depth_model,
                'max_occ_len': max_occ_len,
                'generator': generator,
                'config': config,
                'device': device,
                'batch_size': batch_size,
            }

            # Cache for future use
            self._model_cache[cache_key] = model_bundle

            print(f"[Body4D] Models loaded successfully")
            print(f"[Body4D] Device: {device}")
            print(f"[Body4D] Batch size: {batch_size}")
            print(f"[Body4D] Occlusion handling: {'enabled' if enable_occlusion else 'disabled'}")

            return (model_bundle,)

        except ImportError as e:
            raise ImportError(
                f"Failed to import SAM-Body4D modules. Please ensure SAM-Body4D is installed.\n"
                f"Expected path: {SAM_BODY4D_PATH}\n"
                f"Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {e}")


# Register the node
NODE_CLASS_MAPPINGS = {
    "LoadBody4DModel": LoadBody4DModel
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadBody4DModel": "Load Body4D Model"
}
