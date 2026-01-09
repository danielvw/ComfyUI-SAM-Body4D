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

            # CRITICAL: Apply BFloat16 input conversion patch for cached models
            estimator = model_bundle.get('estimator')
            if estimator and hasattr(estimator, 'model'):
                if hasattr(estimator.model, 'head_pose') and hasattr(estimator.model.head_pose, 'mhr'):
                    try:
                        # Check if already patched (has __wrapped__ attribute)
                        if not hasattr(estimator.model.head_pose.mhr.forward, '__wrapped__'):
                            print("[Body4D] Patching cached MHR model for BFloat16->Float32...")
                            original_forward = estimator.model.head_pose.mhr.forward

                            def mhr_forward_wrapper(*args, **kwargs):
                                args_f32 = []
                                for arg in args:
                                    if isinstance(arg, torch.Tensor) and arg.dtype == torch.bfloat16:
                                        args_f32.append(arg.to(torch.float32))
                                    else:
                                        args_f32.append(arg)
                                kwargs_f32 = {}
                                for k, v in kwargs.items():
                                    if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                                        kwargs_f32[k] = v.to(torch.float32)
                                    else:
                                        kwargs_f32[k] = v
                                return original_forward(*args_f32, **kwargs_f32)

                            mhr_forward_wrapper.__wrapped__ = True  # Mark as patched
                            estimator.model.head_pose.mhr.forward = mhr_forward_wrapper
                            print("[Body4D] Cached MHR model patched successfully")
                    except Exception as e:
                        print(f"[Body4D] Warning: Could not patch cached MHR: {e}")

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

            # CRITICAL: Patch MHR forward to convert inputs to Float32
            # PyTorch CUDA doesn't support "addmm_sparse_cuda" with BFloat16
            # TorchScript models have sub-modules that stay in BFloat16, so we wrap the forward method
            print("[Body4D] Patching MHR model to handle BFloat16->Float32 conversion...")
            if hasattr(estimator.model, 'head_pose') and hasattr(estimator.model.head_pose, 'mhr'):
                try:
                    original_forward = estimator.model.head_pose.mhr.forward

                    def mhr_forward_wrapper(*args, **kwargs):
                        # Convert all tensor args/kwargs to float32
                        args_f32 = []
                        for arg in args:
                            if isinstance(arg, torch.Tensor) and arg.dtype == torch.bfloat16:
                                args_f32.append(arg.to(torch.float32))
                            else:
                                args_f32.append(arg)

                        kwargs_f32 = {}
                        for k, v in kwargs.items():
                            if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                                kwargs_f32[k] = v.to(torch.float32)
                            else:
                                kwargs_f32[k] = v

                        # Call original forward with float32 inputs
                        result = original_forward(*args_f32, **kwargs_f32)
                        return result

                    mhr_forward_wrapper.__wrapped__ = True  # Mark as patched
                    estimator.model.head_pose.mhr.forward = mhr_forward_wrapper
                    print("[Body4D] MHR model patched for BFloat16->Float32 input conversion")
                except Exception as e:
                    print(f"[Body4D] Warning: Could not patch MHR forward: {e}")

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
