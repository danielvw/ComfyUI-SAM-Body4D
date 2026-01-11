# ComfyUI-SAM-Body4D

[![en](https://img.shields.io/badge/lang-en-blue.svg)](README.md)
[![de](https://img.shields.io/badge/lang-de-yellow.svg)](README_de.md)
![Status](https://img.shields.io/badge/status-WIP-orange)

> [!WARNING]
> This project is currently under development and **not functional**.
> 
Custom Node Suite for ComfyUI to convert image sequences into animated FBX files with skeleton rigging.

## Overview

This node suite integrates SAM-Body4D into ComfyUI and enables:
- **Video Tracking** with SAM-3 for temporal consistency
- **3D Mesh Estimation** with SAM-3D-Body (70 MHR joints)
- **Multi-Person Support** with automatic tracking
- **Animated FBX Export** with Blender integration

## Installation

### Prerequisites

- Python 3.10+
- ComfyUI installed
- PyTorch with CUDA (recommended: 40GB+ VRAM for occlusion handling)
- Blender 3.0+ (for FBX export)
- Git

### Setup

1. **Clone repository:**
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/danielvw/ComfyUI-SAM-Body4D.git
   cd ComfyUI-SAM-Body4D
   ```

2. **Install SAM-Body4D:**
   ```bash
   cd ..  # back to custom_nodes/
   git clone https://github.com/gaomingqi/sam-body4d.git
   cd sam-body4d
   pip install -e .
   ```

3. **Install node suite dependencies:**
   ```bash
   cd ../ComfyUI-SAM-Body4D
   python install.py
   ```

4. **Download SAM-Body4D checkpoints:**
   ```bash
   cd ../sam-body4d
   python scripts/setup.py --ckpt-root checkpoints
   ```

   **Important:** SAM-3 and SAM-3D-Body require HuggingFace access:
   - [SAM-3](https://huggingface.co/facebook/sam3)
   - [SAM-3D-Body](https://huggingface.co/facebook/sam-3d-body-dinov3)

   Before downloading:
   ```bash
   huggingface-cli login
   ```

5. **Restart ComfyUI**

## Nodes

### LoadBody4DModel
Loads SAM-Body4D pipeline components.

**Inputs:**
- `config_path`: Path to body4d.yaml (default: sam-body4d/configs/body4d.yaml)
- `enable_occlusion`: Enables Diffusion-VAS for occlusion handling (requires 40GB+ VRAM)
- `batch_size`: Batch size for SAM-3D-Body (default: 64)

**Outputs:**
- `model`: Model bundle for other nodes

### Body4DImageSequence
Validates and prepares image batch.

**Inputs:**
- `images`: ComfyUI IMAGE tensor (batch of frames)
- `fps`: Frames per second (default: 30.0)
- `start_frame`, `end_frame`: Frame range for slicing

**Outputs:**
- `images`: Processed images
- `frame_count`: Number of frames
- `fps`: FPS value

### Body4DProcess
Executes complete Body4D pipeline.

**Inputs:**
- `model`: From LoadBody4DModel
- `images`: Image sequence
- `fps`: FPS value
- `detection_threshold`: Confidence for human detection (default: 0.5)
- `max_persons`: Maximum number of persons to track (default: 5)
- `inference_type`: "full" (Body+Hands), "body", or "hand"

**Outputs:**
- `sequence_data`: Processed sequence with 3D data
- `preview`: Visualization with overlaid meshes

### Body4DSkeletonExtract
Extracts skeleton animation from sequence.

**Inputs:**
- `sequence_data`: From Body4DProcess
- `person_id`: Person ID (1-indexed)
- `joint_subset`: "full_70", "body_17", or "body_hands"
- `smooth_factor`: Temporal smoothing (0=off, 1=max)

**Outputs:**
- `animation`: Skeleton animation data

### Body4DExportFBX
Exports animated FBX via Blender.

**Inputs:**
- `animation`: From Body4DSkeletonExtract
- `output_filename`: Filename without .fbx
- `export_mesh`: Export mesh with skinning (experimental)
- `reference_frame`: Frame for bind pose mesh
- `blender_path`: Path to Blender (default: "blender")
- `output_dir`: Custom output directory

**Outputs:**
- `fbx_path`: Path to exported FBX file

## Workflow Example

```
[Load Video]
    ↓
[Body4DImageSequence]
    ↓
[LoadBody4DModel] → [Body4DProcess]
                          ↓
                    [Body4DSkeletonExtract]
                          ↓
                    [Body4DExportFBX]
```

## Technical Details

### MHR70 Skeleton
The system uses the Momentum Human Rig (MHR) with 70 joints:
- **Body:** 17 main joints (head, torso, arms, legs)
- **Feet:** 6 joints (toes, heels)
- **Hands:** 40 joints (20 per hand, 4 per finger)
- **Extra:** 7 additional joints (neck, olecranon, etc.)

### Performance

**Without Occlusion Handling:**
- VRAM: ~15-20 GB
- Time: ~1-3 minutes per 10-frame sequence

**With Occlusion Handling:**
- VRAM: 40-53 GB
- Time: ~10-30 minutes per 10-frame sequence

### Limitations

1. **SAM-3 Tracking:** Auto-detection in first frame, manual prompts currently not supported
2. **Mesh Export:** Skinning weights still experimental
3. **Blender Dependency:** FBX export requires Blender installation

## Troubleshooting

### "SAM-Body4D not found"
Make sure sam-body4d exists as a sibling directory:
```
ComfyUI/custom_nodes/
├── ComfyUI-SAM-Body4D/
└── sam-body4d/
```

### "Blender not found"
- Install Blender from https://www.blender.org/download/
- Or specify the full path in the node:
  ```
  /usr/bin/blender  (Linux)
  C:\Program Files\Blender Foundation\Blender 3.6\blender.exe  (Windows)
  ```

### "Model checkpoints not found"
Run the setup script:
```bash
cd sam-body4d
python scripts/setup.py --ckpt-root checkpoints
```

### CUDA Out of Memory
- Reduce `batch_size` in LoadBody4DModel
- Disable `enable_occlusion`
- Process shorter sequences (fewer frames)

## License

MIT License

## Credits

Based on:
- [SAM-Body4D](https://github.com/gaomingqi/sam-body4d) by Mingqi Gao et al.
- [ComfyUI-SAM3DBody](https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody) for architecture patterns
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## References

```bibtex
@article{gao2025sambody4d,
  title   = {SAM-Body4D: Training-Free 4D Human Body Mesh Recovery from Videos},
  author  = {Gao, Mingqi and Miao, Yunqi and Han, Jungong},
  journal = {arXiv preprint arXiv:2512.08406},
  year    = {2025}
}
```
