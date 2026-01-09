# Model Checkpoints Setup

## Required Checkpoints

### 1. SAM-3 (Video Tracking)
- **Source**: https://huggingface.co/facebook/sam3
- **File**: `sam3.pt`
- **Size**: ~2.4GB
- **Purpose**: Temporal video tracking and segmentation

**Download:**
```bash
# Requires HuggingFace CLI authentication
huggingface-cli login
huggingface-cli download facebook/sam3 sam3.pt --local-dir ./checkpoints/sam3
```

### 2. SAM-3D-Body (3D Mesh Estimation)
- **Source**: https://huggingface.co/facebook/sam-3d-body-dinov3
- **Files Needed**:
  - `sam_3d_body.pth` - Main model (~1.2GB)
  - `mhr_template.pkl` - MHR skeletal rig template
- **Purpose**: 3D body mesh and pose estimation

**Download:**
```bash
huggingface-cli download facebook/sam-3d-body-dinov3 --local-dir ./checkpoints/sam_3d_body
```

### 3. MOGE2 (FOV Estimation)
- **Source**: Part of SAM-3D-Body repository
- **File**: `moge2.pth`
- **Purpose**: Camera field-of-view estimation

**Download:**
```bash
# Usually included with SAM-3D-Body, or download separately:
wget https://example.com/moge2.pth -P ./checkpoints/moge2/
```

## Optional: Diffusion-VAS (Occlusion Handling)

Only needed if you enable `enable_occlusion: true` in config.

- **Requirements**: 40GB+ VRAM
- **Files**:
  - `mask_model.pth` - Amodal segmentation
  - `rgb_model.pth` - RGB inpainting
  - `depth_model.pth` - Depth completion

**Download:**
```bash
# Clone Diffusion-VAS repository
cd ../
git clone https://github.com/example/diffusion-vas.git
# Follow their setup instructions
```

## Directory Structure

Recommended checkpoint organization:

```
ComfyUI/
├── models/
│   └── body4d/                    # Registered with ComfyUI
│       ├── sam3/
│       │   └── sam3.pt
│       ├── sam_3d_body/
│       │   ├── sam_3d_body.pth
│       │   └── mhr_template.pkl
│       ├── moge2/
│       │   └── moge2.pth
│       └── diffusion_vas/         # Optional
│           ├── mask_model.pth
│           ├── rgb_model.pth
│           └── depth_model.pth
```

## Configuration

After downloading, create your config file:

```bash
cp configs/body4d_example.yaml configs/body4d.yaml
```

Edit `body4d.yaml` and update all paths:

```yaml
sam3:
  ckpt_path: "/path/to/ComfyUI/models/body4d/sam3/sam3.pt"

sam_3d_body:
  ckpt_path: "/path/to/ComfyUI/models/body4d/sam_3d_body/sam_3d_body.pth"
  mhr_path: "/path/to/ComfyUI/models/body4d/sam_3d_body/mhr_template.pkl"
  fov_path: "/path/to/ComfyUI/models/body4d/moge2/moge2.pth"
  batch_size: 64
```

## Quick Setup Script

```bash
#!/bin/bash
# setup_checkpoints.sh

CHECKPOINT_DIR="/path/to/ComfyUI/models/body4d"
mkdir -p "$CHECKPOINT_DIR"/{sam3,sam_3d_body,moge2}

# Login to HuggingFace
huggingface-cli login

# Download SAM-3
echo "Downloading SAM-3..."
huggingface-cli download facebook/sam3 sam3.pt --local-dir "$CHECKPOINT_DIR/sam3"

# Download SAM-3D-Body
echo "Downloading SAM-3D-Body..."
huggingface-cli download facebook/sam-3d-body-dinov3 \
  --local-dir "$CHECKPOINT_DIR/sam_3d_body"

echo "✓ Checkpoints downloaded!"
echo "Update config file with paths:"
echo "  $CHECKPOINT_DIR"
```

## Troubleshooting

### Access Denied from HuggingFace

Some models require accepting terms:
1. Visit the model page on HuggingFace
2. Read and accept the license agreement
3. Try downloading again

### Out of Disk Space

Total size: ~5-8GB (without occlusion models)
Make sure you have at least 10GB free space.

### Wrong Checkpoint Format

If you get "unexpected checkpoint format" errors:
- Verify you downloaded the correct model version
- Check file integrity (compare file sizes)
- Re-download if corrupted
