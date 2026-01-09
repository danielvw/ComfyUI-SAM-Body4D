# SAM-3 API Compatibility Guide

## Overview

This document describes the compatibility issues between two SAM-3 implementations that may be present in ComfyUI custom_nodes:

1. **sam-body4d** (`/home/daniel/scripts/sam-body4d`) - Original implementation from SAM-Body4D authors
2. **ComfyUI-SAM3** (`/home/daniel/scripts/ComfyUI-SAM3`) - Adapted version for ComfyUI

Our `ComfyUI-SAM-Body4D` node is designed to work with **both** implementations, but there are important differences to be aware of.

## Critical API Differences

### 1. `propagate_in_video()` Return Values ⚠️ CRITICAL

**sam-body4d:**
```python
yield frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores, iou_scores  # 6 values
```

**ComfyUI-SAM3:**
```python
yield frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores  # 5 values (no iou_scores)
```

**Our Solution:**
Lines 215-220 in `body4d_process.py` handle both formats:
```python
for result in predictor.propagate_in_video(...):
    if len(result) == 5:
        frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores = result
    elif len(result) == 6:
        frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores, iou_scores = result
```

---

### 2. CUDA Availability Checks

**sam-body4d:**
- Hardcoded `.cuda()` calls throughout
- Assumes CUDA is always available
- **Will crash on CPU-only systems**

**ComfyUI-SAM3:**
- Always checks `torch.cuda.is_available()`
- Falls back to CPU gracefully
- **More robust and portable**

**Our Solution:**
Lines 169-173 detect sam-body4d and verify CUDA is available:
```python
if sam3_impl == "sam-body4d" and not torch.cuda.is_available():
    raise RuntimeError("sam-body4d implementation requires CUDA...")
```

---

### 3. Autocast Dtype Selection

**sam-body4d:**
- Hardcoded to `torch.bfloat16`
- **Will crash on older GPUs** (pre-Ampere: Volta/Turing)

**ComfyUI-SAM3:**
- Dynamically selects dtype based on GPU capability:
  - Ampere+ (compute capability ≥ 8.0): `bfloat16`
  - Volta/Turing (compute capability ≥ 7.0): `float16`
  - Older GPUs: No autocast
- **Better GPU compatibility**

---

### 4. Device Transfer Operations

**sam-body4d:**
```python
tensor.cuda()  # Hardcoded
tensor.to(device, non_blocking=True)  # Always non_blocking
```

**ComfyUI-SAM3:**
```python
tensor.to(inference_state["device"])  # Uses state device
tensor.to(device, non_blocking=torch.cuda.is_available())  # Conditional
```

---

### 5. Safety Checks

**ComfyUI-SAM3 adds extra safety checks:**
- Empty mask checks before processing
- CPU-only environment handling
- GPU capability detection

**sam-body4d:**
- Assumes optimal environment
- Fewer safety checks

---

## Which Implementation Is Loaded?

Our code automatically detects which implementation is loaded at lines 158-166:

```python
if hasattr(predictor, 'bf16_context'):
    sam3_impl = "sam-body4d"
else:
    sam3_impl = "ComfyUI-SAM3"

print(f"[Body4D] Detected SAM-3 implementation: {sam3_impl}")
```

---

## Recommendations

### For Development:
- **Prefer ComfyUI-SAM3** for better portability and robustness
- Test on both CPU and CUDA environments
- Ensure only ONE SAM-3 implementation is in sys.path

### For Production:
- **Use sam-body4d** if you have CUDA + Ampere+ GPU (best performance)
- **Use ComfyUI-SAM3** if you need broader compatibility

### Path Management:
Our `nodes/__init__.py` ensures sam-body4d paths take precedence:
```python
SAM_BODY4D_PATH = Path(__file__).parent.parent.parent / "sam-body4d"
# Removes conflicting paths from ComfyUI-SAM3
# Adds sam-body4d paths to beginning of sys.path
```

---

## Implementation Comparison Table

| Feature | sam-body4d | ComfyUI-SAM3 | Winner |
|---------|-----------|-------------|--------|
| propagate_in_video returns | 6 values (iou) | 5 values | N/A (handled) |
| CPU support | ❌ Crashes | ✅ Works | ComfyUI-SAM3 |
| GPU dtype selection | bf16 only | Dynamic (bf16/fp16) | ComfyUI-SAM3 |
| Device handling | `.cuda()` hardcoded | Uses state device | ComfyUI-SAM3 |
| Safety checks | Minimal | Extensive | ComfyUI-SAM3 |
| Performance | Optimal on CUDA | Slightly slower | sam-body4d |
| Origin | Official authors | Community port | sam-body4d |

---

## Troubleshooting

### Error: "not enough values to unpack (expected 6, got 5)"
**Cause:** ComfyUI-SAM3 is loaded but code expects 6 values
**Solution:** Already fixed in lines 215-220 (flexible unpacking)

### Error: "CUDA error: no kernel image is available"
**Cause:** sam-body4d trying to use bf16 on incompatible GPU
**Solution:** Use ComfyUI-SAM3 or upgrade GPU to Ampere+

### Error: "module 'sam_3d_body' has no attribute..."
**Cause:** Wrong SAM-3 implementation loaded from sys.path
**Solution:** Check path priority in nodes/__init__.py

---

## File Locations

**sam-body4d implementation:**
- `/home/daniel/scripts/sam-body4d/models/sam3/sam3/model/sam3_tracking_predictor.py`

**ComfyUI-SAM3 implementation:**
- `/home/daniel/scripts/ComfyUI-SAM3/nodes/sam3_lib/model/sam3_tracking_predictor.py`

**Our compatibility layer:**
- `/home/daniel/scripts/ComfyUI-SAM-Body4D/nodes/processing/body4d_process.py` (lines 150-225)

---

## Version History

- **2026-01-09**: Initial compatibility analysis
- **2026-01-09**: Added flexible return value handling (5 vs 6 values)
- **2026-01-09**: Added implementation detection and CUDA checks

---

## See Also

- [body4d.yaml Configuration](../configs/body4d_example.yaml)
- [Checkpoints Setup Guide](../configs/CHECKPOINTS.md)
- [Example Workflow](../example_workflow.json)
