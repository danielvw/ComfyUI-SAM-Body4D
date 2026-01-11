# ComfyUI-SAM-Body4D

[![en](https://img.shields.io/badge/lang-en-blue.svg)](README.md)
[![de](https://img.shields.io/badge/lang-de-yellow.svg)](README_de.md)

![Status](https://img.shields.io/badge/status-WIP-orange)

> [!WARNING]
> Dieses Projekt ist aktuell in Entwicklung und **nicht lauffähig**.

Custom Node Suite für ComfyUI zur Konvertierung von Image Sequences in animierte FBX-Dateien mit Skelett-Rigging.

## Überblick

Diese Node-Suite integriert SAM-Body4D in ComfyUI und ermöglicht:
- **Video Tracking** mit SAM-3 für temporale Konsistenz
- **3D Mesh Estimation** mit SAM-3D-Body (70 MHR-Joints)
- **Multi-Person Support** mit automatischem Tracking
- **Animierter FBX Export** mit Blender-Integration

## Installation

### Voraussetzungen

- Python 3.10+
- ComfyUI installiert
- PyTorch mit CUDA (empfohlen: 40GB+ VRAM für Occlusion-Handling)
- Blender 3.0+ (für FBX-Export)
- Git

### Setup

1. **Repository klonen:**
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/danielvw/ComfyUI-SAM-Body4D.git
   cd ComfyUI-SAM-Body4D
   ```

2. **SAM-Body4D installieren:**
   ```bash
   cd ..  # zurück zu custom_nodes/
   git clone https://github.com/gaomingqi/sam-body4d.git
   cd sam-body4d
   pip install -e .
   ```

3. **Node-Suite Abhängigkeiten installieren:**
   ```bash
   cd ../ComfyUI-SAM-Body4D
   python install.py
   ```

4. **SAM-Body4D Checkpoints herunterladen:**
   ```bash
   cd ../sam-body4d
   python scripts/setup.py --ckpt-root checkpoints
   ```

   **Wichtig:** SAM-3 und SAM-3D-Body benötigen HuggingFace-Zugang:
   - [SAM-3](https://huggingface.co/facebook/sam3)
   - [SAM-3D-Body](https://huggingface.co/facebook/sam-3d-body-dinov3)

   Vor dem Download:
   ```bash
   huggingface-cli login
   ```

5. **ComfyUI neu starten**

## Nodes

### LoadBody4DModel
Lädt SAM-Body4D Pipeline-Komponenten.

**Inputs:**
- `config_path`: Pfad zur body4d.yaml (default: sam-body4d/configs/body4d.yaml)
- `enable_occlusion`: Aktiviert Diffusion-VAS für Occlusion-Handling (benötigt 40GB+ VRAM)
- `batch_size`: Batch-Größe für SAM-3D-Body (default: 64)

**Outputs:**
- `model`: Model-Bundle für weitere Nodes

### Body4DImageSequence
Validiert und bereitet Image-Batch vor.

**Inputs:**
- `images`: ComfyUI IMAGE Tensor (Batch von Frames)
- `fps`: Frames per second (default: 30.0)
- `start_frame`, `end_frame`: Frame-Range für Slicing

**Outputs:**
- `images`: Verarbeitete Images
- `frame_count`: Anzahl Frames
- `fps`: FPS-Wert

### Body4DProcess
Führt vollständige Body4D Pipeline aus.

**Inputs:**
- `model`: Von LoadBody4DModel
- `images`: Image Sequence
- `fps`: FPS-Wert
- `detection_threshold`: Confidence für Human-Detection (default: 0.5)
- `max_persons`: Maximale Anzahl zu trackender Personen (default: 5)
- `inference_type`: "full" (Body+Hands), "body", oder "hand"

**Outputs:**
- `sequence_data`: Verarbeitete Sequence mit 3D-Daten
- `preview`: Visualisierung mit überlagerten Meshes

### Body4DSkeletonExtract
Extrahiert Skelett-Animation aus Sequence.

**Inputs:**
- `sequence_data`: Von Body4DProcess
- `person_id`: Person-ID (1-indexed)
- `joint_subset`: "full_70", "body_17", oder "body_hands"
- `smooth_factor`: Temporal Smoothing (0=aus, 1=max)

**Outputs:**
- `animation`: Skelett-Animation Daten

### Body4DExportFBX
Exportiert animiertes FBX via Blender.

**Inputs:**
- `animation`: Von Body4DSkeletonExtract
- `output_filename`: Dateiname ohne .fbx
- `export_mesh`: Mesh mit Skinning exportieren (experimental)
- `reference_frame`: Frame für Bind-Pose Mesh
- `blender_path`: Pfad zu Blender (default: "blender")
- `output_dir`: Custom Output-Verzeichnis

**Outputs:**
- `fbx_path`: Pfad zur exportierten FBX-Datei

## Workflow-Beispiel

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

## Technische Details

### MHR70 Skelett
Das System nutzt das Momentum Human Rig (MHR) mit 70 Joints:
- **Body:** 17 Haupt-Gelenke (Kopf, Torso, Arme, Beine)
- **Feet:** 6 Gelenke (Zehen, Fersen)
- **Hands:** 40 Gelenke (20 pro Hand, 4 pro Finger)
- **Extra:** 7 Zusatz-Gelenke (Neck, Olecranon, etc.)

### Performance

**Ohne Occlusion-Handling:**
- VRAM: ~15-20 GB
- Zeit: ~1-3 Minuten pro 10-Frame Sequenz

**Mit Occlusion-Handling:**
- VRAM: 40-53 GB
- Zeit: ~10-30 Minuten pro 10-Frame Sequenz

### Limitierungen

1. **SAM-3 Tracking:** Auto-Detection im ersten Frame, manuelle Prompts derzeit nicht unterstützt
2. **Mesh Export:** Skinning-Weights noch experimental
3. **Blender Abhängigkeit:** FBX-Export benötigt Blender-Installation

## Troubleshooting

### "SAM-Body4D not found"
Stelle sicher, dass sam-body4d als Sibling-Verzeichnis existiert:
```
ComfyUI/custom_nodes/
├── ComfyUI-SAM-Body4D/
└── sam-body4d/
```

### "Blender not found"
- Installiere Blender von https://www.blender.org/download/
- Oder gib den vollständigen Pfad in der Node an:
  ```
  /usr/bin/blender  (Linux)
  C:\Program Files\Blender Foundation\Blender 3.6\blender.exe  (Windows)
  ```

### "Model checkpoints not found"
Führe das Setup-Script aus:
```bash
cd sam-body4d
python scripts/setup.py --ckpt-root checkpoints
```

### CUDA Out of Memory
- Reduziere `batch_size` in LoadBody4DModel
- Deaktiviere `enable_occlusion`
- Verarbeite kürzere Sequenzen (weniger Frames)

## Lizenz

MIT License

## Credits

Basiert auf:
- [SAM-Body4D](https://github.com/gaomingqi/sam-body4d) von Mingqi Gao et al.
- [ComfyUI-SAM3DBody](https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody) für Architektur-Patterns
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## Referenzen

```bibtex
@article{gao2025sambody4d,
  title   = {SAM-Body4D: Training-Free 4D Human Body Mesh Recovery from Videos},
  author  = {Gao, Mingqi and Miao, Yunqi and Han, Jungong},
  journal = {arXiv preprint arXiv:2512.08406},
  year    = {2025}
}
```
