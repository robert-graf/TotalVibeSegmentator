# TotalVibeSegmentator: Volumetric interpolated breath-hold examination torso segmentation for the NAKO and UKBB full body images  


## Installation

```bash
conda create -n "TotalVibeSegmentator" python=3.11.0  
conda activate TotalVibeSegmentator
## Install Pytorch that works with your GPU https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio
pip install TPTBox ruamel.yaml configargparse
pip install nnunetv2 
#if nnunetv2 does not work try version 2.2.1 (uninstall and reinstall with pip install nnunetv2==2.2.1)
```

## Download

Download the nnUNet weights from **TBA** and put them in [Path to this project]/nnUNet/nnUNet_results/
For the TotalVibeSegmentator you need the ROI model (278) and the newest TotalVibeSegmentator model

## Run
```bash
# Download the nnUNet weights and put them in [Path to this project]/nnUNet/nnUNet_results/
conda activate TotalVibeSegmentator
# Total Segmentation
python run_TotalVibeSegmentator.py --img [IMAGE-PATH] --out_path [OUTPATH] --roi_path [roi_out_path (optional)]
# Spine Instance
python run_instance_spine_segmentation.py --img [IMAGE-PATH] --out_path [OUTPATH]
# Spine Semantic
python run_semantic_spine_segmentation.py --img [IMAGE-PATH] --out_path [OUTPATH]
```