# Under Construction


## Installation

```bash
conda create -n "InstanceSegmentation" python=3.11.0  
conda activate InstanceSegmentation
## Install Pytorch that works with your GPU https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio
pip install TPTBox ruamel.yaml configargparse
pip install nnunetv2

#>=2.2.1
```

## Run

```bash
conda activate InstanceSegmentation
python run_axial_instance_5mm.py --img [IMAGE-PATH] --out_path [OUTPATH]
```