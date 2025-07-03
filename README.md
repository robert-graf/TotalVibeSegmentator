# TotalVibeSegmentator:  Full Body MRI Segmentation for the NAKO and UK Biobank 

![3D Render](/imgs/3d_render_github.png)

## Versions

You can select older versions with `--dataset_id [ID]`

|ID   |             NAME             | Version                       |Modality       | Resolution          |Link|
|---  | ----------------             | ------------------------------|--------       | --------------------|----|
|100  |TotalVIBESegmentator          | current                       | MRI/CT        | axial 1.41 mm, S=3mm|[preprint](https://arxiv.org/abs/2406.00125)|
|99   |TotalVIBESegmentator          | current (sagittal resolution) | MRI/CT        | sagittal 1.41 mm, R=3mm|[preprint](https://arxiv.org/abs/2406.00125)|
|278  |Splits the body in 11 regions |current                        | NAKO VIBE-only| iso 4 mm            |    |
|282  |Torso Water-Fat-swap detection|current (use script from linked repo)      | multi-echo-Vibe VIBE  | axial 1.41 mm, S=3mm|[preprint](https://arxiv.org/abs/2502.14659); [Other Repo](https://github.com/robert-graf/MAGO-SP)|
|511  |Spine Vertebra instance       |Fast, but may produce off-by-one errors. Use [SPINEPS](https://github.com/Hendrik-code/spineps) if this method fails.| VIBE-only     | axial 1.41 mm, S=3mm|[See also](https://github.com/Hendrik-code/spineps) |
|512  |Spine Vertebra subregion      |incorporated in SPINEPS       | VIBE-only     | axial 1.41 mm, S=3mm|[See also](https://github.com/Hendrik-code/spineps) |
|520  | CT Bone segmentations        | preliminary                   | CT            | iso 0.8 mm          |     |

### Outdated/deprecated

|ID   |             NAME             | Version                       |Modality       | Resolution          |Link|
|---  | ----------------             | ------------------------------|--------       | --------------------|----|
|98   |TotalVIBESegmentator| (deprecated-spinal channel bugged in CT)| MRI| axial 1.41 mm, S=3mm|    |
|80   |TotalVIBESegmentator          | preprint (deprecated)| MRI           | axial 1.41 mm, S=3mm|    |
|85-87|TotalVIBESegmentator          | early preprint (deprecated)   | MRI           | axial 1.41 mm, S=3mm|    |


## Installation Guide

### System Requirements
- Python 3.10 or higher.
- Tested on Ubuntu and Windows.
- One of the following:
  - Nvidia-GPU with 4 GB of RAM or more.
  - A mps device (like a newer Mac with M2/M3 , but we could not test this.) (--ddevice mps)
  - A strong CPU (--ddevice cpu)

### Installation Guide

1. Open a command line (search for Terminal on Mac or cmd on Windows in your OS search).
2. Ensure you have Anaconda (Python) installed.
3. Navigate to the folder where you want to download the script using `cd [FOLDER PATH]`.

```bash
# Run this commands by coping in to the Terminal
# Recommended: make a virtual Python environment (example shows Anaconda)
conda create -n "TotalVibeSegmentator" python=3.11.0
conda activate TotalVibeSegmentator

# Install PyTorch that works with your GPU (follow instructions at https://pytorch.org/get-started/locally/)
pip install torch torchvision torchaudio

# Install required Python packages
pip install TPTBox ruamel.yaml configargparse
pip install nnunetv2 
# We recommend the newest versions. Tested versions: TPTBox==1.6, ruamel.yaml==0.18.6, configargparse==1.7, nnunetv2==2.4.2

# If e. g. nnunetv2 does not work, try version 2.4.2
# Uninstall the current version and reinstall with the specified version
#pip uninstall nnunetv2
#pip install nnunetv2==2.4.2

# Download the scripts (they will be downloaded to your current folder)
git clone https://github.com/robert-graf/TotalVibeSegmentator.git
cd TotalVibeSegmentator
```
It should take less than 30 minutes to install, excluding installing conda/python. The longest time takes to install pytroch. 
## Download

Download the nnUNet weights automatically. They are put in `[Path to this project]/nnUNet/nnUNet_results/`. 

## Run
```bash
# Run this commands in a Terminal where you navigate to the folder where `run_TotalVibeSegmentator.py` is
conda activate TotalVibeSegmentator

# Total Segmentation
python run_TotalVibeSegmentator.py --img [IMAGE-PATH] --out_path [OUTPATH] --ddevice [cpu|cuda|mpu (optional)]

# Total Segmentation with postprocessing and combining of masks
python run_TotalVibeSegmentator_multi.py --img_inphase [IMAGE-PATH] --img_water [IMAGE-PATH] --img_outphase [IMAGE-PATH]  --out_path [OUTPATH] --ddevice [cpu|cuda|mpu (optional)]

# Spine Instance
python run_instance_spine_segmentation.py --img [IMAGE-PATH] --out_path [OUTPATH] --ddevice [cpu|cuda|mpu (optional)]

# Spine Semantic
python run_semantic_spine_segmentation.py --img [IMAGE-PATH] --out_path [OUTPATH] --ddevice [cpu|cuda|mpu (optional)]


```

### Running an Example
```bash
conda activate TotalVibeSegmentator
#cd to the folder where you downloaded the GitHub files (the folder where the run_TotalVibeSegmentator.py is located)
```
Let's run some examples that were never seen during training. You can download all four examples from [here](https://github.com/robert-graf/TotalVibeSegmentator/releases/download/example/example_mri.zip) 

(Source: [link](https://zenodo.org/records/11367005) Number s0284, s0286, s0287, s0288)

Unzip the file. For this example, we copied all files to a folder called 'example' where the run_TotalVibeSegmentator.py is located.
```bash

python run_TotalVibeSegmentator.py --img example/mri1.nii.gz --out_path  example/seg1.nii.gz

python run_TotalVibeSegmentator.py --img example/mri2.nii.gz --out_path  example/seg2.nii.gz

python run_TotalVibeSegmentator.py --img example/mri3.nii.gz --out_path  example/seg3.nii.gz

python run_TotalVibeSegmentator.py --img example/mri4.nii.gz --out_path  example/seg4.nii.gz
```
The segmentation is stored in "example/segX.nii.gz." You can view the results with [ITKSnap](http://www.itksnap.org/pmwiki/pmwiki.php) by first drag-and-drop the mriX.nii.gz and then drag-and-drop the segX.nii.gz (with X is a number from before) and click on "Open as Segmentation"

To run this script on your data, update the paths.

## Label overview


|ID | NAME|
| -------- | --------|
|1|spleen|
|2|kidney_right|
|3|kidney_left|
|4|gallbladder|
|5|liver|
|6|stomach|
|7|pancreas|
|8|adrenal_gland_right|
|9|adrenal_gland_left|
|10|lung_upper_lobe_left|
|11|lung_lower_lobe_left|
|12|lung_upper_lobe_right|
|13|lung_middle_lobe_right|
|14|lung_lower_lobe_right|
|15|esophagus|
|16|trachea|
|17|thyroid_gland|
|18|intestine|
|19|duodenum|
|20|unused|
|21|urinary_bladder|
|22|prostate|
|23|sacrum|
|24|heart|
|25|aorta|
|26|pulmonary_vein|
|27|brachiocephalic_trunk|
|28|subclavian_artery_right|
|29|subclavian_artery_left|
|30|common_carotid_artery_right|
|31|common_carotid_artery_left|
|32|brachiocephalic_vein_left|
|33|brachiocephalic_vein_right|
|34|atrial_appendage_left|
|35|superior_vena_cava|
|36|inferior_vena_cava|
|37|portal_vein_and_splenic_vein|
|38|iliac_artery_left|
|39|iliac_artery_right|
|40|iliac_vena_left|
|41|iliac_vena_right|
|42|humerus_left|
|43|humerus_right|
|44|scapula_left|
|45|scapula_right|
|46|clavicula_left|
|47|clavicula_right|
|48|femur_left|
|49|femur_right|
|50|hip_left|
|51|hip_right|
|52|spinal_cord|
|53|gluteus_maximus_left|
|54|gluteus_maximus_right|
|55|gluteus_medius_left|
|56|gluteus_medius_right|
|57|gluteus_minimus_left|
|58|gluteus_minimus_right|
|59|autochthon_left|
|60|autochthon_right|
|61|iliopsoas_left|
|62|iliopsoas_right|
|63|sternum|
|64|costal_cartilages|
|65|subcutaneous_fat|
|66|muscle|
|67|inner_fat|
|68|IVD|
|69|vertebra_body|
|70|vertebra_posterior_elements|
|71|spinal_channel|
|72|bone_other|



## How to Cite
The related paper is available as preprint: [arXiv:2406.00125](https://arxiv.org/abs/2406.00125)
```
@article{graf2024totalvibesegmentator,
  title={TotalVibeSegmentator:  Full Body MRI Segmentation for the NAKO and UK Biobank },
  author={Graf, Robert and Platzek, Paul-S{\"o}ren and Riedel, Evamaria Olga and Ramsch{\"u}tz, Constanze and Starck, Sophie and M{\"o}ller, Hendrik Kristian and Atad, Matan and V{\"o}lzke, Henry and B{\"u}low, Robin and Schmidt, Carsten Oliver and others},
  journal={arXiv preprint arXiv:2406.00125},
  year={2024}
}
```

## Other networks

`run_TotalVibeSegmentator.py [...] --dataset_id 278` 

![Slices](/imgs/roi.jpg)
