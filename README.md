# TotalVibeSegmentator: Volumetric interpolated breath-hold examination torso segmentation for the NAKO and UKBB full body images  

![3D Render](/imgs/3d_render_github.png)
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
## Label overview
![Slices](/imgs/slices_github.jpg)

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
|65|outer_skin|
|66|muscle|
|67|inner_fat|
|68|IVD|
|69|vertebra_body|
|70|vertebra_posterior_elements|
|71|spinal_channel|
|72|bone_other|
