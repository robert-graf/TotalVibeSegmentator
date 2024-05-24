import os
import shutil
import sys
import time
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import numpy as np

try:
    from fury import actor, window
    from totalsegmentator.map_to_binary import class_map
    from totalsegmentator.vtk_utils import plot_mask
except Exception:
    print("This code uses parts of totalsegmentator; use 'pip install TotalSegmentator' to install it")
    raise
from TPTBox import NII
from tqdm import tqdm
from xvfbwrapper import Xvfb

from TypeSaveArgParse.autoargs import Class_to_ArgParse

labels = {
    1: {"typ": "organ", "name": "spleen", "min": 1, "max": 1, "autofix": 100},
    2: {"typ": "organ", "name": "kidney_right", "min": 1, "max": 1, "autofix": 100},
    3: {"typ": "organ", "name": "kidney_left", "min": 1, "max": 1, "autofix": 100},
    4: {"typ": "organ", "name": "gallbladder", "min": 1, "max": 1, "autofix": 40},
    5: {"typ": "organ", "name": "liver", "min": 1, "max": 1, "autofix": 200},
    6: {"typ": "digenstion", "name": "stomach", "min": 1, "max": 1, "autofix": 10},
    7: {"typ": "digenstion", "name": "pancreas", "min": 1, "max": 1, "rois": [4, 3, 8, 9], "autofix": 5},
    8: {"typ": "vessel", "name": "adrenal_gland_right", "min": 1, "max": 1, "autofix": 30},
    9: {"typ": "vessel", "name": "adrenal_gland_left", "min": 1, "max": 1, "autofix": 30},
    10: {"typ": "lung", "name": "lung_upper_lobe_left", "min": 1, "max": 1, "rois": [5, 6, 9, 10], "autofix": 100, "rm_roi": [1, 2, 3]},
    11: {"typ": "lung", "name": "lung_lower_lobe_left", "min": 1, "max": 1, "rois": [5, 6, 9, 10], "autofix": 100, "rm_roi": [1, 2, 3]},
    12: {"typ": "lung", "name": "lung_upper_lobe_right", "min": 1, "max": 1, "rois": [5, 6, 9, 10], "autofix": 100, "rm_roi": [1, 2, 3]},
    13: {"typ": "lung", "name": "lung_middle_lobe_right", "min": 1, "max": 1, "rois": [5, 6, 9, 10], "autofix": 100, "rm_roi": [1, 2, 3]},
    14: {"typ": "lung", "name": "lung_lower_lobe_right", "min": 1, "max": 1, "rois": [5, 6, 9, 10], "autofix": 100, "rm_roi": [1, 2, 3]},
    15: {"typ": "digenstion", "name": "esophagus", "min": 1, "max": 1, "rois": [5, 9, 10], "autofix": 10},
    16: {"typ": "lung", "name": "trachea", "min": 1, "max": 1, "autofix": 10},
    17: {"typ": "organ", "name": "thyroid_gland", "min": 2, "max": 2, "autofix": 10},
    18: {"typ": "digenstion", "name": "intestine", "min": 1, "max": 1, "autofix": 20},
    19: {"typ": "digenstion", "name": "duodenum", "min": 1, "max": 1, "autofix": 10},
    20: {"typ": "digenstion", "name": "unused", "min": 1, "max": 1, "autofix": 10},
    21: {"typ": "organ", "name": "urinary_bladder", "min": 1, "max": 1, "autofix": 15},
    22: {"typ": "organ", "name": "prostate", "min": 1, "max": 1, "autofix": 10},
    23: {"typ": "bone", "name": "sacrum", "min": 1, "max": 1, "autofix": 10},
    24: {"typ": "organ", "name": "heart", "min": 1, "max": 1, "autofix": 10},
    25: {"typ": "vessel", "name": "aorta", "min": 1, "max": 1, "autofix": 10},
    26: {"typ": "vessel", "name": "pulmonary_vein", "min": 2, "max": 2, "autofix": 30},
    27: {"typ": "vessel", "name": "brachiocephalic_trunk", "min": 1, "max": 1, "autofix": 30},
    28: {"typ": "vessel", "name": "subclavian_artery_right", "min": 1, "max": 1, "autofix": 30},
    29: {"typ": "vessel", "name": "subclavian_artery_left", "min": 1, "max": 1, "autofix": 30},
    30: {"typ": "vessel", "name": "common_carotid_artery_right", "min": 1, "max": 1, "autofix": 30},
    31: {"typ": "vessel", "name": "common_carotid_artery_left", "min": 1, "max": 1, "autofix": 30},
    32: {"typ": "vessel", "name": "brachiocephalic_vein_left", "min": 1, "max": 1, "autofix": 30},
    33: {"typ": "vessel", "name": "brachiocephalic_vein_right", "min": 1, "max": 1, "autofix": 30},
    34: {"typ": "vessel", "name": "atrial_appendage_left", "min": 1, "max": 1, "autofix": 30},
    35: {"typ": "vessel", "name": "superior_vena_cava", "min": 1, "max": 1, "autofix": 30},
    36: {"typ": "vessel", "name": "inferior_vena_cava", "min": 1, "max": 1, "autofix": 30},
    37: {"typ": "vessel", "name": "portal_vein_and_splenic_vein", "min": 1, "max": 1, "autofix": 30},
    38: {"typ": "vessel", "name": "iliac_artery_left", "min": 1, "max": 1, "autofix": 30},
    39: {"typ": "vessel", "name": "iliac_artery_right", "min": 1, "max": 1, "autofix": 30},
    40: {"typ": "vessel", "name": "iliac_vena_left", "min": 1, "max": 1, "autofix": 30},
    41: {"typ": "vessel", "name": "iliac_vena_right", "min": 1, "max": 1, "autofix": 30},
    42: {"typ": "bone", "name": "humerus_left", "min": 1, "max": 1, "autofix": 200},
    43: {"typ": "bone", "name": "humerus_right", "min": 1, "max": 1, "autofix": 200},
    44: {"typ": "bone", "name": "scapula_left", "min": 1, "max": 1, "autofix": 50},
    45: {"typ": "bone", "name": "scapula_right", "min": 1, "max": 1, "autofix": 50},
    46: {"typ": "bone", "name": "clavicula_left", "min": 1, "max": 1, "autofix": 50},
    47: {"typ": "bone", "name": "clavicula_right", "min": 1, "max": 1, "autofix": 50},
    48: {"typ": "bone", "name": "femur_left", "min": 1, "max": 1, "autofix": 200},
    49: {"typ": "bone", "name": "femur_right", "min": 1, "max": 1, "autofix": 200},
    50: {"typ": "bone", "name": "hip_left", "min": 1, "max": 1, "autofix": 100},
    51: {"typ": "bone", "name": "hip_right", "min": 1, "max": 1, "autofix": 100},
    52: {"typ": "cns", "name": "spinal_cord", "min": 1, "max": 1, "autofix": 5},
    53: {"typ": "muscle", "name": "gluteus_maximus_left", "min": 1, "max": 1, "autofix": 400},
    54: {"typ": "muscle", "name": "gluteus_maximus_right", "min": 1, "max": 1, "autofix": 400},
    55: {"typ": "muscle", "name": "gluteus_medius_left", "min": 1, "max": 1, "autofix": 400},
    56: {"typ": "muscle", "name": "gluteus_medius_right", "min": 1, "max": 1, "autofix": 400},
    57: {"typ": "muscle", "name": "gluteus_minimus_left", "min": 1, "max": 1, "autofix": 400},
    58: {"typ": "muscle", "name": "gluteus_minimus_right", "min": 1, "max": 1, "autofix": 400},
    59: {"typ": "muscle", "name": "autochthon_left", "min": 1, "max": 1, "autofix": 100},
    60: {"typ": "muscle", "name": "autochthon_right", "min": 1, "max": 1, "autofix": 100},
    61: {"typ": "muscle", "name": "iliopsoas_left", "min": 1, "max": 1, "autofix": 600},
    62: {"typ": "muscle", "name": "iliopsoas_right", "min": 1, "max": 1, "autofix": 600},
    63: {"typ": "bone", "name": "sternum", "min": 1, "max": 1, "autofix": 15, "rois": [4, 5, 6, 9, 10], "rm_roi": [1, 2, 3]},
    64: {"typ": "bone", "name": "costal_cartilages", "min": 10, "max": 30, "autofix": 2, "rois": [4, 5, 6, 9, 10], "rm_roi": [1, 2, 3]},
    65: {"typ": "rest", "name": "outer_skin", "min": 1, "max": 1000, "autofix": 2},
    66: {"typ": "rest", "name": "muscle", "min": 1, "max": 1000, "autofix": 2},
    67: {"typ": "rest", "name": "inner_fat", "min": 1, "max": 1000, "autofix": 2},
    68: {"typ": "bone", "name": "IVD", "min": 1, "max": 25, "autofix": 2},
    69: {"typ": "bone", "name": "vertebra_body", "min": 1, "max": 25, "autofix": 2},
    70: {"typ": "bone", "name": "vertebra_posterior_elements", "min": 1, "max": 25},
    71: {"typ": "cns", "name": "spinal_channel", "min": 1, "max": 1, "autofix": 5},
    72: {"typ": "bone", "name": "bone_other", "min": 0, "max": 10, "autofix": 50},
}

id_mapping = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "adrenal_gland_right",
    9: "adrenal_gland_left",
    10: "lung_upper_lobe_left",
    11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
    15: "esophagus",
    16: "trachea",
    17: "thyroid_gland",
    18: "small_bowel",
    19: "duodenum",
    20: "colon",
    21: "urinary_bladder",
    22: "prostate",
    # 23: "kidney_cyst_left",
    # 24: "kidney_cyst_right",
    25: "sacrum",
    # 26: "vertebrae_S1",
    # 27: "vertebrae_L5",
    # 28: "vertebrae_L4",
    # 29: "vertebrae_L3",
    # 30: "vertebrae_L2",
    # 31: "vertebrae_L1",
    # 32: "vertebrae_T12",
    # 33: "vertebrae_T11",
    # 34: "vertebrae_T10",
    # 35: "vertebrae_T9",
    # 36: "vertebrae_T8",
    # 37: "vertebrae_T7",
    # 38: "vertebrae_T6",
    # 39: "vertebrae_T5",
    # 40: "vertebrae_T4",
    # 41: "vertebrae_T3",
    # 42: "vertebrae_T2",
    # 43: "vertebrae_T1",
    # 44: "vertebrae_C7",
    # 45: "vertebrae_C6",
    # 46: "vertebrae_C5",
    # 47: "vertebrae_C4",
    # 48: "vertebrae_C3",
    # 49: "vertebrae_C2",
    # 50: "vertebrae_C1",
    51: "heart",
    52: "aorta",
    53: "pulmonary_vein",
    54: "brachiocephalic_trunk",
    55: "subclavian_artery_right",
    56: "subclavian_artery_left",
    57: "common_carotid_artery_right",  # 30
    58: "common_carotid_artery_left",
    59: "brachiocephalic_vein_left",
    60: "brachiocephalic_vein_right",
    61: "atrial_appendage_left",
    62: "superior_vena_cava",
    63: "inferior_vena_cava",
    64: "portal_vein_and_splenic_vein",
    65: "iliac_artery_left",
    66: "iliac_artery_right",
    67: "iliac_vena_left",  # 40
    68: "iliac_vena_right",
    69: "humerus_left",
    70: "humerus_right",
    71: "scapula_left",
    72: "scapula_right",
    73: "clavicula_left",
    74: "clavicula_right",  #! Known
    75: "femur_left",
    76: "femur_right",
    77: "hip_left",
    78: "hip_right",
    79: "spinal_cord",
    80: "gluteus_maximus_left",
    81: "gluteus_maximus_right",
    82: "gluteus_medius_left",
    83: "gluteus_medius_right",
    84: "gluteus_minimus_left",
    85: "gluteus_minimus_right",
    86: "autochthon_left",
    87: "autochthon_right",
    88: "iliopsoas_left",
    89: "iliopsoas_right",
    # 90: "brain",  #!
    # 91: "skull",  #!
    # 92: "rib_right_4",
    # 93: "rib_right_3",
    # 94: "rib_left_1",
    # 95: "rib_left_2",
    # 96: "rib_left_3",
    # 97: "rib_left_4",
    # 98: "rib_left_5",
    # 99: "rib_left_6",
    # 100: "rib_left_7",
    # 101: "rib_left_8",
    # 102: "rib_left_9",
    # 103: "rib_left_10",
    # 104: "rib_left_11",
    # 105: "rib_left_12",
    # 106: "rib_right_1",
    # 107: "rib_right_2",
    # 108: "rib_right_5",
    # 109: "rib_right_6",
    # 110: "rib_right_7",
    # 111: "rib_right_8",
    # 112: "rib_right_9",
    # 113: "rib_right_10",
    # 114: "rib_right_11",
    # 115: "rib_right_12",
    116: "sternum",  #!
    117: "costal_cartilages",
    118: "outer_skin",
    119: "muscle",
    120: "inner_fat",
    121: "IVD",
    122: "vertebra_body",
    123: "vertebra_posterior_elements",
    124: "spinal_channel",  #!
    125: "bone_other",
    # 125: "rib",
    # 126: "rib_area",
    # 127: "fat_muskel",
}
dataset_mapping = {}
array_mapping_ids = {94: 0}
for e, (k, v) in enumerate(id_mapping.items(), 1):
    dataset_mapping[e] = v
    array_mapping_ids[k] = e
dataset_mapping[0] = "background"


roi_groups = {}
for k, v in id_mapping.items():
    class_map["total"][k] = v
keys = sorted({v["typ"] for v in labels.values()})
keys.remove("cns")
keys.remove("rest")
keys.append("cns")
keys.append("rest")
task_name = "total"

roi_groups[task_name] = [[v["name"] for v in labels.values() if v["typ"] == k] for k in keys]
reverse = {v: k for k, v in array_mapping_ids.items()}
np.random.seed(1234)
random_colors = np.random.rand(100, 4)


def plot_roi_group(ref_img, scene, rois, x, y, smoothing, roi_data, affine, task_name):
    # ref_img = nib.load(subject_path)
    # roi_actors = []

    for idx, roi in enumerate(rois):
        color = random_colors[idx]

        classname_2_idx = {v: k for k, v in class_map[task_name].items()}
        data = roi_data == classname_2_idx[roi]
        # data = data.astype(np.uint8)  # needed?

        if data.max() > 0:  # empty mask
            affine[:3, 3] = 0  # make offset the same for all subjects
            cont_actor = plot_mask(scene, data, affine, x, y, smoothing=smoothing, color=color, opacity=1)
            scene.add(cont_actor)
            # roi_actors.append(cont_actor)


def plot_subject(ct_img, output_path, roi_data=None, smoothing=20, task_name="total"):
    subject_width = 330
    # subject_height = 700
    nr_cols = 9

    # window_size = (2000, 400)
    window_size = (2000, 1200)  # if we need higher res image of single class

    scene = window.Scene()
    showm = window.ShowManager(scene, size=window_size, reset_camera=False)
    showm.initialize()

    for idx, roi_group in enumerate(roi_groups[task_name]):
        x = (idx % nr_cols) * subject_width
        y = 0
        plot_roi_group(ct_img, scene, roi_group, x, y, smoothing, roi_data, ct_img.affine, task_name)

    scene.projection(proj_type="parallel")
    scene.reset_camera_tight(margin_factor=1.02)  # need to do reset_camera=False in record for this to work in

    output_path.parent.mkdir(parents=True, exist_ok=True)
    window.record(scene, size=window_size, out_path=output_path, reset_camera=False)  # , reset_camera=False
    scene.clear()


def generate_preview(ct_in, file_out, roi_data, smoothing, task_name):
    # np.random.seed(time.time_ns() % 2**20)
    # do not set random seed, otherwise can not call xvfb in parallel, because all generate same tmp dir
    # ??? Then why is setting this script a seed...
    with Xvfb() as xvfb:
        plot_subject(ct_in, file_out, roi_data, smoothing, task_name)


def snap_shot(
    paths: list[Path] | Path,
    snap_folder: Path | None = None,
    scale=2.0,
    cpus=None,
    name_addendum="",
    orientation: Literal["R", "A", "P", "L"] = "A",
    smoothing=50,
):
    if not isinstance(paths, list):
        paths = [paths]

    if snap_folder is not None:
        snap_folder.mkdir(exist_ok=True)
    cpus = cpus if cpus is not None else int(os.cpu_count()) // 2 + 1
    print("make snaps", len(paths))
    with Pool(cpus) as p:  # type: ignore
        p.map(
            partial(
                _make_img, out_folder=snap_folder, scale=scale, name_addendum=name_addendum, orientation=orientation, smoothing=smoothing
            ),
            paths,
        )


def _make_img(path: Path, out_folder: Path | None, scale, smoothing=50, name_addendum="", orientation="A"):
    o = ("R", "S", "A")
    st = time.time()
    nii = NII.load(path, True).rescale((scale, scale, scale)).reorient(o)

    nii.map_labels_(reverse, verbose=False)
    arr = nii.get_array()
    if orientation == "A":
        pass
    elif orientation == "L":
        arr = arr.swapaxes(0, 2)[:, :, ::-1].copy()
    elif orientation == "R":
        arr = arr.swapaxes(0, 2)
    elif orientation == "P":
        arr = arr[:, :, ::-1].copy()
    else:
        raise NotImplementedError(orientation)

    if name_addendum != "":
        name_addendum = "_desc-" + name_addendum if "-" not in name_addendum else "_" + name_addendum
    # preview_dir = path.parent
    # img = NII.load(preview_dir / "inphase.nii.gz", False).rescale((scale, scale, scale)).reorient(orientation)
    out2 = path.parent / f"{path.name.replace('.nii.gz','').rsplit('_',1)[0]}{name_addendum}_snp.png"
    generate_preview(nii.nii, out2, arr, smoothing, task_name)
    if out_folder is None:
        out_folder = Path(path).parent
        out = out_folder / f"{path.name.replace('.nii.gz','').rsplit('_',1)[0]}{name_addendum}_snp.png"
        shutil.copy(out2, out)
    print(f"  Generated in {time.time() - st:.2f}s")


@dataclass
class Arguments(Class_to_ArgParse):
    imgs: list[Path]
    override: bool = False


if __name__ == "__main__":
    import time

    t = time.time()
    arg = Arguments.get_opt()
    snap_shot(arg.imgs, name_addendum="A", orientation="A")
    snap_shot(arg.imgs, name_addendum="P", orientation="P")
    snap_shot(arg.imgs, name_addendum="R", orientation="R")
    snap_shot(arg.imgs, name_addendum="L", orientation="L")
    print(f"Took {time.time()-t} seconds.")
