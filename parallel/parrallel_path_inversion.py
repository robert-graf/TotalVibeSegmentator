import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from TPTBox import NII, to_nii
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from parallel.get_gpu import check_gpu_memory
from run_TotalVibeSegmentator import run_total_seg

total_vibe_map = {
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
    18: "intestine",
    19: "duodenum",
    20: "unused",
    21: "urinary_bladder",
    22: "prostate",
    23: "sacrum",
    24: "heart",
    25: "aorta",
    26: "pulmonary_vein",
    27: "brachiocephalic_trunk",
    28: "subclavian_artery_right",
    29: "subclavian_artery_left",
    30: "common_carotid_artery_right",
    31: "common_carotid_artery_left",
    32: "brachiocephalic_vein_left",
    33: "brachiocephalic_vein_right",
    34: "atrial_appendage_left",
    35: "superior_vena_cava",
    36: "inferior_vena_cava",
    37: "portal_vein_and_splenic_vein",
    38: "iliac_artery_left",
    39: "iliac_artery_right",
    40: "iliac_vena_left",
    41: "iliac_vena_right",
    42: "humerus_left",
    43: "humerus_right",
    44: "scapula_left",
    45: "scapula_right",
    46: "clavicula_left",
    47: "clavicula_right",
    48: "femur_left",
    49: "femur_right",
    50: "hip_left",
    51: "hip_right",
    52: "spinal_cord",
    53: "gluteus_maximus_left",
    54: "gluteus_maximus_right",
    55: "gluteus_medius_left",
    56: "gluteus_medius_right",
    57: "gluteus_minimus_left",
    58: "gluteus_minimus_right",
    59: "autochthon_left",
    60: "autochthon_right",
    61: "iliopsoas_left",
    62: "iliopsoas_right",
    63: "sternum",
    64: "costal_cartilages",
    65: "subcutaneous_fat",
    66: "muscle",
    67: "inner_fat",
    68: "IVD",
    69: "vertebra_body",
    70: "vertebra_posterior_elements",
    71: "spinal_channel",
    72: "bone_other",
}


@dataclass
class IMG:
    """Represents an image with counts for water, fat, and disagreements (inversions),
    and calculates related percentages.
    """

    sub: str
    count_water: float
    count_fat: float
    count_disagree: float
    affected_structures: list[str] | None

    @property
    def percent(self) -> float:
        """Calculates the proportion of fat in the total water-fat composition."""
        total = self.count_fat + self.count_water
        if total == 0:
            return 1
        return self.count_fat / total

    @property
    def percent_dis(self) -> float:
        """Calculates the proportion of disagreements in the total composition."""
        total = self.count_fat + self.count_water + self.count_disagree
        return self.count_disagree / total


@dataclass
class Subject:
    """Represents a subject with associated image files and paths to processed outputs."""

    name: str
    inphase_file: str | Path
    outphase_file: str | Path
    water_file: str | Path
    fat_file: str | Path
    water_swap_out_file: Path
    fat_swap_out_file: Path
    total_vibe: str | Path | None = None


def run_single_water_fat_swap_detection(subject: Subject, override=False, gpu=0):
    """Runs segmentation on water and fat images for a subject.

    Args:
        subject (Subject): The subject containing paths to the images.
        override (bool): Whether to override existing files. Default is False.
        gpu (int): The GPU to use for processing.
    """
    run_total_seg(
        [subject.water_file, subject.inphase_file, subject.outphase_file],
        subject.water_swap_out_file,
        override=override,
        dataset_id=282,
        gpu=gpu,
    )
    run_total_seg(
        [subject.fat_file, subject.inphase_file, subject.outphase_file],
        subject.fat_swap_out_file,
        override=override,
        dataset_id=282,
        gpu=gpu,
    )


def run_parallel_water_fat_swap_detection(subs: list[Subject], override=False, gpu=0, max_workers=16, threshold=50):
    """Runs water-fat swap detection in parallel, checking GPU memory before each task.

    Args:
        subs (list[Subject]): List of subjects to process.
        override (bool): Whether to override existing output files. Default is False.
        gpu (int): GPU index to use.
        max_workers (int): Maximum number of threads to use.
        threshold (int): GPU memory usage threshold to pause submission.
    """
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for sub in subs:
            while check_gpu_memory(gpu, threshold):
                print(f"GPU memory usage exceeded {threshold}%. Pausing submission...")
                time.sleep(10)  # Pause for 10 seconds

            futures.append(executor.submit(run_single_water_fat_swap_detection, sub, override=override, gpu=gpu))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in execution: {e}")


def run_evaluation(subs: list[Subject], excel_out_path: Path | str, segment=False, max_workers=40):
    """Evaluates all subjects and exports results to an Excel file.

    Args:
        subs (list[Subject]): List of subjects to evaluate.
        excel_out_path (Path | str): Path to save the Excel output.
        segment (bool): If True, runs the segmentation before evaluation.
        max_workers (int): Number of processes to use for evaluation.
    """
    if segment:
        run_parallel_water_fat_swap_detection(subs, max_workers=max_workers)

    imgs_list = []
    with ProcessPoolExecutor(max_workers) as executor:
        futures = {
            executor.submit(check_single_file, sub.name, sub.water_swap_out_file, sub.fat_swap_out_file, sub.total_vibe) for sub in subs
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Water shift measure"):
            result = future.result()
            if result is not None:
                imgs_list.append(result)
            else:
                print("result", None)

    data = {
        "sub": [],
        "count_water": [],
        "count_fat": [],
        "count_disagree": [],
        "percent": [],
        "percent_dis": [],
        "affected_structures": [],
    }
    for img in imgs_list:
        data["sub"].append(img.sub)
        data["count_water"].append(img.count_water)
        data["count_fat"].append(img.count_fat)
        data["count_disagree"].append(img.count_disagree)
        data["percent"].append(img.percent)
        data["percent_dis"].append(img.percent_dis)
        data["affected_structures"].append(str(img.affected_structures))

    df_inversion = pd.DataFrame(data)
    df_inversion.to_excel(excel_out_path)
    print(df_inversion["percent"].describe([0.01, 0.02, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99]))


def check_single_file(subj_name: str, seg1_water: Path, seg2_fat: Path, total_vibe: Path | str | None):
    """Identifies water-fat inversions in the segmentation, optionally using total VIBE segmentation.

    Args:
        subj_name (str): Subject identifier.
        seg1_water (Path): Water segmentation file.
        seg2_fat (Path): Fat segmentation file.
        total_vibe (Path | str | None): Optional total VIBE segmentation for affected structure analysis.

    Returns:
        IMG: An image instance with inversion counts and affected structures if applicable.
    """
    nii: NII = to_nii(seg1_water, True)
    nii2: NII = to_nii(seg2_fat, True)
    nii2.map_labels_({1: 2, 2: 1}, verbose=False)
    if nii.shape != nii2.shape:
        nii2.resample_from_to_(nii)
    nii[nii2 != nii] = 3
    nii[nii2 == 0] = 0

    if total_vibe is not None:
        total_vibe_nii: NII = to_nii(total_vibe, True)
        if nii.shape != total_vibe_nii.shape:
            total_vibe_nii.resample_from_to_(nii)
        total_vibe_nii[nii.extract_label(2).erode_msk(1, verbose=False) != 1] = 0

    return IMG(
        sub=subj_name,
        count_water=nii.extract_label(1).sum(),
        count_fat=nii.extract_label(2).sum(),
        count_disagree=nii.extract_label(3).sum(),
        affected_structures=[total_vibe_map[i] for i in total_vibe_nii.unique()] if total_vibe is not None else None,
    )


if __name__ == "__main__":
    # Load subjects: populate a list of Subject objects from a directory of files
    subject_dir = Path("/DATA/NAS/ongoing_projects/robert/code/totalvibesegmentor/data")  # TODO Update all Paths
    subs: list[Subject] = []
    for subj_path in subject_dir.iterdir():
        if not subj_path.is_dir():
            continue
        water = subj_path / "wat.nii.gz"
        fat = subj_path / "fat.nii.gz"
        inp = subj_path / "inp.nii.gz"
        opp = subj_path / "opp.nii.gz"
        sub = Subject(
            name=subj_path.name,
            inphase_file=inp,
            outphase_file=opp,
            water_file=water,
            fat_file=fat,
            water_swap_out_file=subj_path
            / "water_inversion_detection.nii.gz",  # TODO you defintly do not to put it in the same folder as you data
            fat_swap_out_file=subj_path
            / "fat_inversion_detection.nii.gz",  # TODO you defintly do not to put it in the same folder as you data
            total_vibe=None,  # TODO you may add the total_vibe path to get a list of affected structurs
        )
        subs.append(sub)

    # Perform water-fat swap detection for all subjects in parallel
    run_parallel_water_fat_swap_detection(subs)
    # If you are to greede with the threshould and max_workers, you may have to run the algorithm multiple times.
    run_parallel_water_fat_swap_detection(subs)

    # Run evaluation and save output to an Excel file. Note that a percent over 0.001 may indicate water swaps.
    run_evaluation(subs, "water-fat-swap.xlsx")  # TODO Update the xlsx Path

    """
    Water-Fat Swap Detection Report: Interpretation Guide

    This report provides insights into water-fat inversions detected in the medical images processed. The Excel file contains key metrics for each subject, indicating the likelihood and extent of such inversions. The table below explains each column in detail.
    Column Descriptions

        sub
        The unique identifier for each subject whose images were analyzed. Each row corresponds to one subject.

        count_water
        This is the total count of pixels (or voxels) identified as water within the processed image, based on the segmentation algorithm.

        count_fat
        The total count of pixels identified as fat. A significant count in both water and fat may indicate clear anatomical structures, whereas an imbalance might suggest potential swap areas.

        count_disagree
        ...
        percent
        This value represents the proportion of water swaps:
        \text{percent} = \frac{\text{count_fat}}{\text{count_fat} + \text{count_water}}
        Values over 0.001 could indicate potential water-fat swaps.

        percent_dis
        The proportion of disagreement areas in relation to the total content of water, fat, and disagreements:
        \text{percent_dis} = \frac{\text{count_disagree}}{\text{count_fat} + \text{count_water} + \text{count_disagree}}

        affected_structures
        This field lists anatomical structures from the Total VIBE segmentation map that have been affected by water-fat inversion.
        """
