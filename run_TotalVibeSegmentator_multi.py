import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from TPTBox import NII, Log_Type, Print_Logger, to_nii, to_nii_seg

from TypeSaveArgParse import Class_to_ArgParse

sys.path.append(str(Path(__file__).parent))
from inference.auto_download import download_weights  # noqa: I001
from inference.inference_nnunet import get_ds_info
from inference.inference_nnunet import p as model_path
from inference.inference_nnunet import run_inference_on_file

logger = Print_Logger()
idx_models = [86, 85]  # first found is used

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
    63: {"typ": "bone", "name": "sternum", "min": 1, "max": 1, "autofix": 15},
    64: {"typ": "prone_to_over_seg", "name": "costal_cartilages", "min": 4, "max": 30, "autofix": 2},
    65: {"typ": "rest", "name": "subcutaneous_fat", "min": 1, "max": 1000, "autofix": 2},
    66: {"typ": "rest", "name": "muscle", "min": 1, "max": 1000, "autofix": 2},
    67: {"typ": "rest", "name": "inner_fat", "min": 1, "max": 1000, "autofix": 2},
    68: {"typ": "bone", "name": "IVD", "min": 1, "max": 25, "autofix": 2},
    69: {"typ": "bone", "name": "vertebra_body", "min": 1, "max": 25, "autofix": 2},
    70: {"typ": "prone_to_over_seg", "name": "vertebra_posterior_elements", "min": 1, "max": 25},
    71: {"typ": "cns", "name": "spinal_channel", "min": 1, "max": 1, "autofix": 5},
    72: {"typ": "bone", "name": "bone_other", "min": 0, "max": 10, "autofix": 50},
}


def run_roi(nii: str | Path, out_file: Path | str | None, gpu=None, dataset_id=278, keep_size=False, override=False):
    try:
        download_weights(dataset_id)
        next(next(iter(model_path.glob(f"*{dataset_id}*"))).glob("*__nnUNetPlans*"))
    except StopIteration:
        raise FileNotFoundError(
            f"Could not find roi-model. Download the model {dataset_id} an put it into {model_path.absolute()}"
        ) from None
    nii_seg, _ = run_inference_on_file(
        dataset_id, [to_nii(nii, False)], gpu=gpu, out_file=out_file, keep_size=keep_size, override=override, logits=False
    )
    return nii_seg


def run_total_seg(
    img_inphase: Path | str,
    img_water: Path | str,
    img_outphase: Path | str,
    out_path: Path,
    override=False,
    dataset_id=None,
    gpu: int | None = None,
    logits=False,
    known_idx=idx_models,
    roi_path: str | Path | None = None,
    keep_size=False,
    fill_holes=False,
    **kargs,
):
    if dataset_id is None:
        for idx in known_idx:
            download_weights(idx)
            try:
                next(next(iter(model_path.glob(f"*{idx}*"))).glob("*__nnUNetPlans*"))
                dataset_id = idx
                break
            except StopIteration:
                pass
        else:
            logger.print(f"Could not find model. Download the model an put it into {model_path.absolute()}", Log_Type.FAIL)
            return
    if out_path.exists() and not override:
        logger.print(out_path, "already exists. SKIP!", Log_Type.OK)
        return out_path
    inter_file = {}
    for name, path in [("water", img_water), ("inphase", img_inphase), ("outphase", img_outphase)]:
        if path is None:
            continue
        if not Path(path).exists():
            raise FileNotFoundError(path)
        n = out_path.name.split(".")[0].rsplit("_", maxsplit=1)
        n2 = "" if len(n) == 1 else n[1]
        out_path_individual = out_path.parent / (n[0] + f"_part-{name}_" + n2 + ".nii.gz")
        inter_file[name] = (Path(path), out_path_individual)
    if len(inter_file) == 0:
        raise ValueError("Must set at least on of: --img_inphase --img_water --img_outphase")
    selected_gpu = gpu
    if gpu is None:
        gpu = "auto"  # type: ignore
    logger.print("run", f"{dataset_id=}, {gpu=}", Log_Type.STAGE)
    try:
        ds_info = get_ds_info(dataset_id)
        orientation = ds_info["orientation"]
        if "roi" in ds_info:
            roi_seg = run_roi(
                next(iter(inter_file.values()))[0], roi_path, gpu=selected_gpu, dataset_id=ds_info.get("roi", 278), override=override
            )
            roi_seg = [to_nii_seg(roi_seg)]

        else:
            roi_seg = []

        for img_, out_ in inter_file.values():
            a = run_inference_on_file(
                dataset_id,
                [to_nii(img_), *roi_seg],
                override=override,
                out_file=out_,
                gpu=selected_gpu,
                orientation=orientation,
                logits=logits,
                keep_size=keep_size,
            )
            validate_seg(to_nii_seg(out_) if a is None else to_nii_seg(a[0]), out_, aggressiveness=5)
        combine(inter_file, out_path=Path(out_path), override=override, fill_holes=fill_holes)
    except Exception:
        logger.print_error()


def validate_seg(nii: NII | Path, path_seg: Path, save_prob=False, aggressiveness=1, verbose=True, fill_holes=False):
    nii = to_nii_seg(nii)
    path = path_seg.parent
    logger = Print_Logger()
    arrs, dicts = nii.get_segmentation_connected_components(nii.unique())
    for idx, n_cc in dicts.items():
        problem = False
        target = labels[idx]
        if n_cc > target["max"]:
            logger.print(f"{target['name']}: got {n_cc} cc, but we expected less than {target['max']}", verbose=verbose)
            problem = True
        if n_cc < target["min"]:
            logger.print(f"{target['name']}: got {n_cc} cc, but we expected more than {target['min']}", verbose=verbose)
            problem = True
        if problem:
            arr = arrs[idx]
            if save_prob:
                out = nii.set_array(arr)
                out = out + out.dilate_msk(2) * 100
                out.save(path / "problematic" / (target["name"] + ".nii.gz"))
            for i in np.unique(arr):
                if i == 0:
                    continue
                arr2 = arr.copy()
                arr2[arr2 != i] = 0
                arr2[arr2 == i] = 1
                vol = arr2.sum()
                forb = ""
                if "autofix" in target:
                    max_cc_vol = int(target["autofix"]) * aggressiveness
                    if vol <= max_cc_vol:
                        forb += " --> autofix"
                        nii = nii * nii.set_array(1 - arr2)
                    else:
                        forb += " --> unchanged"
                logger.print(f"CC id={i}; volumen={vol:10},\t {forb}", verbose=verbose)
    if fill_holes:
        nii.fill_holes_()
    nii.save(path_seg)
    logger.close()


def combine(inter_file: dict[str, tuple[Path, Path]], out_path: Path, override=False, verbose=True, female=False, fill_holes=False):
    if out_path.exists() and not override:
        return

    if "water" in inter_file:
        dict_source_from_file = {
            "vessel": None,
            "lung": None,
            "cns": None,
            "prone_to_over_seg": ["water"],
            "bone": None,
            "digenstion": None,
            "muscle": ["water"],
            "organ": None,
            "rest": None,
        }
    else:
        dict_source_from_file = {
            "vessel": None,
            "lung": None,
            "cns": None,
            "prone_to_over_seg": None,
            "bone": None,
            "digenstion": None,
            "muscle": None,
            "organ": None,
            "rest": None,
        }
    ## Types
    types = {}
    for i, d in labels.items():
        t = d["typ"]
        if t not in types:
            types[t] = []
        types[t].append(i)
    ### Resample
    first_key = next(iter(inter_file.keys()))
    segs = {first_key: to_nii_seg(inter_file[first_key][1])}
    #### MASKING
    # mask = roi.clamp(0, 1).dilate_msk(3)
    for name, (_, p2) in inter_file.items():
        if name == first_key:
            continue
        nii = NII.load(p2, True)
        if segs[first_key].shape != nii.shape:
            nii.resample_from_to_(segs[first_key])
        segs[name] = nii

    out = segs[first_key].copy() * 0
    arr = out.get_array()

    for name, img_names in dict_source_from_file.items():
        imgs = list(segs.values()) if img_names is None else [segs[n] for n in img_names]
        for x in imgs:
            for idx in types[name]:
                if female and idx == 22:  # Prostate
                    continue
                label = x.extract_label(idx)
                if labels[idx]["max"] == 1 and name != "vessel":
                    label = label.get_largest_k_segmentation_connected_components(1)
                label = label.get_array()
                arr[arr == 0] = label[arr == 0] * idx
    validate_seg(out.set_array(arr), out_path, aggressiveness=15, verbose=verbose, fill_holes=fill_holes)


@dataclass
class Arguments(Class_to_ArgParse):
    img_inphase: Path | None = None
    img_water: Path | None = None
    img_outphase: Path | None = None
    out_path: Path = Path("seg_combined.nii.gz")
    roi_path: Path | None = None
    override: bool = False
    gpu: int | None = None
    dataset_id: int | None = None
    keep_size: bool = False
    fill_holes: bool = False


if __name__ == "__main__":
    import time

    t = time.time()
    arg = Arguments.get_opt()
    run_total_seg(**arg.__dict__)
    print(f"Took {time.time()-t} seconds.")
# --img_inphase inphase.nii.gz --img_water water.nii.gz --img_outphase outphase.nii.gz
