import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from TPTBox import NII, Image_Reference, Log_Type, Print_Logger

sys.path.append(str(Path(__file__).parent.parent))
from inference.auto_download import download_weights

idx = 70

out_base = Path(__file__).parent.parent / "nnUNet/"
p = out_base / "nnUNet_results"


def get_ds_info(idx) -> dict:
    try:
        nnunet_path = next(next(iter(p.glob(f"*{idx}*"))).glob("*__nnUNetPlans*"))
    except StopIteration:
        Print_Logger().print(f"Please add Dataset {idx} to {p}", Log_Type.FAIL)
        p.mkdir(exist_ok=True, parents=True)
        exit()
    with open(Path(nnunet_path, "dataset.json")) as f:
        ds_info = json.load(f)
    return ds_info


def squash_so_it_fits_in_float16(x: NII):
    m = x.max()
    if m > 10000:
        x /= m / 1000  # new max will be 1000
    return x


def run_inference_on_file(
    idx,
    input_nii: list[NII],
    out_file: str | Path | None = None,
    orientation=None,
    override=False,
    gpu=None,
    keep_size=False,
    fill_holes=False,
    logits=False,
    mapping=None,
    crop=False,
    max_folds=None,
) -> tuple[Image_Reference, np.ndarray | None]:
    if out_file is not None and Path(out_file).exists() and not override:
        return out_file, None

    from spineps_.utils.inference_api import load_inf_model, run_inference

    download_weights(idx)

    nnunet_path = next(next(iter(p.glob(f"*{idx}*"))).glob("*__nnUNetPlans*"))
    folds = [int(f.name.split("fold_")[-1]) for f in nnunet_path.glob("fold*")]
    if max_folds is not None:
        folds = folds[:max_folds]

    # if idx in _unets:
    #    nnunet = _unets[idx]
    # else:
    nnunet = load_inf_model(nnunet_path, allow_non_final=True, use_folds=tuple(folds) if len(folds) != 5 else None, gpu=gpu)
    #    _unets[idx] = nnunet
    with open(Path(nnunet_path, "plans.json")) as f:
        plans_info = json.load(f)
    with open(Path(nnunet_path, "dataset.json")) as f:
        ds_info = json.load(f)
    if "orientation" in ds_info:
        orientation = ds_info["orientation"]
    zoom = None
    og_nii = input_nii[0].copy()

    try:
        zoom = plans_info["configurations"]["3d_fullres"]["spacing"][::-1]
    except Exception:
        pass
    assert len(ds_info["channel_names"]) == len(input_nii), (ds_info["channel_names"], len(input_nii), "\n", nnunet_path)
    if orientation is not None:
        input_nii = [i.reorient(orientation) for i in input_nii]

    if zoom is not None:
        input_nii = [i.rescale_(zoom) for i in input_nii]
    input_nii = [squash_so_it_fits_in_float16(i) for i in input_nii]
    if crop:
        crop = input_nii[0].compute_crop(minimum=20)
        input_nii = [i.apply_crop(crop) for i in input_nii]
    seg_nii, uncertainty_nii, softmax_logits = run_inference(input_nii, nnunet, logits=logits)
    if mapping is not None:
        seg_nii.map_labels_(mapping)
    if not keep_size:
        seg_nii.resample_from_to_(og_nii)
    if fill_holes:
        seg_nii.fill_holes_()
    if out_file is not None and (not Path(out_file).exists() or override):
        seg_nii.save(out_file)
    del nnunet

    torch.cuda.empty_cache()
    return seg_nii, softmax_logits
