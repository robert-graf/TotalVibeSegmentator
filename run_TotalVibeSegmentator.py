import sys
from dataclasses import dataclass
from pathlib import Path

from TPTBox import Log_Type, Print_Logger, to_nii

from TypeSaveArgParse import Class_to_ArgParse

sys.path.append(str(Path(__file__).parent))
from inference.auto_download import download_weights  # noqa: I001
from inference.inference_nnunet import get_ds_info
from inference.inference_nnunet import p as model_path
from inference.inference_nnunet import run_inference_on_file

logger = Print_Logger()
idx_models = [80, 87, 86, 85]  # first found is used; Recommended is the earlier ones.


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
    img: Path | str,
    out_path: Path,
    override=False,
    dataset_id=None,
    gpu: int | None = None,
    logits=False,
    known_idx=idx_models,
    roi_path: str | Path | None = None,
    keep_size=False,
    fill_holes=False,
    crop=False,
    max_folds: int | None = None,
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
    else:
        download_weights(dataset_id)
    if out_path.exists() and not override:
        logger.print(out_path, "already exists. SKIP!", Log_Type.OK)
        return out_path
    selected_gpu = gpu
    if gpu is None:
        gpu = "auto"  # type: ignore
    logger.print("run", f"{dataset_id=}, {gpu=}", Log_Type.STAGE)
    try:
        ds_info = get_ds_info(dataset_id)
        orientation = ds_info["orientation"]
        if "roi" in ds_info:
            roi_seg = run_roi(img, roi_path, gpu=selected_gpu, dataset_id=ds_info.get("roi", 278), override=override)
            roi_seg = to_nii(roi_seg, True)
            in_niis = [to_nii(img), roi_seg]
        else:
            in_niis = [to_nii(img)]
        if (in_niis[0].affine == np.eye(4)).all():
            from warnings import warn

            warn(
                "Your affine matrix is the identity. Make sure that the spacing and orientation is correct. For NAKO it should be 1.40625 mm for R/L and A/P and 3 mm S/I. For UKBB R/L and A/P should be around 2.2 mm",
                stacklevel=3,
            )
        return run_inference_on_file(
            dataset_id,
            in_niis,
            override=override,
            out_file=out_path,
            gpu=selected_gpu,
            orientation=orientation,
            logits=logits,
            keep_size=keep_size,
            fill_holes=fill_holes,
            crop=crop,
            max_folds=max_folds,
        )
    except Exception:
        logger.print_error()


@dataclass
class Arguments(Class_to_ArgParse):
    img: Path = Path("img.nii.gz")
    out_path: Path = Path("seg.nii.gz")
    roi_path: Path | None = None
    override: bool = False
    gpu = None
    dataset_id: int | None = None
    keep_size: bool = False
    fill_holes: bool = False
    crop: bool = False
    max_folds: int | None = None


if __name__ == "__main__":
    import time

    t = time.time()
    arg = Arguments.get_opt()
    if not arg.img.exists():
        raise FileNotFoundError(arg.img)
    import numpy as np
    from TPTBox import NII

    # nii = NII.load(arg.img, False)
    # arr = nii.nii.get_fdata()
    # nii.set_array_(arr)
    # arg.img = nii  # type: ignore
    run_total_seg(**arg.__dict__)
    print(f"Took {time.time()-t} seconds.")
