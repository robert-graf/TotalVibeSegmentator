import sys
from dataclasses import dataclass
from pathlib import Path

from TPTBox import Location, Log_Type, Print_Logger, to_nii, v_name2idx

from TypeSaveArgParse import Class_to_ArgParse

sys.path.append(str(Path(__file__).parent))
from inference.auto_download import download_weights  # noqa: I001
from inference.inference_nnunet import get_ds_info
from inference.inference_nnunet import p as model_path
from inference.inference_nnunet import run_inference_on_file

logger = Print_Logger()
instance_models = [511]  # first found is used


def run_seg(
    img: Path | str,
    out_path: Path,
    override=False,
    dataset_id=None,
    gpu=None,
    logits=False,
    known_idx=instance_models,
    mapping=None,
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
            logger.print(
                f"Could not find model. Download the model an put it into {model_path.absolute()}; Known idx {known_idx}", Log_Type.FAIL
            )
            return
    if out_path.exists() and not override:
        logger.print(out_path, "already exists. SKIP!", Log_Type.OK)
        return
    selected_gpu = gpu
    if gpu is None:
        gpu = "auto"
    logger.print("run", f"{dataset_id=}, {gpu=}", Log_Type.STAGE)
    try:
        ds_info = get_ds_info(dataset_id)
        orientation = ds_info["orientation"]
        return run_inference_on_file(
            dataset_id,
            [to_nii(img)],
            override=override,
            out_file=out_path,
            gpu=selected_gpu,
            orientation=orientation,
            logits=logits,
            mapping=mapping,
        )
    except Exception:
        logger.print_error()


@dataclass
class Arguments(Class_to_ArgParse):
    img: Path = Path("img.nii.gz")
    out_path: Path = Path("seg.nii.gz")
    override: bool = False
    gpu: int | None = None
    dataset_id: int | None = None


if __name__ == "__main__":
    import time

    t = time.time()
    arg = Arguments.get_opt()
    if not arg.img.exists():
        raise FileNotFoundError(arg.img)
    try:
        mapping = {27: Location.Vertebra_Disc.value, 16: v_name2idx["S1"]}
    except Exception:
        mapping = None
    run_seg(**arg.__dict__, mapping=mapping)
    print(f"Took {time.time()-t} seconds.")
