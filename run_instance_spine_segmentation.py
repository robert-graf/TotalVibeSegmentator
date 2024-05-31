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
instance_models = [511]  # first found is used


def run_seg(img: Path | str, out_path: Path, override=False, dataset_id=None, gpu=None, logits=False, known_idx=instance_models, **kargs):
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

    logger.print("run", dataset_id, gpu, kargs, Log_Type.STAGE)
    try:
        ds_info = get_ds_info(dataset_id)
        orientation = ds_info["orientation"]
        return run_inference_on_file(
            dataset_id, [to_nii(img)], override=override, out_file=out_path, gpu=gpu, orientation=orientation, logits=logits
        )
    except Exception:
        logger.print_error()


@dataclass
class Arguments(Class_to_ArgParse):
    img: Path = Path("img.nii.gz")
    out_path: Path = Path("seg.nii.gz")
    override: bool = False
    gpu = None
    dataset_id: int | None = None


if __name__ == "__main__":
    import time

    t = time.time()
    arg = Arguments.get_opt()
    if not arg.img.exists():
        raise FileNotFoundError(arg.img)
    run_seg(**arg.__dict__)
    print(f"Took {time.time()-t} seconds.")
