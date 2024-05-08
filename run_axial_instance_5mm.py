import sys
from dataclasses import dataclass
from pathlib import Path

from TPTBox import Log_Type, Print_Logger, to_nii

from TypeSaveArgParse import Class_to_ArgParse

sys.path.append(str(Path(__file__).parent))
from inference.inference_nnunet import get_ds_info, run_inference_on_file

logger = Print_Logger()


def run_510(
    img: Path | str,
    out_path: Path,
    override=False,
    _idx=510,
    gpu=None,
    logits=False,
    **kargs,
):
    logger.print("run", _idx, gpu, kargs, Log_Type.STAGE)
    try:
        ds_info = get_ds_info(_idx)
        orientation = ds_info["orientation"]
        return run_inference_on_file(
            _idx, [to_nii(img)], override=override, out_file=out_path, gpu=gpu, orientation=orientation, logits=logits
        )
    except Exception:
        logger.print_error()


@dataclass
class Arguments(Class_to_ArgParse):
    img: Path = Path("img.nii.gz")
    out_path: Path = Path("seg.nii.gz")
    override: bool = True
    gpu = None


if __name__ == "__main__":
    import time

    t = time.time()
    arg = Arguments.get_opt()
    if not arg.img.exists():
        raise FileNotFoundError(arg.img)
    run_510(**arg.__dict__)
    print(f"Took {time.time()-t} seconds.")
