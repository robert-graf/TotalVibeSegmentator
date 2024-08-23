import os
import os.path
import random
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
from get_gpu import get_all_gpus, get_free_gpus, scan_tree, thread_print
from joblib import Parallel, delayed
from TPTBox import BIDS_FILE, BIDS_Global_info, Log_Type, No_Logger, Subject_Container

from TypeSaveArgParse.autoargs import Class_to_ArgParse

supported_file_endings = ["nii.gz", "nii", "mha"]


@dataclass
class Arguments(Class_to_ArgParse):
    dataset: Path = Path("~/dataset-my-dataset")
    endswith: str | None = None  # only files that mach *[ending].nii* will be segmented
    out_folder: str | None = None  # like "derivative" to make a derivative folder, else use the same folder
    override: bool = False
    gpu: list[int] | None = None
    dataset_id: list[int] | None = None
    max_inf_p_gpu: int = 6
    verbose: bool = False
    n_jobs: int = 20

    @property
    def blocked_gpus(self):
        if hasattr(self, "_blocked_gpus"):
            return self._blocked_gpus  # type: ignore
        elif self.gpu is None:
            self._blocked_gpus = {i.id: False for i in get_all_gpus()}
        else:
            self._blocked_gpus = {i.id: i.id not in self.gpu for i in get_all_gpus()}
        return self._blocked_gpus

    gpu_inf_usage = {0: 0, 1: 0, 2: 0, 3: 0}  # noqa: RUF012


# INPUT
head_logger = No_Logger()  # (in_ds, log_filename="source-convert-to-unet-train", default_verbose=True)


def __inf(idx: int, path: Path, args: Arguments):
    try:
        idx = idx % 10
        path = Path(path)
        logger = head_logger.add_sub_logger(name=(path).name)
        if not any(str(path).endswith(i) for i in supported_file_endings):
            logger.on_neutral("Not supported file type", path)
            return
        if args.endswith is not None and not path.name.split(".")[0].endswith(args.endswith):
            logger.on_neutral(f"Not endswith {args.endswith}", path)
            return
        # if "sub" not in path.name:
        #    in_file = BIDS_FILE(Path(path.parent, "sub-" + path.name.replace("_", "-")), args.dataset, verbose=False)
        # else:
        in_file = BIDS_FILE(path, args.dataset, verbose=False)

        if args.dataset_id is None:
            args.dataset_id = [""]  # type: ignore
        for dataset_id in args.dataset_id:  # type: ignore
            out = in_file.get_changed_path(
                bids_format="msk",
                parent=args.out_folder if args.out_folder is not None else in_file.parent,
                info={"sub": in_file.get(in_file, path.name.split(".")[0].replace("_", "-")), "seg": f"TotalVibeSegmentator{dataset_id}"},
            )
            if out.exists() and not args.override:
                logger.on_ok(f"Exists {out.name}") if args.verbose else None
            else:
                call = partial(call_TotalVibeSegmentator, dataset_id=dataset_id if dataset_id != "" else None)
                _process(idx, args, path, out, logger, call=call, override=args.override)

    except Exception:
        head_logger.print_error()


run_scripts = {511: "run_instance_spine_segmentation.py", 512: "run_semantic_spine_segmentation.py"}


def call_TotalVibeSegmentator(
    ref: Path | list[Path] | list[str],
    out: Path,
    args: Arguments,
    gpu: int,
    logger: No_Logger,
    dataset_id: int | None = None,
    run_script="run_TotalVibeSegmentator.py",
    override=False,
):
    if dataset_id in run_scripts:
        run_script = run_scripts[dataset_id]
    args.gpu_inf_usage[gpu] += 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    ref = [str(ref)] if not isinstance(ref, Sequence) else [str(i) for i in ref]
    if dataset_id is None:
        addendum: Sequence[str] = ()
    else:
        addendum = ("--dataset_id", str(dataset_id))
    command = [
        "python",
        str(Path(__file__).parent.parent / run_script),
        "--img",
        *ref,
        "--out_path",
        str(out),
        *addendum,
        "--gpu",
        str(gpu),
    ]
    if override:
        command.append("--override")
    logger.on_neutral(f"Command called with args: {' '.join(command)}") if args.verbose else None  # [2:-4]
    start_time = time.perf_counter()
    my_env = {**os.environ}  # "CUDA_VISIBLE_DEVICES": str(gpu),
    subprocess.call(command, env=my_env)
    args.gpu_inf_usage[gpu] -= 1

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    logger.print(f"Inference time is: {execution_time}") if args.verbose else None


def _process(idx, args: Arguments, ref: Path | list[Path], out: Path, logger, call=call_TotalVibeSegmentator, override=False):
    time.sleep(idx * 1)  # start with 1 sec separation
    # thread_print(fold, "started")
    n_waited = 0
    while True:
        gpus = get_free_gpus(blocked_gpus=args.blocked_gpus)
        gpus = [g for g in gpus if args.gpu_inf_usage[g] < args.max_inf_p_gpu]

        if len(gpus) == 0:
            wait_am = min(int(5 * pow(n_waited, 5 / 3)), 60 * 2)
            n_waited += 1
            thread_print(idx, f"Wait {n_waited} = {wait_am} sec", Log_Type.NEUTRAL)
            time.sleep(wait_am)
        else:
            thread_print(0, f"takes free gpu {gpus[0]}")
            call(ref, out, args, gpus[0], logger, override=override)
            time.sleep(random.random() * 3)
            break


def main(args: Arguments):
    def filter_fun(path: str):
        name = path.rsplit("/", maxsplit=1)[-1]

        if any(x in name for x in ["_msk.", "_seg"]):
            return False
        return any(name.endswith(x) for x in supported_file_endings)

    s = [x for x in scan_tree(args.dataset) if filter_fun(x)]

    head_logger.print(f"Start Threading, files_found = {len(s)}", Log_Type.LOG)
    # s = s[:4]
    random.shuffle(s)
    try:
        Parallel(n_jobs=args.n_jobs, backend="threading")(delayed(__inf)(idx=idx, path=path, args=args) for idx, path in enumerate(s))
    except Exception:
        head_logger.print_error()


if __name__ == "__main__":
    t = time.time()
    arg = Arguments.get_opt()
    if not arg.dataset.exists():
        raise FileNotFoundError(arg.dataset)
    main(arg)
    print(f"Took {time.time()-t} seconds.")
