import os  # noqa: INP001
import os.path
import random
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from get_gpu import get_free_gpus, thread_print
from joblib import Parallel, delayed
from TPTBox import BIDS_FILE, BIDS_Global_info, Log_Type, No_Logger, Subject_Container

# INPUT
in_ds = Path("/DATA/NAS/datasets_processed/NAKO/dataset-nako/")  # Path("/media/raid_sym/NAS/datasets_processed/NAKO/")
raw = ["rawdata_stitched", "rawdata"]
der = "derivatives_inversion"

blocks = list(range(100, 131))
verbose = False

head_logger = No_Logger()  # (in_ds, log_filename="source-convert-to-unet-train", default_verbose=True)

brea = False
cont_only = False
run_total_vibe = True
run_roi = True
run_body_comp = True
run_total_mevibe = True
run_paraspinal_vibe = False
run_paraspinal_mevibe = True
counts = {"VIBE": 0, "MEVIBE": 0}

blocked_gpus = {3: False, 2: False, 1: False, 0: False}
max_inf_p_gpu = 25
gpu_inf_usage = {0: 0, 1: 0, 2: 0, 3: 0}


def _check(l: list[BIDS_FILE]):
    assert len(l) == 1, l
    return l[0]


def __mevibe(subject: Subject_Container, logger: No_Logger, idx):
    q = subject.new_query(flatten=True)
    q.filter_format(["mevibe", "msk"])
    q.unflatten()
    # q.filter_format("T1w") #Test if it ends with _T1w.nii.gz
    # q.filter_format("T2w")
    q.filter_format("mevibe")
    q.filter_filetype("nii.gz")
    families = q.loop_dict(sort=True, key_addendum=["part"])
    for f in families:
        fid = f.family_id
        logger.print(f"MEVIBE Process {fid}") if verbose else None
        if "mevibe_part-eco1-pip1" not in f:
            continue
        if "mevibe_part-fat-fraction" not in f:
            continue
        if "mevibe_part-water-fraction" not in f or "mevibe_part-eco1-pip1" not in f:
            continue
        try:
            ref_ip = _check(f["mevibe_part-eco1-pip1"])
            ref_op = _check(f["mevibe_part-eco0-opp1"])
            ref_w = _check(f["mevibe_part-water-fraction"])
            ref_f = _check(f["mevibe_part-fat-fraction"])
        except Exception as e:
            logger.on_fail(f, str(e))
            continue

        out = ref_w.get_changed_path(bids_format="msk", parent=der, info={"seg": "water-fat-map", "mod": "mevibe"})
        if out.exists():
            counts["MEVIBE"] += 1
            logger.on_ok(f"Exists {out.name} {counts}") if verbose else None
        elif not cont_only:
            _process(
                idx,
                ref_ip.get("sub"),
                [ref_w.file["nii.gz"], ref_op.file["nii.gz"], ref_ip.file["nii.gz"]],
                out,
                logger,
                call=call_water_fat_map,
            )
            counts["MEVIBE"] += 1
        out = ref_f.get_changed_path(
            bids_format="msk",
            parent=der,
            info={"seg": "water-fat-map", "mod": "mevibe"},
        )
        if out.exists():
            counts["MEVIBE"] += 1
            logger.on_ok(f"Exists {out.name} {counts}") if verbose else None
        elif not cont_only:
            _process(
                idx,
                ref_ip.get("sub"),
                [ref_f.file["nii.gz"], ref_op.file["nii.gz"], ref_ip.file["nii.gz"]],
                out,
                logger,
                call=call_water_fat_map,
            )
            counts["MEVIBE"] += 1


def __inf(idx: int, name, subject: Subject_Container, override=False):
    try:
        idx = idx % 10
        logger = head_logger.add_sub_logger(name=name)
        __mevibe(subject, logger, idx)  # type: ignore
        q = subject.new_query()
        q.flatten()
        q.filter("sequ", "stitched")
        q.unflatten()
        # q.filter_format("T1w") #Test if it ends with _T1w.nii.gz
        # q.filter_format("T2w")
        q.filter_format("vibe")
        q.filter_filetype("nii.gz")
        families = list(q.loop_dict(sort=True, key_addendum=["part"]))
        for f in families:
            fid = f.family_id
            if "vibe_part-water" not in f:
                continue
            if "vibe_part-inphase" not in f:
                continue
            if "vibe_part-outphase" not in f:
                continue

            logger.print(f"Process {fid}") if verbose else None
            ref = _check(f["vibe_part-water"])
            ref_ip = _check(f["vibe_part-inphase"])
            ref_op = _check(f["vibe_part-outphase"])
            if run_body_comp:
                out = ref.get_changed_path(bids_format="msk", parent=der, info={"part": None, "seg": "water-fat-map", "mod": "vibe"})

                if out.exists() and not override:
                    counts["VIBE"] += 1
                    logger.on_ok(f"Exists {out.name} {counts}") if verbose else None
                elif not cont_only:
                    _process(
                        idx, name, [ref.file["nii.gz"], ref_op.file["nii.gz"], ref_ip.file["nii.gz"]], out, logger, call=call_water_fat_map
                    )
                    counts["VIBE"] += 1
            try:
                ref_fat = _check(f["vibe_part-fat"])
            except Exception as e:
                logger.on_fail(f, str(e))
                continue
            if run_body_comp:
                out = ref_fat.get_changed_path(
                    bids_format="msk",
                    parent=der,
                    info={"seg": "water-fat-map", "mod": "vibe"},
                )

                if out.exists() and not override:
                    counts["VIBE"] += 1
                    logger.on_ok(f"Exists {out.name} {counts}") if verbose else None
                elif not cont_only:
                    _process(
                        idx,
                        name,
                        [ref_fat.file["nii.gz"], ref_op.file["nii.gz"], ref_ip.file["nii.gz"]],
                        out,
                        logger,
                        call=call_water_fat_map,
                    )
                    counts["VIBE"] += 1

            # ref = f["vibe_part-water"][0]
    except Exception:
        logger.print_error()


def call_water_fat_map(ref: list[Path], out: Path, gpu: int, logger):
    call_TotalVibeSegmentator(ref, out, gpu, logger, ["--dataset_id", "282"], run_script="run_.py")


def call_TotalVibeSegmentator(
    ref: Path | list[Path] | list[str],
    out: Path,
    gpu: int,
    logger: No_Logger,
    addendum: Sequence[str] = (),
    run_script="run_TotalVibeSegmentator.py",
):
    gpu_inf_usage[gpu] += 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    ref = [str(ref)] if not isinstance(ref, Sequence) else [str(i) for i in ref]

    command = [
        "python",
        str(Path(__file__).parent.parent / run_script),
        "--img",
        *ref,
        "--out_path",
        str(out),
        *addendum,
        # "--gpu",str(gpu)
    ]
    logger.on_neutral(f"Command called with args: {' '.join(command)}") if verbose else None  # [2:-4]
    start_time = time.perf_counter()
    my_env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "SPINEPS_TURN_OF_CITATION_REMINDER": "TRUE",
    }
    subprocess.call(command, env=my_env)
    gpu_inf_usage[gpu] -= 1

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    logger.print(f"Inference time is: {execution_time}") if verbose else None


def _process(
    idx,
    subject,
    ref: Path | list[Path],
    out: Path,
    logger: No_Logger,
    call=call_TotalVibeSegmentator,
):
    try:
        time.sleep(idx * 1)  # start with 1 sec separation
        # thread_print(fold, "started")
        n_waited = 0
        while True:
            gpus = get_free_gpus(blocked_gpus=blocked_gpus)
            gpus = [g for g in gpus if gpu_inf_usage[g] < max_inf_p_gpu]

            if len(gpus) == 0:
                n_waited += 1
                wait_am = min(
                    int(10 * pow(n_waited, 5 / 3)),
                    60 * 2,
                )
                # thread_print(fold, f"Wait {n_waited} = {wait_am} sec", Log_Type.NEUTRAL)
                time.sleep(wait_am)
            else:
                thread_print(0, f"takes free gpu {gpus[0]}")
                call(ref, out, gpus[0], logger)
                time.sleep(random.random() * 3)
                break
    except Exception:
        logger.print_error()


# random.shuffle(blocks)
# for block in blocks:

# blocks = blocks[:1]


# block = opt.block
def filter_folder(p: Path, i: int):
    # if i == 1:
    #    try:
    #        return int(p.name) in blocks
    #    except Exception:
    #        return False
    # if i == 2:
    #    return p.name == "100451"
    if i == 3:
        return p.name in ["vibe", "mevibe"]
    return True


random.shuffle(blocks)
for b in blocks:
    parent_der = [der + "/" + str(b), *[raw + "/" + str(b) for raw in raw]]
    bids_ds = BIDS_Global_info(datasets=[in_ds], parents=parent_der, verbose=False, filter_folder=filter_folder)
    head_logger.print("Start Threading", Log_Type.LOG)
    s = list(enumerate(bids_ds.enumerate_subjects(sort=False)))
    # s = s[:4]
    random.shuffle(s)
    try:
        Parallel(n_jobs=50, backend="threading")(delayed(__inf)(idx=idx, name=name, subject=subject) for idx, (name, subject) in s)
    except Exception:
        pass
