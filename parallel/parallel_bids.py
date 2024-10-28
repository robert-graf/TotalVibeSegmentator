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
der = "derivatives_Abdominal-Segmentation"

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
counts = {
    "TotalVibeSegmentator": 0,
    "ROI": 0,
    "BodyComp": 0,
    "BodyCompME": 0,
    "TotalVibeSegmentatorME": 0,
    "paraspinalME516": 0,
    "paraspinalME517": 0,
}


def _check(l: list[BIDS_FILE]):
    assert len(l) == 1, l
    return l[0]


blocked_gpus = {3: False, 2: False, 1: False, 0: False}
max_inf_p_gpu = 12
gpu_inf_usage = {0: 0, 1: 0, 2: 0, 3: 0}


def __mevibe(subject: Subject_Container, logger: No_Logger, idx, override=False):
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

        logger.print(f"Process {fid}") if verbose else None
        for echo_key in ["mevibe_part-eco0-opp1"]:
            if echo_key not in f:
                continue
            ref = _check(f[echo_key])
            if run_total_mevibe:
                out = ref.get_changed_path(
                    bids_format="msk",
                    parent=der,
                    info={"part": None, "seg": "TotalVibeSegmentator80", "mod": "mevibe"},
                )
                if out.exists() and not override:
                    counts["TotalVibeSegmentatorME"] += 1
                    logger.on_ok(f"Exists {out.name} {counts}") if verbose else None
                elif not cont_only:
                    _process(idx, ref.get("sub"), ref.file["nii.gz"], out, logger, override=override)
                    counts["TotalVibeSegmentatorME"] += 1
            if run_paraspinal_mevibe:
                out = ref.get_changed_path(
                    bids_format="msk", parent=der, info={"part": None, "seg": "paraspinal-muscles-516", "mod": "mevibe"}
                )
                if out.exists() and not override:
                    counts["paraspinalME516"] += 1
                    logger.on_ok(f"Exists {out.name} {counts}") if verbose else None
                elif not cont_only:
                    _process(idx, ref.get("sub"), ref.file["nii.gz"], out, logger, call=call_paraspinal_muscles_516, override=override)
                    counts["paraspinalME516"] += 1
        for echo_key in ["mevibe_part-fat-fraction", "mevibe_part-water"]:
            ref = _check(f[echo_key])

            if run_paraspinal_mevibe:
                out = ref.get_changed_path(bids_format="msk", parent=der, info={"seg": "paraspinal-muscles-517", "mod": "mevibe"})
                if out.exists() and not override:
                    counts["paraspinalME517"] += 1
                    logger.on_ok(f"Exists {out.name} {counts}") if verbose else None
                elif not cont_only:
                    _process(idx, ref.get("sub"), ref.file["nii.gz"], out, logger, call=call_paraspinal_muscles_517, override=override)
                    counts["paraspinalME517"] += 1

        ip = "mevibe_part-water"
        op = "mevibe_part-eco2-opp2"
        if ip not in f:
            continue
        if op not in f:
            continue
        ref_ip = _check(f[ip])
        ref_op = _check(f[op])
        ref = ref_ip
        if run_body_comp:
            out = ref.get_changed_path(bids_format="msk", parent=der, info={"part": None, "seg": "body-composition", "mod": "mevibe"})
            if out.exists() and not override:
                counts["BodyCompME"] += 1
                logger.on_ok(f"Exists {out.name} {counts}") if verbose else None
            elif not cont_only:
                try:
                    _process(
                        idx,
                        ref.get("sub"),
                        [ref_ip.file["nii.gz"], ref_op.file["nii.gz"]],
                        out,
                        logger,
                        override=override,
                        call=call_BodyComp,
                    )
                    counts["BodyCompME"] += 1
                except KeyError:
                    logger.print_error()
                    logger.on_fail(ref_ip, ref_op)


def __inf(idx: int, name, subject: Subject_Container):
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
        families = q.loop_dict(sort=True, key_addendum=["part"])
        for f in families:
            fid = f.family_id
            if "vibe_part-water" not in f:
                continue

            logger.print(f"Process {fid}") if verbose else None
            ref = f["vibe_part-water"][0]
            if run_total_vibe:
                out = ref.get_changed_path(
                    bids_format="msk",
                    parent=der,
                    info={"part": None, "seg": "TotalVibeSegmentator", "mod": "vibe"},
                )

                if out.exists():
                    counts["TotalVibeSegmentator"] += 1
                    logger.on_ok(f"Exists {out.name} {counts}") if verbose else None
                elif not cont_only:
                    _process(idx, name, ref.file["nii.gz"], out, logger)
                    counts["TotalVibeSegmentator"] += 1
            if run_roi:
                out = ref.get_changed_path(
                    bids_format="msk",
                    parent=der,
                    info={"part": None, "seg": "ROI", "mod": "vibe"},
                )

                if out.exists():
                    counts["ROI"] += 1
                    logger.on_ok(f"Exists {out.name} {counts}") if verbose else None
                elif not cont_only:
                    _process(idx, name, ref.file["nii.gz"], out, logger, call=call_ROI)
                    counts["ROI"] += 1
            # if run_paraspinal_vibe:
            #    out = ref.get_changed_path(bids_format="msk", parent=der, info={"part": None, "seg": "paraspinal-muscles-516", "mod": "mevibe"})
            #    if out.exists():
            #        counts["paraspinal"] += 1
            #        logger.on_ok(f"Exists {out.name} {counts}")
            #    elif not cont_only:
            #        _process(idx, ref.get("sub"), ref.file["nii.gz"], out, logger, call=call_paraspinal_muscles_516)
            #        counts["paraspinal"] += 1
            if "vibe_part-inphase" not in f:
                continue
            if "vibe_part-outphase" not in f:
                continue
            ref_ip = f["vibe_part-inphase"][0]
            ref_op = f["vibe_part-outphase"][0]
            if run_body_comp:
                out = ref.get_changed_path(
                    bids_format="msk",
                    parent=der,
                    info={"part": None, "seg": "body-composition", "mod": "vibe"},
                )

                if out.exists():
                    counts["BodyComp"] += 1
                    logger.on_ok(f"Exists {out.name} {counts}") if verbose else None
                elif not cont_only:
                    _process(
                        idx,
                        name,
                        [ref_ip.file["nii.gz"], ref_op.file["nii.gz"]],
                        out,
                        logger,
                        call=call_BodyComp,
                    )
                    counts["BodyComp"] += 1

            # ref = f["vibe_part-water"][0]
    except Exception:
        logger.print_error()


def call_ROI(ref: Path, out: Path, gpu: int, logger, override=False):
    call_TotalVibeSegmentator(ref, out, gpu, logger, ["--dataset_id", "278"], override=override)


def call_paraspinal_muscles_516(ref: Path, out: Path, gpu: int, logger, override=False):
    call_TotalVibeSegmentator(ref, out, gpu, logger, ["--dataset_id", "516"], override=override)


def call_paraspinal_muscles_517(ref: Path, out: Path, gpu: int, logger, override=False):
    call_TotalVibeSegmentator(ref, out, gpu, logger, ["--dataset_id", "517"], override=override)


def call_BodyComp(ref: list[Path], out: Path, gpu: int, logger, override=False):
    call_TotalVibeSegmentator(ref, out, gpu, logger, ["--dataset_id", "281"], run_script="run_.py", override=override)


def call_TotalVibeSegmentator(
    ref: Path | list[Path] | list[str],
    out: Path,
    gpu: int,
    logger: No_Logger,
    addendum: Sequence[str] = (),
    run_script="run_TotalVibeSegmentator.py",
    override=False,
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
    if override:
        command.append("--override")
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


def _process(idx, subject, ref: Path | list[Path], out: Path, logger, call=call_TotalVibeSegmentator, override=False):
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
            call(ref, out, gpus[0], logger, override=override)
            time.sleep(random.random() * 3)
            break


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
        Parallel(n_jobs=40, backend="threading")(delayed(__inf)(idx=idx, name=name, subject=subject) for idx, (name, subject) in s)
    except Exception:
        pass
