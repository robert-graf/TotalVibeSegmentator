import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from run_TotalVibeSegmentator import run_total_seg
from TypeSaveArgParse import Class_to_ArgParse


@dataclass
class Arguments(Class_to_ArgParse):
    img: list[Path] | None = None
    out_path: Path = Path("seg.nii.gz")
    roi_path: Path | None = None
    override: bool = False
    gpu: int | None = None
    dataset_id: int | None = None
    keep_size: bool = False
    fill_holes: bool = False
    crop: bool = False
    max_folds: int | None = None


if __name__ == "__main__":
    # This file can call all types of nnunets, even if they require multiple outputs
    import time

    t = time.time()
    arg = Arguments.get_opt()
    assert arg.img is not None, "You must set an image --img path1 path2"
    for img in arg.img:
        if not img.exists():
            raise FileNotFoundError(img)
    print(f"Took {time.time()-t} seconds.")
    run_total_seg(**arg.__dict__)
