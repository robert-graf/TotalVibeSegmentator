import sys
from pathlib import Path

from TPTBox import Print_Logger

sys.path.append(str(Path(__file__).parent))
from run_instance_spine_segmentation import Arguments, run_seg

logger = Print_Logger()
instance_models = [518, 516, 514, 512]


if __name__ == "__main__":
    import time

    t = time.time()
    arg = Arguments.get_opt()
    if not arg.img.exists():
        raise FileNotFoundError(arg.img)
    run_seg(**arg.__dict__, known_idx=instance_models)
    print(f"Took {time.time()-t} seconds.")
