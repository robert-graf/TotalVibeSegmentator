import sys
from pathlib import Path

from TPTBox import Location, Print_Logger, v_name2idx

sys.path.append(str(Path(__file__).parent))
from run_instance_spine_segmentation import Arguments, run_seg

logger = Print_Logger()
instance_models = [512]


if __name__ == "__main__":
    import time

    t = time.time()
    arg = Arguments.get_opt()
    if not arg.img.exists():
        raise FileNotFoundError(arg.img)
    try:
        mapping = {
            1: Location.Arcus_Vertebrae.value,
            2: Location.Spinosus_Process.value,
            3: Location.Costal_Process_Left.value,
            4: Location.Costal_Process_Right.value,
            5: Location.Superior_Articular_Left.value,
            6: Location.Superior_Articular_Right.value,
            7: Location.Inferior_Articular_Left.value,
            8: Location.Inferior_Articular_Right.value,
            9: Location.Vertebra_Corpus_border.value,  # 49 and 50 are the same in MRI segmentation
            13: Location.Vertebra_Disc.value,
            14: Location.Spinal_Cord.value,
            15: Location.Spinal_Canal.value,
            16: v_name2idx["S1"],
        }
    except Exception:
        mapping = None
    run_seg(**arg.__dict__, known_idx=instance_models, mapping=mapping)
    print(f"Took {time.time()-t} seconds.")
