import time

from inference.parallel_bids import Arguments, main

# --dataset: Path = Path("~/dataset-my-dataset")
# --endswith: str | None = None  # only files that mach *[ending].nii* will be segmented
# --out_folder: str | None = None  # like "derivative" to make a derivative folder, else use the same folder
# --override: bool = False
# --gpu: list[int] | None = None
# --dataset_id: list[int] | None = [80]
# --max_inf_p_gpu: int = 6
# --verbose: bool = False
# --n_jobs: int = 20

if __name__ == "__main__":
    t = time.time()
    arg = Arguments.get_opt()
    if not arg.dataset.exists():
        raise FileNotFoundError(arg.dataset)
    main(arg)
    print(f"Took {time.time()-t} seconds.")

    # python run_TotalVibeSegmentator_parallel.py --dataset /media/data/robert/datasets/dataset-ammos22/amos22/ --out_folder derivative --dataset_id 511 512
