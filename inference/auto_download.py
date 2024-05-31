# Copyright 2024 Hartmut Häntze
# Edited by Robert Graf 2024

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from tqdm import tqdm

logger = logging.getLogger(__name__)
WEIGHTS_URL_ = "https://github.com/robert-graf/TotalVibeSegmentator/releases/download/v1.0.0/"
env_name = "TOTALVIBE_WEIGHTS_PATH"


def get_weights_dir(idx) -> Path:
    if env_name in os.environ:
        weights_dir: Path = Path(os.environ[env_name])
    else:
        assert Path(__file__).parent.name == "inference"

        weights_dir = Path(__file__).parent.parent / "nnUNet/nnUNet_results"

    weights_dir.parent.mkdir(exist_ok=True)
    weights_dir.mkdir(exist_ok=True)
    weights_dir = weights_dir / f"Dataset{idx:03}"

    return weights_dir


def read_config(idx) -> dict["str", float]:
    weights_dir = get_weights_dir(idx)
    ds_path = weights_dir / "dataset.json"
    if ds_path.exists():
        with open(ds_path) as f:
            config_info: dict[str, float] = json.load(f)
        return config_info
    else:
        return {"dataset_release": 0.0}


def user_guard(func: Any) -> Any:
    """Check for user defined environment variables. We do NOT want to change user directories"""
    if env_name in os.environ:
        logger.info("User defined environment variables detected, skip directory operations.")
        return
    else:
        return func


@user_guard
def _download_weights(idx=85) -> None:
    weights_dir = get_weights_dir(idx)
    weights_url = WEIGHTS_URL_ + f"{idx:03}.zip"
    try:
        # Retrieve file size
        with urllib.request.urlopen(str(weights_url)) as response:
            file_size = int(response.info().get("Content-Length", -1))
    except Exception:
        print("Download attempt failed:", weights_url)
        return
    print("Downloading pretrained weights...")

    with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024, desc=Path(weights_url).name) as pbar:

        def update_progress(block_num: int, block_size: int, total_size: int) -> None:
            if pbar.total != total_size:
                pbar.total = total_size
            pbar.update(block_num * block_size - pbar.n)

        zip_path = weights_dir.parent / Path(weights_url).name
        # Download the file
        urllib.request.urlretrieve(str(weights_url), zip_path, reporthook=update_progress)

    print("Extracting pretrained weights...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(weights_dir)
    os.remove(zip_path)  # noqa: PTH107


def download_weights(idx) -> Path:
    weights_dir = get_weights_dir(idx)

    # Check if weights are downloaded
    if not weights_dir.exists():
        _download_weights(idx)
    return weights_dir
