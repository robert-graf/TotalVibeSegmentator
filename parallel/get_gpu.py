import time  # noqa: INP001

import GPUtil
from TPTBox import Log_Type, No_Logger

logger = No_Logger()


def get_gpu(verbose: bool = False, max_load: float = 0.3, max_memory: float = 0.4):
    GPUtil.showUtilization() if verbose else None
    device_ids = GPUtil.getAvailable(
        order="load", limit=4, maxLoad=max_load, maxMemory=max_memory, includeNan=False, excludeID=[], excludeUUID=[]
    )
    return device_ids


def intersection(lst1, lst2):
    return set(lst1).intersection(lst2)


def get_free_gpus(blocked_gpus=None, max_load: float = 0.3, max_memory: float = 0.4):
    # print("get_free_gpus")
    if blocked_gpus is None:
        blocked_gpus = {0: False, 1: False, 2: False, 3: False}
    cached_list = get_gpu(max_load=max_load, max_memory=max_memory)
    for _ in range(15):
        time.sleep(0.25)
        cached_list = intersection(cached_list, get_gpu())
    # print("result:", list(cached_list))
    gpulist = [i for i in list(cached_list) if i not in blocked_gpus or blocked_gpus[i] is False]
    # print("result:", gpulist)
    return gpulist


def thread_print(fold, *text):
    logger.print(f"Fold [{fold}]: ", *text)
