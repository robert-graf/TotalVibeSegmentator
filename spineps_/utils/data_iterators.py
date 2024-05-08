# Adapted from https://github.com/MIC-DKFZ/nnUNet
# Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring
# method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

from typing import List, Union

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader

from spineps_.utils.default_preprocessor import DefaultPreprocessor
from spineps_.utils.plans_handler import ConfigurationManager, PlansManager


class PreprocessAdapterFromNpy(DataLoader):
    def __init__(
        self,
        list_of_images: list[np.ndarray],
        list_of_segs_from_prev_stage: list[np.ndarray] | None,
        list_of_image_properties: list[dict],
        truncated_ofnames: list[str] | None,
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_threads_in_multithreaded: int = 1,
        verbose: bool = False,
    ):
        preprocessor = DefaultPreprocessor(verbose=verbose)
        self.preprocessor, self.plans_manager, self.configuration_manager, self.dataset_json, self.truncated_ofnames = (
            preprocessor,
            plans_manager,
            configuration_manager,
            dataset_json,
            truncated_ofnames,
        )

        self.label_manager = plans_manager.get_label_manager(dataset_json)

        if list_of_segs_from_prev_stage is None:
            list_of_segs_from_prev_stage = [None] * len(list_of_images)
        if truncated_ofnames is None:
            truncated_ofnames = [None] * len(list_of_images)

        super().__init__(
            list(zip(list_of_images, list_of_segs_from_prev_stage, list_of_image_properties, truncated_ofnames, strict=False)),
            1,
            num_threads_in_multithreaded,
            seed_for_shuffle=1,
            return_incomplete=True,
            shuffle=False,
            infinite=False,
            sampling_probabilities=None,
        )

        self.indices = list(range(len(list_of_images)))

    def generate_train_batch(self):
        idx = self.get_indices()[0]
        image = self._data[idx][0]
        seg_prev_stage = self._data[idx][1]
        props = self._data[idx][2]
        ofname = self._data[idx][3]
        # if we have a segmentation from the previous stage we have to process it together with the images so that we
        # can crop it appropriately (if needed). Otherwise it would just be resized to the shape of the data after
        # preprocessing and then there might be misalignments
        data, seg = self.preprocessor.run_case_npy(
            image, seg_prev_stage, props, self.plans_manager, self.configuration_manager, self.dataset_json
        )
        if seg_prev_stage is not None:
            seg_onehot = convert_labelmap_to_one_hot(seg[0], self.label_manager.foreground_labels, data.dtype)
            data = np.vstack((data, seg_onehot))

        data = torch.from_numpy(data)

        return {"data": data, "data_properites": props, "ofile": ofname}


def convert_labelmap_to_one_hot(
    segmentation: np.ndarray | torch.Tensor, all_labels: list | torch.Tensor | np.ndarray | tuple, output_dtype=None
) -> np.ndarray | torch.Tensor:
    """
    if output_dtype is None then we use np.uint8/torch.uint8
    if input is torch.Tensor then output will be on the same device

    np.ndarray is faster than torch.Tensor

    if segmentation is torch.Tensor, this function will be faster if it is LongTensor. If it is somethine else we have
    to cast which takes time.

    IMPORTANT: This function only works properly if your labels are consecutive integers, so something like 0, 1, 2, 3, ...
    DO NOT use it with 0, 32, 123, 255, ... or whatever (fix your labels, yo)
    """
    if isinstance(segmentation, torch.Tensor):
        result = torch.zeros(
            (len(all_labels), *segmentation.shape),
            dtype=output_dtype if output_dtype is not None else torch.uint8,
            device=segmentation.device,
        )
        # variant 1, 2x faster than 2
        result.scatter_(0, segmentation[None].long(), 1)  # why does this have to be long!?
        # variant 2, slower than 1
        # for i, l in enumerate(all_labels):
        #     result[i] = segmentation == l
    else:
        result = np.zeros((len(all_labels), *segmentation.shape), dtype=output_dtype if output_dtype is not None else np.uint8)
        # variant 1, fastest in my testing
        for i, l in enumerate(all_labels):
            result[i] = segmentation == l
        # variant 2. Takes about twice as long so nah
        # result = np.eye(len(all_labels))[segmentation].transpose((3, 0, 1, 2))
    return result
