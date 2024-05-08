# Adapted from https://github.com/MIC-DKFZ/nnUNet
# Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring
# method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

import os
from copy import deepcopy
from typing import List, Union

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from batchgenerators.utilities.file_and_folder_operations import isfile, load_json, save_pickle
from nnunetv2.utilities.label_handling.label_handling import LabelManager

from spineps_.utils.plans_handler import ConfigurationManager, PlansManager


def convert_predicted_logits_to_segmentation_with_correct_shape(
    predicted_logits: torch.Tensor | np.ndarray,
    plans_manager: PlansManager,
    configuration_manager: ConfigurationManager,
    label_manager: LabelManager,
    properties_dict: dict,
    return_probabilities: bool = False,
    num_threads_torch: int = 8,
):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape
    current_spacing = (
        configuration_manager.spacing
        if len(configuration_manager.spacing) == len(properties_dict["shape_after_cropping_and_before_resampling"])
        else [properties_dict["spacing"][0], *configuration_manager.spacing]
    )
    print("shape_after_cropping_and_before_resampling", properties_dict["shape_after_cropping_and_before_resampling"])
    print(current_spacing, properties_dict["spacing"])
    predicted_logits = configuration_manager.resampling_fn_probabilities(
        predicted_logits,
        properties_dict["shape_after_cropping_and_before_resampling"],
        current_spacing,
        properties_dict["spacing"],
    )
    # return value of resampling_fn_probabilities can be ndarray or Tensor but that doesnt matter because
    # apply_inference_nonlin will covnert to torch
    predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
    del predicted_logits
    segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)

    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(
        properties_dict["shape_before_cropping"],
        dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16,
    )
    slicer = bounding_box_to_slice(properties_dict["bbox_used_for_cropping"])
    segmentation_reverted_cropping[slicer] = segmentation
    del segmentation

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)
    if return_probabilities:
        # revert cropping
        try:
            predicted_probabilities = label_manager.revert_cropping_on_probabilities(
                predicted_probabilities,
                properties_dict["bbox_used_for_cropping"],
                properties_dict["shape_before_cropping"],
            )
        except:
            predicted_probabilities = label_manager.revert_cropping(
                predicted_probabilities.numpy() if isinstance(predicted_probabilities, torch.Tensor) else predicted_probabilities,
                properties_dict["bbox_used_for_cropping"],
                properties_dict["shape_before_cropping"],
            )

        if isinstance(predicted_probabilities, torch.Tensor):
            predicted_probabilities = predicted_probabilities.cpu().numpy()
        # revert transpose
        predicted_probabilities = predicted_probabilities.transpose([0] + [i + 1 for i in plans_manager.transpose_backward])
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping, predicted_probabilities
    else:
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping


def export_prediction_from_logits(
    predicted_array_or_file: np.ndarray | torch.Tensor,
    properties_dict: dict,
    configuration_manager: ConfigurationManager,
    plans_manager: PlansManager,
    dataset_json_dict_or_file: dict | str,
    output_file_truncated: str,
    save_probabilities: bool = False,
):
    # if isinstance(predicted_array_or_file, str):
    #     tmp = deepcopy(predicted_array_or_file)
    #     if predicted_array_or_file.endswith('.npy'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)
    #     elif predicted_array_or_file.endswith('.npz'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)['softmax']
    #     os.remove(tmp)
    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file,
        plans_manager,
        configuration_manager,
        label_manager,
        properties_dict,
        return_probabilities=save_probabilities,
    )
    del predicted_array_or_file

    # save
    if save_probabilities:
        segmentation_final, probabilities_final = ret
        np.savez_compressed(output_file_truncated + ".npz", probabilities=probabilities_final)
        save_pickle(properties_dict, output_file_truncated + ".pkl")
        del probabilities_final, ret
    else:
        segmentation_final = ret
        del ret

    rw = plans_manager.image_reader_writer_class()
    out_fname = output_file_truncated + dataset_json_dict_or_file["file_ending"]
    rw.write_seg(segmentation_final, out_fname, properties_dict)
    print(f"Saved Segmentation into {out_fname}")
