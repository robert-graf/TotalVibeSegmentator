from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
from TPTBox import NII, Log_Type, No_Logger
from TPTBox.core import sitk_utils

from .predictor import nnUNetPredictor

logger = No_Logger()
logger.override_prefix = "API"

_interop = False


# Adapted from https://github.com/MIC-DKFZ/nnUNet
# Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring
# method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.
def load_inf_model(
    model_folder: str | Path,
    step_size: float = 0.5,
    ddevice: str = "cuda",
    use_folds: tuple[str | int, ...] | None = None,
    init_threads: bool = True,
    allow_non_final: bool = True,
    inference_augmentation: bool = False,
    verbose: bool = True,
    gpu=None,
) -> nnUNetPredictor:
    """Loades the Nako-Segmentor Model Predictor

    Args:
        model_folder (str, optional): nnUNet Result Model Folder containing the "fold_x" directories. Default to the basic folder
        step_size (float, optional): Step size for sliding window prediction. The larger it is the faster but less accurate "
        "the prediction. Default: 0.5. Cannot be larger than 1.
        ddevice (str, optional): The device the inference should run with. Available options are 'cuda' "
        "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID!. Defaults to "cuda".

    Returns:
        predictor: Loaded model predictor object
    """
    if isinstance(model_folder, str):
        model_folder = Path(model_folder)
    if ddevice == "cpu":
        # let's allow torch to use hella threads
        import multiprocessing

        torch.set_num_threads(multiprocessing.cpu_count()) if init_threads else None
        device = torch.device("cpu")
    elif ddevice == "cuda":
        # multithreading in torch doesn't help nnU-Net if run on GPU
        try:
            torch.set_num_threads(1) if init_threads else None
            global _interop  # noqa: PLW0603
            if not _interop:
                torch.set_num_interop_threads(1) if init_threads else None
                _interop = True
        except Exception as e:
            print(e)
        device = torch.device("cuda")
    else:
        device = torch.device("mps")

    assert model_folder.exists(), f"model-folder not found: got path {model_folder}"

    predictor = nnUNetPredictor(
        tile_step_size=step_size,
        use_gaussian=True,
        use_mirroring=inference_augmentation,  # <- mirroring augmentation!
        perform_everything_on_gpu=ddevice != "cpu",
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        cuda_id=0 if gpu is None else gpu,
    )
    check_name = "checkpoint_final.pth"  # if not allow_non_final else "checkpoint_best.pth"
    try:
        predictor.initialize_from_trained_model_folder(str(model_folder), checkpoint_name=check_name, use_folds=use_folds)
    except Exception as e:
        if allow_non_final:
            predictor.initialize_from_trained_model_folder(
                str(model_folder),
                checkpoint_name="checkpoint_best.pth",
                use_folds=use_folds,
            )
            logger.print("Checkpoint final not found, will load from best instead", Log_Type.WARNING)
        else:
            raise e  # noqa: TRY201
    logger.print(f"Inference Model loaded from {model_folder}") if verbose else None
    return predictor


def run_inference(
    input_nii: str | NII | list[NII],
    predictor: nnUNetPredictor,
    reorient_PIR: bool = False,  # noqa: N803
    logits=False,
) -> tuple[NII, NII | None, np.ndarray | None]:
    """Runs nnUnet model inference on one input.

    Args:
        input (str | NII): Path to a nifty file or a NII object.
        predictor (_type_, optional): Loaded model predictor. If none, will load the default one. Defaults to None.

    Raises:
        AssertionError: If the input is not of expected type

    Returns:
        Segmentation (NII), Uncertainty Map (NII), Softmax Logits (numpy arr)
    """
    if isinstance(input_nii, str):
        assert input_nii.endswith(".nii.gz"), f"input file is not a .nii.gz! Got {input_nii}"
        input_nii = NII.load(input_nii, seg=False)

    assert isinstance(input_nii, NII | list), f"input must be a NII or str or list[NII], got {type(input_nii)}"
    if isinstance(input_nii, NII):
        input_nii = [input_nii]
    orientation = input_nii[0].orientation
    zoom = input_nii[0].zoom

    img_arrs = []
    # Prepare for nnUNet behavior
    for i in input_nii:
        if reorient_PIR:
            i.reorient_()
        sitk_nii = sitk_utils.nii_to_sitk(i)
        nii_img_converted = sitk.GetArrayFromImage(sitk_nii).astype(np.float16)[np.newaxis, :]
        # nii_img_converted = i.get_array()
        # nii_img_converted = np.pad(nii_img_converted, pad_width=pad_size, mode="edge")
        # nii_img_converted = np.swapaxes(nii_img_converted, 0, 2)[np.newaxis, :].astype(np.float16)
        img_arrs.append(nii_img_converted)
    try:
        img = np.vstack(img_arrs)
    except Exception:
        print([a.shape for a in img_arrs])
        raise
    props = {
        "sitk_stuff": {
            # this saves the sitk geometry information. This part is NOT used by nnU-Net!
            "spacing": sitk_nii.GetSpacing(),  # type:ignore
            "origin": sitk_nii.GetOrigin(),  # type:ignore
            "direction": sitk_nii.GetDirection(),  # type:ignore
        },
        "spacing": zoom[::-1],  # PIR
    }
    out = predictor.predict_single_npy_array(img, props, save_or_return_probabilities=logits)
    if logits:
        segmentation, _, softmax_logits = out  # type: ignore
        print("logits", segmentation.shape, softmax_logits.shape)
        softmax_logits = np.expand_dims(softmax_logits.astype(np.float16), 0)
        # softmax_logits = np.swapaxes(softmax_logits, 0, 3)
        # PRI label
        # softmax_logits = np.swapaxes(softmax_logits, 1, 2)
    else:
        segmentation, _ = out  # type: ignore
        softmax_logits = None
    print("Start Post")
    itk_image = sitk.GetImageFromArray(segmentation.astype(np.uint8))
    itk_image.SetSpacing(sitk_nii.GetSpacing())
    itk_image.SetOrigin(sitk_nii.GetOrigin())
    itk_image.SetDirection(sitk_nii.GetDirection())
    seg_nii = sitk_utils.sitk_to_nii(itk_image, True)

    # segmentation = np.swapaxes(segmentation, 0, 2)
    # assert isinstance(segmentation, np.ndarray)
    # seg_nii = NII(nib.ni1.Nifti1Image(segmentation, affine=affine, header=header), seg=True)

    seg_nii.reorient_(orientation)
    return seg_nii, None, softmax_logits
