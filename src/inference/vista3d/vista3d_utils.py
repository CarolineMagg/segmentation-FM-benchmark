import sys
from functools import partial
from typing import Union, Any

import numpy as np
import torch
import torchio as tio
from monai.data import MetaTensor

from src.inference.utils_3dprompts import reorder_point_coordinates
from src.project_root import PROJECT_ROOT
from submodules.vista3d.vista3d.scripts.utils.trans_utils import get_largest_connected_component_point, \
    VistaPostTransform

VISTA_MODULE_PATH = PROJECT_ROOT / "submodules" / "vista3d" / "vista3d"  # Get the absolute path
if str(VISTA_MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(VISTA_MODULE_PATH))

from submodules.vista3d.vista3d.scripts.sliding_window import point_based_window_inferer
from submodules.vista3d.vista3d.vista3d import vista_model_registry


def create_vista3d_predictor(device: torch.device):
    model_registry = "vista3d_segresnet_d"
    patch_size = [128, 128, 128]
    input_channels = 1
    ckpt_name = PROJECT_ROOT / "checkpoints" / "vista3d" / "model.pt"
    model = vista_model_registry[model_registry](
        in_channels=input_channels, image_size=patch_size
    )
    model.to(device)

    if not ckpt_name.exists():
        raise FileExistsError(
            f"The checkpoint file {ckpt_name} does not exist. Make sure the checkpoint file is downloaded into the correct folder.")
    pretrained_ckpt = torch.load(ckpt_name, map_location=device, weights_only=False)
    model.load_state_dict(pretrained_ckpt, strict=False)

    model.eval()

    return model


def process_volume_like_vista3d(input_image: np.ndarray, affine_orig: np.ndarray,
                                pixdim: Union[np.ndarray, list]) -> tuple[Any, np.ndarray]:
    # https://github.com/Project-MONAI/VISTA/blob/main/vista3d/configs/infer.yaml#L12
    input_tensor = torch.from_numpy(input_image)
    D, H, W = input_tensor.shape
    # assert H == W
    subject = tio.Subject(image=tio.ScalarImage(tensor=input_tensor.unsqueeze(0), affine=affine_orig))
    assert subject.shape == (1, D, H, W)
    # cannot use original MONAI pipeline, as monai 1.2.0 is incompatibly with numpy >= 2.0 -> resort to tio
    transforms = tio.Compose([
        tio.RescaleIntensity(
            in_min_max=(-963.8247715525971, 1053.678477684517),
            out_min_max=(0.0, 1.0),
        ),
        tio.ToCanonical(),  # Converts to RAS orientation
        tio.Resample(
            target=pixdim,  # e.g. (1.0, 1.0, 1.0)
            image_interpolation='linear',
        ),
        tio.Lambda(lambda x: x.type(torch.float32)),  # Equivalent to CastToTyped
    ])
    subject_resampled = transforms(subject)

    meta_tensor = MetaTensor(subject_resampled.image.data,
                             affine=subject_resampled.image.affine.copy(),
                             meta={"original_affine": subject_resampled.image.affine.copy(),
                                   "spatial_shape": list(subject_resampled.shape[1:])})
    return meta_tensor, subject_resampled.image.affine.copy()


def invers_process_volume_like_vista3d(pred_msk: np.ndarray, spatial_shape: list,
                                       affine_original: np.ndarray, affine_resampled: np.ndarray):
    pred_label_map = tio.Subject(
        label=tio.LabelMap(tensor=pred_msk, affine=affine_resampled))

    ref_shape = (1, *spatial_shape)
    ref_image = tio.ScalarImage(
        tensor=torch.zeros(ref_shape),  # Content is irrelevant
        affine=affine_original
    )
    resample = tio.Resample(target=ref_image, image_interpolation="nearest")
    subject_resampled = resample(pred_label_map)  # preserves spacing

    return subject_resampled.label.data


def infer_wrapper(inputs, model, **kwargs):
    outputs = model(input_images=inputs, **kwargs)
    return outputs.transpose(1, 0)


def prompt_based_prediction_vista3d_style_combination(predictor, input_image, prompt_points, device):
    # https://github.com/Project-MONAI/VISTA/blob/main/vista3d/scripts/validation/val_multigpu_autopoint_patch.py#L256
    if len(prompt_points) == 0:  # no prompt
        return None
    else:
        pc: list[int] = reorder_point_coordinates(prompt_points[:, :3])[np.newaxis, ...]  # need (z,y,x)
        pl: int = prompt_points[:, -1][np.newaxis, ...]
    prev_mask = None
    label_prompt = None
    prompt_class = None

    sliding_window_inferer = partial(point_based_window_inferer, point_start=0)

    with torch.no_grad():
        prediction = sliding_window_inferer(
            inputs=input_image.to(device),
            roi_size=[128, 128, 128],
            sw_batch_size=1,
            predictor=partial(infer_wrapper, model=predictor),
            mode="gaussian",
            overlap=0.625,  # seems to be the default for eval,
            # see eg https://github.com/Project-MONAI/VISTA/blob/main/vista3d/configs/zeroshot_eval/infer_iter_point_adrenal.yaml#L8
            progress=True,
            sw_device=device,
            device=device,
            point_coords=(
                torch.tensor(pc).to(device) if pc is not None else None
            ),
            point_labels=(
                torch.tensor(pl).to(device) if pl is not None else None
            ),
            class_vector=(
                torch.tensor(label_prompt).to(device) if label_prompt is not None else None
            ),
            prompt_class=(
                torch.tensor(prompt_class).to(device) if prompt_class is not None else None
            ),
            prev_mask=(
                torch.tensor(prev_mask).to(device) if prev_mask is not None else None
            ),
        )
    if label_prompt is None and prompt_points is not None:
        prediction = get_largest_connected_component_point(
            prediction, point_coords=pc, point_labels=pl
        )
    post_transforms = VistaPostTransform(keys="pred")
    prediction = post_transforms({"pred": prediction})["pred"]
    prediction = prediction.squeeze().to("cpu")
    torch.cuda.empty_cache()

    return prediction
