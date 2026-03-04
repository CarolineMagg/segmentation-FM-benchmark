import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.inference.utils_3dprompts import transform_3dpoint_into_components, transform_3dbox_into_components
from src.project_root import PROJECT_ROOT

MED_SAM2_MODULE_PATH = PROJECT_ROOT / "submodules" / "MedSAM2"  # Get the absolute path
if str(MED_SAM2_MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(MED_SAM2_MODULE_PATH))

from submodules.MedSAM2.sam2.build_sam import build_sam2_video_predictor_npz, build_sam2_video_predictor

import hydra
from hydra.core.global_hydra import GlobalHydra

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
    sam2_config_paths = (PROJECT_ROOT / "submodules" / "MedSAM2" / "sam2" / "configs")
    hydra.initialize_config_dir(config_dir=str(sam2_config_paths), version_base="1.2")
    cfg = hydra.compose(config_name="sam2.1_hiera_t512.yaml")


def create_med_sam2_predictor(device=torch.device, model_type: str = "MedSAM2_latest"):
    model_cfg = "sam2.1_hiera_t512.yaml"
    checkpoint_folder = MED_SAM2_MODULE_PATH / "checkpoints"
    if model_type == "MedSAM2_latest":
        checkpoint = checkpoint_folder / "MedSAM2_latest.pt"
    elif model_type == "MedSAM2_2411":
        checkpoint = checkpoint_folder / "MedSAM2_2411.pt"
    else:
        raise ValueError(f"invalid model_type {model_type}")
    if not checkpoint.exists():
        raise FileExistsError(
            f"The checkpoint file {checkpoint} does not exist. Make sure the checkpoint file is downloaded into the correct folder.")
    predictor = build_sam2_video_predictor_npz(model_cfg, ckpt_path=checkpoint, device=device)
    return predictor


def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    Resize a 3D grayscale NumPy array to an RGB image and then resize it.

    Parameters:
        array (np.ndarray): Input array of shape (d, h, w).
        image_size (int): Desired size for the width and height.

    Returns:
        np.ndarray: Resized array of shape (d, 3, image_size, image_size).
    """
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))

    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
        resized_array[i] = img_array

    return resized_array


def process_volume_like_med_sam2(input_image: np.ndarray) -> torch.Tensor:
    input_data = (input_image - np.min(input_image)) / (
            np.max(input_image) - np.min(input_image)) * 255.0
    input_data = np.uint8(input_data)
    assert np.max(input_data) < 256, f'input data should be in range [0, 255], but got {np.max(input_image)}'
    img_resized = resize_grayscale_to_rgb_and_resize(input_data, 512)
    img_resized = img_resized / 255.0
    img_resized = torch.from_numpy(img_resized).cuda()
    img_mean = (0.485, 0.456, 0.406)
    img_std = (0.229, 0.224, 0.225)
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].cuda()
    img_resized -= img_mean
    img_resized /= img_std
    return img_resized


def extract_and_process_slice_like_med_sam2(input_image: np.ndarray, slice_idx: int):
    key_slice_img: np.ndarray = input_image[slice_idx]
    img_resized = resize_grayscale_to_rgb_and_resize(input_image, 512)
    img_resized = img_resized / 255.0
    img_resized = torch.from_numpy(img_resized).cuda()
    img_mean = (0.485, 0.456, 0.406)
    img_std = (0.229, 0.224, 0.225)
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].cuda()
    img_resized -= img_mean
    img_resized /= img_std
    return img_resized


def prompt_based_prediction_med_sam2_video_style_combination(predictor, inference_state,
                                                             prompt_points: np.ndarray, prompt_bbox: np.ndarray,
                                                             label: int, reverse: Optional[bool] = None,
                                                             start_idx: Optional[int] = None,
                                                             max_frame_numbers: Optional[int] = None):
    video_segments: dict[int: np.ndarray] = {}

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.reset_state(inference_state)
        if len(prompt_points) == 0 and len(prompt_bbox) == 0:  # no prompt
            return video_segments
        elif len(prompt_points) > 0 and len(prompt_bbox) > 0:  # both bounding box and point
            point_coordinates, point_labels, point_frames = transform_3dpoint_into_components(prompt_points)
            box_coordinates, box_frames = transform_3dbox_into_components(prompt_bbox)
            assert point_frames == box_frames, "init frames for box and point need to be the same"
            assert len(box_frames) > 0, "init needs at least one frame"
            for init_frame in point_frames:
                assert len(box_coordinates[init_frame]) <= 1, "not more than one box allowed"
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=init_frame,
                    obj_id=label,
                    points=point_coordinates[init_frame],
                    labels=point_labels[init_frame],
                    box=box_coordinates[init_frame],
                )
        elif len(prompt_bbox) == 0 and len(prompt_points) > 0:  # only point
            point_coordinates, point_labels, frames = transform_3dpoint_into_components(prompt_points)
            for init_frame in frames:
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=init_frame,
                    obj_id=label,
                    points=point_coordinates[init_frame],
                    labels=point_labels[init_frame],
                )
        elif len(prompt_bbox) > 0 and len(prompt_points) == 0:  # only bbox
            box_coordinates, frames = transform_3dbox_into_components(prompt_bbox)
            for init_frame in frames:
                assert len(box_coordinates[init_frame]) <= 1, "not more than one box allowed"
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=init_frame,
                    obj_id=label,
                    box=box_coordinates[init_frame],
                )

        # propagate through volume
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,
                                                                                        reverse=reverse,
                                                                                        start_frame_idx=start_idx,
                                                                                        max_frame_num_to_track=max_frame_numbers):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
    return video_segments
