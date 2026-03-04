import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from src.inference.utils_3dprompts import transform_3dpoint_into_components, transform_3dbox_into_components
from src.project_root import PROJECT_ROOT

SAM_MODULE_PATH = PROJECT_ROOT / "submodules" / "sam2"  # Get the absolute path to `submodules/sam2`
if str(SAM_MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(SAM_MODULE_PATH))

from submodules.sam2.sam2.build_sam import build_sam2_video_predictor, build_sam2
from submodules.sam2.sam2.sam2_image_predictor import SAM2ImagePredictor

all_sam2_model_type_options = ["sam2_hiera_large", "sam2.0_hiera_large",
                               "sam2_hiera_tiny", "sam2.0_hiera_tiny",
                               "sam2_hiera_small", "sam2.0_hiera_small",
                               "sam2_hiera_base_plus", "sam2.0_hiera_base_plus",
                               "sam2_new_hiera_large", "sam2.1_hiera_large",
                               "sam2_new_hiera_tiny", "sam2.1_hiera_tiny",
                               "sam2_new_hiera_small", "sam2.1_hiera_small",
                               "sam2_new_hiera_base_plus", "sam2.1_hiera_base_plus"]

import hydra
from hydra.core.global_hydra import GlobalHydra

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
    sam2_config_paths = (PROJECT_ROOT / "submodules" / "sam2" / "sam2" / "configs")
    hydra.initialize_config_dir(config_dir=str(sam2_config_paths), version_base="1.2")
    cfg = hydra.compose(config_name="sam2.1/sam2.1_hiera_t.yaml")


def create_sam2_predictor(device, model_type, video_predictor=False):
    checkpoint_folder = PROJECT_ROOT / "checkpoints" / "sam2"
    if model_type == "sam2_hiera_large" or model_type == "sam2.0_hiera_large":
        sam2_checkpoint = checkpoint_folder / "sam2_hiera_large.pt"
        model_cfg = "sam2/sam2_hiera_l.yaml"
    elif model_type == "sam2_hiera_base_plus" or model_type == "sam2.0_hiera_base_plus":
        sam2_checkpoint = checkpoint_folder / "sam2_hiera_base_plus.pt"
        model_cfg = "sam2/sam2_hiera_b+.yaml"
    elif model_type == "sam2_hiera_tiny" or model_type == "sam2.0_hiera_tiny":
        sam2_checkpoint = checkpoint_folder / "sam2_hiera_tiny.pt"
        model_cfg = "sam2/sam2_hiera_t.yaml"
    elif model_type == "sam2_hiera_small" or model_type == "sam2.0_hiera_small":
        sam2_checkpoint = checkpoint_folder / "sam2_hiera_small.pt"
        model_cfg = "sam2/sam2_hiera_s.yaml"
    elif model_type == "sam2_new_hiera_large" or model_type == "sam2.1_hiera_large":
        sam2_checkpoint = checkpoint_folder / "sam2.1_hiera_large.pt"
        model_cfg = "sam2.1/sam2.1_hiera_l.yaml"
    elif model_type == "sam2_new_hiera_base_plus" or model_type == "sam2.1_hiera_base_plus":
        sam2_checkpoint = checkpoint_folder / "sam2.1_hiera_base_plus.pt"
        model_cfg = "sam2.1/sam2.1_hiera_b+.yaml"
    elif model_type == "sam2_new_hiera_tiny" or model_type == "sam2.1_hiera_tiny":
        sam2_checkpoint = checkpoint_folder / "sam2.1_hiera_tiny.pt"
        model_cfg = "sam2.1/sam2.1_hiera_t.yaml"
    elif model_type == "sam2_new_hiera_small" or model_type == "sam2.1_hiera_small":
        sam2_checkpoint = checkpoint_folder / "sam2.1_hiera_small.pt"
        model_cfg = "sam2.1/sam2.1_hiera_s.yaml"
    else:
        raise ValueError(f"invalid model_type {model_type}")
    if not sam2_checkpoint.exists():
        raise FileExistsError(
            f"The checkpoint file {sam2_checkpoint} does not exist. Make sure the checkpoint file is downloaded into the correct folder.")
    if video_predictor:
        predictor = build_sam2_video_predictor(str(model_cfg), str(sam2_checkpoint), device=device)
    else:
        sam2_model = build_sam2(str(model_cfg), str(sam2_checkpoint), device=device)
        predictor = SAM2ImagePredictor(sam2_model)
    return predictor


def create_png_video_dir(video_dir: Path, input_image: np.ndarray):
    os.makedirs(video_dir, exist_ok=True)
    for slice_idx in range(len(input_image)):
        # create temporary video dir
        input_array = input_image[slice_idx]
        input_array = np.uint8((input_array - input_array.min()) / (input_array.max() - input_array.min()) * 255)
        im = Image.fromarray(input_array)
        im.save(os.path.join(video_dir, f"{slice_idx}.jpeg"))


def prompt_based_prediction_sam2_video_style_combination(predictor, inference_state,
                                                         prompt_points: np.ndarray, prompt_bbox: np.ndarray,
                                                         label: int, reverse: Optional[bool] = None,
                                                         start_idx: Optional[int] = None,
                                                         max_frame_numbers: Optional[int] = None):
    predictor.reset_state(inference_state)
    video_segments: dict[int: np.ndarray] = {}
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
