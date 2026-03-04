import sys
from typing import Optional

import numpy as np
import torch

from src.project_root import PROJECT_ROOT

SAM_MODULE_PATH = PROJECT_ROOT / "submodules" / "sam"  # Get the absolute path to `submodules/sam`
if str(SAM_MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(SAM_MODULE_PATH))

from submodules.sam.segment_anything import SamPredictor, sam_model_registry


def create_sam_predictor(device: torch.device, model_type: str = "vit_b"):
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "sam"
    if model_type == "vit_b":
        checkpoint = checkpoint_path / "sam_vit_b_01ec64.pth"
    elif model_type == "vit_l":
        checkpoint = checkpoint_path / "sam_vit_l_0b3195.pth"
    elif model_type == "vit_h":
        checkpoint = checkpoint_path / "sam_vit_h_4b8939.pth"
    else:
        raise ValueError(f"invalid model_type {model_type}")
    if not checkpoint.exists():
        raise FileExistsError(
            f"The checkpoint file {checkpoint} does not exist. Make sure the checkpoint file is downloaded into the correct folder.")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    return SamPredictor(sam)


def extract_and_process_slice_like_sam(input_image: np.ndarray, slice_idx: int):
    input_array: np.ndarray = input_image[slice_idx]
    img: np.ndarray = np.uint8((input_array - input_array.min()) / (input_array.max() - input_array.min()) * 255)
    return np.repeat(img[:, :, None], 3, axis=-1)


def prompt_based_prediction_sam_style_combination(predictor, prompt_points: np.ndarray, prompts_bbox: np.ndarray) -> \
        Optional[np.ndarray]:
    prediction: Optional[np.ndarray] = None
    if len(prompt_points) == 0 and len(prompts_bbox) == 0:  # no prompt
        return prediction
    elif len(prompts_bbox) > 0 and len(prompt_points) > 0:  # both bounding box and point
        pc: np.ndarray = prompt_points[:, :2]
        pl: np.ndarray = prompt_points[:, -1]
        for box in prompts_bbox:
            preds_single, scores, _ = predictor.predict(point_coords=pc, point_labels=pl, box=box,
                                                        multimask_output=False)
            if prediction is None:
                prediction = preds_single
            else:
                prediction += preds_single
    elif len(prompts_bbox) == 0 and len(prompt_points) > 0:  # only point
        pc: np.ndarray = prompt_points[:, :2]
        pl: np.ndarray = prompt_points[:, -1]
        prediction, scores, _ = predictor.predict(point_coords=pc, point_labels=pl, box=None,
                                                  multimask_output=False)
    elif len(prompts_bbox) > 0 and len(prompt_points) == 0:  # only bounding box
        for box in prompts_bbox:
            preds_single, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=box,
                                                        multimask_output=False)
            if prediction is None:
                prediction = preds_single
            else:
                prediction += preds_single
    # filter which mask to use -> default: take the first output
    prediction = prediction[0]  # take the first output -> no score_filtering or multi-mask output
    prediction = np.array(prediction > 0, dtype=np.uint8)
    return prediction
