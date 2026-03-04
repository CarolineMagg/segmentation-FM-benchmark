import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from skimage import transform

from src.project_root import PROJECT_ROOT

SCRIBBLE_MODULE_PATH = PROJECT_ROOT / "submodules" / "ScribblePrompt"  # Get the absolute path to `submodules/sam`
if str(SCRIBBLE_MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIBBLE_MODULE_PATH))

from submodules.ScribblePrompt.scribbleprompt import ScribblePromptUNet, ScribblePromptSAM


def create_scribble_prompt_predictor(device: torch.device, model_type: str = "vit_b"):
    if model_type == "unet":
        predictor = ScribblePromptUNet(device=device)
    elif model_type == "sam":
        predictor = ScribblePromptSAM(device=device)
    else:
        raise ValueError(f"invalid model_type {model_type}")
    return predictor


def extract_and_process_slice_like_scribble_prompt(input_image: np.ndarray, slice_idx: int,
                                                   device: torch.device) -> torch.Tensor:
    # following https://colab.research.google.com/drive/14ExpVy3PjCCp4VzgTo27Yh_aLBafK8cX?usp=sharing#scrollTo=kvP5J0uDjdc2
    input_array: np.ndarray = input_image[slice_idx]
    img_128: np.ndarray = transform.resize(input_array, (128, 128), order=3, preserve_range=True,
                                           anti_aliasing=True)
    img_128: np.ndarray = np.uint8((img_128 - img_128.min()) / (img_128.max() - img_128.min()) * 255)  # [0,255]
    img_tensor: torch.Tensor = torch.from_numpy(img_128[np.newaxis, np.newaxis, :, :]) / 255  # [0,1]
    return img_tensor.float().to(device)


def prompt_based_prediction_scribble_prompt_style_combination(predictor, image: torch.Tensor,
                                                              prompt_points: np.ndarray, prompts_bbox: np.ndarray,
                                                              H: int, W: int) -> Optional[np.ndarray]:
    # following https://colab.research.google.com/drive/14ExpVy3PjCCp4VzgTo27Yh_aLBafK8cX?usp=sharing#scrollTo=kvP5J0uDjdc2
    prediction: Optional[np.ndarray] = None
    if len(prompt_points) == 0 and len(prompts_bbox) == 0:  # no prompt
        return prediction
    elif len(prompts_bbox) > 0 and len(prompt_points) > 0:  # both bounding box and point
        pc: np.ndarray = prompt_points[:, :2]
        pl: np.ndarray = prompt_points[:, -1]
        pc_tensor = torch.tensor(pc[None, ...], device=predictor.device)
        pl_tensor = torch.tensor(pl[None, ...], device=predictor.device)
        box_tensor = torch.tensor(prompts_bbox[None, ...], device=predictor.device)
        result = predictor.predict(img=image, box=box_tensor,
                                   point_coords=pc_tensor, point_labels=pl_tensor)
    elif len(prompts_bbox) == 0 and len(prompt_points) > 0:  # only point
        pc: np.ndarray = prompt_points[:, :2]
        pl: np.ndarray = prompt_points[:, -1]
        pc_tensor = torch.tensor(pc[None, ...], device=predictor.device)
        pl_tensor = torch.tensor(pl[None, ...], device=predictor.device)
        result = predictor.predict(img=image, point_coords=pc_tensor, point_labels=pl_tensor)
    elif len(prompts_bbox) > 0 and len(prompt_points) == 0:  # only bounding box
        box_tensor = torch.tensor(prompts_bbox[None, ...], device=predictor.device)
        result = predictor.predict(img=image, box=box_tensor)
    else:
        raise RuntimeError

    if len(result) == 3:
        low_res_pred = result[0]
    else:
        low_res_pred = result
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    ) # 1, 1, H, W
    low_res_pred: np.ndarray = low_res_pred.squeeze().cpu().numpy()
    prediction: np.ndarray = (low_res_pred > 0.5).astype(np.uint8)
    if len(prediction.shape) == 3:  # happens if multiple prompts are used
        prediction = (prediction.sum(axis=0) > 0).astype(np.uint8)
    return prediction
