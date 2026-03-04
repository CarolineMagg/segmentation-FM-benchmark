import sys
from typing import Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np
from skimage import transform

from src.project_root import PROJECT_ROOT

SAM_MODULE_PATH = PROJECT_ROOT / "submodules" / "MedSAM"  # Get the absolute path to `submodules/MedSAM`
if str(SAM_MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(SAM_MODULE_PATH))

from submodules.MedSAM.segment_anything import sam_model_registry


def create_medsam_model(device: torch.device):
    model_type = "vit_b"
    checkpoint = PROJECT_ROOT / "checkpoints" / "med_sam" / "medsam_vit_b.pth"
    if not checkpoint.exists():
        raise FileExistsError(
            f"The checkpoint file {checkpoint} does not exist. Make sure the checkpoint file is downloaded into the correct folder.")
    medsam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    return medsam_model


def extract_and_process_slice_like_medsam(input_image: np.ndarray, slice_idx: int) -> Tuple[torch.Tensor, int, int]:
    # https://github.com/bowang-lab/MedSAM/blob/main/MedSAM_Inference.py#L114
    input_array = input_image[slice_idx]
    H, W = input_array.shape
    img_3c: np.ndarray = np.repeat(input_array[:, :, None], 3, axis=-1)
    img_1024: np.ndarray = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True,
                                            anti_aliasing=True).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), a_min=1e-8,
                                                     a_max=None)  # normalize to [0, 1], (H, W, 3)
    # convert the shape to (3, H, W)
    input_tensor: torch.Tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to("cuda")
    return input_tensor, H, W


def transform_bbox_medsam(input_prompts_bbox: Union[np.ndarray, list[list[int]]], H: int, W: int) -> list[np.ndarray]:
    # https://github.com/bowang-lab/MedSAM/blob/main/MedSAM_Inference.py#L132
    boxes_list: list[np.ndarray] = []
    for box in input_prompts_bbox:
        box_np: np.ndarray = np.array([box])  # (1,4)
        box_1024: np.ndarray = box_np / np.array([W, H, W, H]) * 1024  # transfer box_np t0 1024x1024 scale
        boxes_list.append(box_1024)
    return boxes_list


@torch.no_grad()
def bbox_based_prediction_medsam_style(medsam_model, img_embed: torch.Tensor, box_1024: np.ndarray, H: int,
                                       W: int) -> np.ndarray:
    # https://github.com/bowang-lab/MedSAM/blob/main/MedSAM_Inference.py#L44
    box_torch: torch.Tensor = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred: torch.Tensor = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred: np.ndarray = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg: np.ndarray = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg
