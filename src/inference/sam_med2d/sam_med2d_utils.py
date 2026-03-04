import argparse

import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
import albumentations as A

from src.project_root import PROJECT_ROOT
from submodules.SAMMed2D.segment_anything import sam_model_registry
from submodules.SAMMed2D.segment_anything.predictor_sammed import SammedPredictor


def create_sammed2d_predictor(device: torch.device, model_type: str = "vit_b"):
    # follows https://github.com/OpenGVLab/SAM-Med2D/blob/main/predictor_example.ipynb
    args_sam = argparse.Namespace()
    args_sam.image_size = 256
    args_sam.encoder_adapter = True  # adapter layer in encoder
    args_sam.sam_checkpoint = PROJECT_ROOT / "checkpoints" / "sam_med2d" / "sam-med2d_b.pth"
    if not args_sam.sam_checkpoint.exists():
        raise FileExistsError(
            f"The checkpoint file {args_sam.sam_checkpoint} does not exist. Make sure the checkpoint file is downloaded into the correct folder.")
    model = sam_model_registry[model_type](args_sam).to(device)
    return SammedPredictor(model)


# currently unused, since SammedPredictor takes care of this
def extract_and_process_slice_like_sam_med2d(input_image: np.ndarray, slice_idx: int):
    # https://github.com/OpenGVLab/SAM-Med2D/blob/main/DataLoader.py#L17 and
    pixel_mean = [123.675, 116.28, 103.53]
    pixel_std = [58.395, 57.12, 57.375]
    img_size = 256
    input_array: np.ndarray = input_image[slice_idx]
    input_array = (input_array - pixel_mean) / pixel_std
    ori_h, ori_w = input_image.shape
    # get transforms
    # https://github.com/OpenGVLab/SAM-Med2D/blob/main/utils.py#L161
    transforms = []
    if ori_h < img_size and ori_w < img_size:
        transforms.append(
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    else:
        transforms.append(A.Resize(int(img_size), int(img_size), interpolation=cv2.INTER_NEAREST))
    transforms.append(ToTensorV2(p=1.0))
    transform = A.Compose(transforms, p=1.)
    image_transformed = transform(image=input_array)
    return image_transformed["image"]
