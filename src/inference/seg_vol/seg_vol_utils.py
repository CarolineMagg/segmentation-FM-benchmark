import sys
from argparse import Namespace
from typing import Union, Optional

import monai.transforms as transforms
import numpy as np
import torch
import torch.nn.functional as F

from src.inference.utils_3dprompts import reorder_bbox_coordinates, reorder_point_coordinates
from src.project_root import PROJECT_ROOT

SEGVOL_MODULE_PATH = PROJECT_ROOT / "submodules" / "SegVol"  # Get the absolute path
if str(SEGVOL_MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(SEGVOL_MODULE_PATH))

from submodules.SegVol.network.model import SegVol
from submodules.SegVol.segment_anything_volumetric import sam_model_registry
from submodules.SegVol.data_process.demo_data_process import ForegroundNormalization
from submodules.SegVol.data_utils import MinMaxNormalization
from submodules.SegVol.utils.monai_inferers_utils import logits2roi_coor, sliding_window_inference, build_binary_cube, \
    build_binary_points


def create_seg_vol_predictor(device: torch.device, spatial_size: tuple, patch_size: tuple):
    clip_ckpt = SEGVOL_MODULE_PATH / "config" / "clip"
    segvol_ckpt = PROJECT_ROOT / "checkpoints" / "seg_vol" / "SegVol_v1.pth"
    if not segvol_ckpt.exists():
        raise FileExistsError(
            f"The checkpoint file {segvol_ckpt} does not exist. Make sure the checkpoint file is downloaded into the correct folder.")

    test_mode = True
    args_model = Namespace()
    args_model.patch_size = patch_size
    args_model.spatial_size = spatial_size
    sam_model = sam_model_registry["vit"](args=args_model)
    segvol_model = SegVol(image_encoder=sam_model.image_encoder,
                          mask_decoder=sam_model.mask_decoder,
                          prompt_encoder=sam_model.prompt_encoder,
                          clip_ckpt=clip_ckpt,
                          roi_size=spatial_size,
                          patch_size=patch_size,
                          test_mode=test_mode).cuda()
    segvol_model.to(device)

    checkpoint = torch.load(str(segvol_ckpt), map_location=device, weights_only=True)
    state_dict = checkpoint["model"]
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    segvol_model.load_state_dict(state_dict, strict=False)

    return segvol_model


def process_volume_like_segvol(input_image: np.ndarray, spatial_size: Union[list, tuple]) -> dict:
    # https://github.com/BAAI-DCAI/SegVol/blob/main/data_process/demo_data_process.py#L49
    transform = transforms.Compose([ForegroundNormalization(keys=["image"]),
                                    MinMaxNormalization(),
                                    # transforms.SpatialPadd(keys=["image"], spatial_size=spatial_size, mode="constant"),  # does not do anything
                                    transforms.CropForegroundd(keys=["image"], source_key="image"),
                                    transforms.ToTensor(),
                                    ])
    zoom_out_transform = transforms.Resized(keys=["image"], spatial_size=spatial_size, mode="nearest-exact")
    item = {"image": np.expand_dims(input_image, 0)}
    item = transform(item)
    item_zoom_out = zoom_out_transform(item)
    item["zoom_out_image"] = item_zoom_out["image"]
    return item


def inverse_process_volume_like_segvol(mask: np.ndarray, foreground_start_coord: np.ndarray, foreground_end_coord: np.ndarray,
                                       img_shape: np.ndarray, spatial_size: Union[list, tuple, np.ndarray]):
    assert mask.ndim > 2, "expect at least 3 dimensions"
    # invert foreground crop
    foreground_shape = tuple(np.maximum(spatial_size, img_shape))
    num_channels = len(mask)
    new_mask = np.zeros([num_channels, *foreground_shape], np.float64)
    fg_slices = tuple(slice(s, e) for s, e in zip(foreground_start_coord, foreground_end_coord))
    for idx in range(num_channels):
        new_mask[idx][fg_slices] = mask[idx]

    # if necessary, invert Padding
    if foreground_shape != img_shape:
        total_pad = [p - o for p, o in zip(foreground_shape, img_shape)]
        pad_before = [p // 2 for p in total_pad]
        pad_after = [p - b for p, b in zip(total_pad, pad_before)]

        crop_slices = tuple(
            slice(b, fs - a)
            for b, fs, a in zip(pad_before, foreground_shape, pad_after)
        )
        restored_mask = np.zeros((num_channels, *img_shape), dtype=np.float64)
        for idx in range(num_channels):
            restored_mask[idx] = new_mask[idx][crop_slices]
        return restored_mask
    return new_mask


def prompt_based_prediction_segvol_style_combination(predictor, image: torch.Tensor, image_resized: torch.Tensor,
                                                     prompts_point_resized: np.ndarray, prompts_box_resized: np.ndarray,
                                                     spatial_size: tuple[int, int, int], use_zoom_in: bool = True):
    # https://github.com/BAAI-DCAI/SegVol/blob/main/inference_demo.py#L157 and
    # https://github.com/BAAI-DCAI/SegVol/blob/main/inference_demo.py#L47
    ori_shape = image.shape[2:]
    resized_shape = image_resized.shape[2:]
    use_box_prompt = False if prompts_box_resized is None else True
    use_point_prompt = False if prompts_point_resized is None else True
    prompts_box_resized_torch: Optional[torch.Tensor] = None
    prompts_point_resized_torch: Optional[tuple[torch.Tensor, : torch.Tensor]] = None
    if use_box_prompt:
        prompts_box_resized_torch = torch.from_numpy(reorder_bbox_coordinates(prompts_box_resized)).float().cuda()
        binary_cube_resize: torch.Tensor = build_binary_cube(prompts_box_resized_torch, binary_cube_shape=resized_shape)
    if use_point_prompt:
        prompts_point_resized_reordered: np.ndarray = reorder_point_coordinates(prompts_point_resized)
        points: torch.Tensor = torch.tensor(prompts_point_resized_reordered[:, :-1])
        points_label: torch.Tensor = torch.tensor(prompts_point_resized_reordered[:, -1])
        prompts_point_resized_torch: tuple[torch.Tensor, : torch.Tensor] = (
            points.unsqueeze(0).float().cuda(), points_label.unsqueeze(0).float().cuda())
        binary_points_resized: torch.Tensor = build_binary_points(points, points_label, resized_shape)
    if not use_point_prompt and not use_box_prompt:
        return None, None

    ## zoom-out inference
    with torch.no_grad():
        logits_global: torch.Tensor = predictor(image_resized.cuda(),
                                                text=None,
                                                boxes=prompts_box_resized_torch,
                                                points=prompts_point_resized_torch)
    logits_global: torch.Tensor = F.interpolate(logits_global.cpu(), size=ori_shape, mode='nearest')[0][0]
    logits_corrected: torch.Tensor = logits_global.clone()

    ## zoom-in inference
    if use_zoom_in:
        # get ROI
        min_d, min_h, min_w, max_d, max_h, max_w = logits2roi_coor(spatial_size, logits_global)
        if min_d is None:
            print('Fail to detect foreground!')
            return None, None

        # get prompts in ROI
        prompt_reflection = None
        global_preds: torch.Tensor = (
                torch.sigmoid(logits_global[min_d:max_d + 1, min_h:max_h + 1, min_w:max_w + 1]) > 0.5).long()
        if use_box_prompt:
            binary_cube: torch.Tensor = F.interpolate(binary_cube_resize.unsqueeze(0).unsqueeze(0).float(),
                                                      size=ori_shape, mode="nearest")[0][0]
            binary_cube_cropped = binary_cube[min_d:max_d + 1, min_h:max_h + 1, min_w:max_w + 1]
            prompt_reflection = (
                binary_cube_cropped.unsqueeze(0).unsqueeze(0),
                global_preds.unsqueeze(0).unsqueeze(0)
            )
        if use_point_prompt:
            binary_points = F.interpolate(binary_points_resized.unsqueeze(0).unsqueeze(0).float(),
                                          size=ori_shape, mode='nearest')[0][0]
            binary_points_cropped = binary_points[min_d:max_d + 1, min_h:max_h + 1, min_w:max_w + 1]
            prompt_reflection = (
                binary_points_cropped.unsqueeze(0).unsqueeze(0),
                global_preds.unsqueeze(0).unsqueeze(0)
            )

        # get image ROI
        image_cropped: torch.Tensor = image[:, :, min_d:max_d + 1, min_h:max_h + 1, min_w:max_w + 1].float()
        # prediction
        with torch.no_grad():
            logits_cropped: torch.Tensor = sliding_window_inference(
                image_cropped.cuda(), prompt_reflection,
                spatial_size, 1, predictor, 0.5,
                text=None,
                use_box=use_box_prompt,
                use_point=use_point_prompt,
            )
            logits_cropped = logits_cropped.cpu().squeeze()
        # merge prediction back into image
        logits_corrected[min_d:max_d + 1, min_h:max_h + 1, min_w:max_w + 1] = logits_cropped

    return logits_corrected, logits_global
