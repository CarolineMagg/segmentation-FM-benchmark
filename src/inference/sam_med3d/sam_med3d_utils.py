import sys
from typing import Callable, Optional, Union, Any

import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
from torch import Tensor

from src.inference.utils_3dprompts import transform_3dpoint_resample, \
    transform_3dpoint_crop_or_pad, reorder_point_coordinates
from src.project_root import PROJECT_ROOT

SAMMed3d_MODULE_PATH = PROJECT_ROOT / "submodules" / "SAMMed3D"
if str(SAMMed3d_MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(SAMMed3d_MODULE_PATH))

from submodules.SAMMed3D.segment_anything import sam_model_registry3D


def create_sammed3d_predictor(device, model_type: str):
    if model_type == "sam_med3d":
        model_type = "vit_b_ori"
        checkpoint_path = PROJECT_ROOT / "checkpoints" / "sam_med3d" / "sam_med3d.pth"
    elif model_type == "sam_med3d_turbo":
        model_type = "vit_b_ori"
        checkpoint_path = PROJECT_ROOT / "checkpoints" / "sam_med3d" / "sam_med3d_turbo.pth"
    else:
        raise ValueError(f"invalid model_type {model_type}")

    sam_model_tune = sam_model_registry3D[model_type](checkpoint=None).to(device)
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileExistsError(
                f"The checkpoint file {checkpoint_path} does not exist. Make sure the checkpoint file is downloaded into the correct folder.")
        model_dict = torch.load(checkpoint_path, map_location=device)
        state_dict = model_dict['model_state_dict']
        sam_model_tune.load_state_dict(state_dict)

    return sam_model_tune


def correct_roi_dim(roi_tensor):
    if roi_tensor.ndim == 3:
        roi_tensor = roi_tensor.unsqueeze(0).unsqueeze(0)
    if roi_tensor.ndim == 4:
        roi_tensor = roi_tensor.unsqueeze(0)
    return roi_tensor


def create_crop_mask_from_prompt(input_point, mask_shape):
    crop_mask = torch.zeros_like(mask_shape)
    input_point_ = reorder_point_coordinates(input_point)
    for point in input_point_:
        indices = (0,) + tuple(point[:-1])
        crop_mask[indices] = 1
    return crop_mask


def process_data_like_sammed3d(input_image: np.ndarray, new_spacing: Union[list, np.ndarray], crop_size: int,
                               meta_info: dict, device: torch.device,
                               data_processing: str, input_point_original: np.ndarray = None) -> tuple[
    Any, np.ndarray, dict]:
    # SAM-Med3d performs CropOrPad to 128x128x128:
    # following: https://github.com/uni-medical/SAM-Med3D/blob/main/utils/infer_utils.py#L246
    # this is a problem for structures that are bigger than the crop -> some background might be missing
    # thus, we resize based on the longest side and optional pad as an alternative

    if data_processing == "crop" and input_point_original is None:
        raise RuntimeError("need to provide input_point for 'crop' data-processing")

    input_tensor = torch.from_numpy(input_image)
    D, H, W = input_tensor.shape
    # assert H == W
    subject = tio.Subject(image=tio.ScalarImage(tensor=input_tensor.unsqueeze(0),
                                                affine=meta_info['original_subject_affine']))
    assert subject.shape == (1, D, H, W)

    # resample to sam med 3d spacing
    resample_sammed3d: Callable = tio.Resample(target=new_spacing)  # preserves spacing
    subject_resampled = resample_sammed3d(subject)
    _, D_, H_, W_ = subject_resampled.shape
    # assert H_ == W_
    meta_info["resampled_sammed3d_subject_affine"] = subject_resampled.image.affine.copy()
    meta_info["resampled_sammed3d_subject_spacing"] = np.asarray(subject_resampled.spacing)
    meta_info["resampled_sammed3d_subject_spatial_shape"] = subject_resampled.image.spatial_shape
    input_point_resampled = transform_3dpoint_resample(input_point_original,
                                                       meta_info["original_subject_affine"],
                                                       meta_info["resampled_sammed3d_subject_affine"],
                                                       meta_info["resampled_sammed3d_subject_spatial_shape"])

    if data_processing == "crop":
        # centered-crop to size 128
        crop_mask = create_crop_mask_from_prompt(input_point_resampled, subject_resampled.image.data)
        subject_resampled.add_image(tio.LabelMap(tensor=crop_mask, affine=subject_resampled.image.affine),
                                    image_name="crop_mask")
        crop_transform = tio.CropOrPad(mask_name="crop_mask", target_shape=crop_size)
        pad_params, crop_params = crop_transform.compute_crop_or_pad(subject_resampled)
        subject_cropped_padded = crop_transform(subject_resampled)

    elif data_processing == "resample":
        # resize to 128 for longest side
        ori_shape = subject_resampled.image.spatial_shape  # (x, y, z)
        ori_spacing = subject_resampled.image.spacing  # (H, W, D)
        scale_factors = [sh * sp for sh, sp in zip(ori_shape, ori_spacing)]
        assert len(scale_factors) == 3
        longest_dim = scale_factors.index(max(scale_factors))
        scale_ratio: float = ori_shape[longest_dim] / crop_size
        new_spacing_longest_side = [sp * scale_ratio for sp in ori_spacing]
        resample_longest_side = tio.Resample(target=new_spacing_longest_side)  # preserves spacing
        subject_resampled_longest_side = resample_longest_side(subject_resampled)

        meta_info["resampled2_subject_spacing"] = np.asarray(subject_resampled_longest_side.spacing)
        meta_info["resampled2_subject_affine"] = np.asarray(subject_resampled_longest_side.image.affine)
        meta_info["resampled2_subject_spatial_shape"] = subject_resampled_longest_side.image.spatial_shape
        input_point_resampled = transform_3dpoint_resample(input_point_resampled,
                                                           meta_info["resampled_sammed3d_subject_affine"],
                                                           meta_info["resampled2_subject_affine"],
                                                           meta_info["resampled2_subject_spatial_shape"])

        # pad to target_size in all dimensions
        pad_or_crop: Callable = tio.CropOrPad(target_shape=(crop_size, crop_size, crop_size))
        pad_params, crop_params = pad_or_crop._compute_center_crop_or_pad(subject_resampled_longest_side)
        subject_cropped_padded = pad_or_crop(subject_resampled_longest_side)
    else:
        raise ValueError(f"{data_processing} is not implemented")

    norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
    img3D_roi: Tensor = subject_cropped_padded.image.data.clone().detach()
    img3D_roi = norm_transform(img3D_roi.squeeze(dim=1))  # [:, None, :, :, :])  # (N, C, W, H, D)
    img3D_roi = img3D_roi.unsqueeze(dim=1)  # [:, 0, :, :, :]
    roi_image = correct_roi_dim(img3D_roi).float().to(device)

    meta_info["padding_params"] = pad_params if pad_params is not None else (0, 0, 0, 0, 0, 0)
    meta_info["cropping_params"] = crop_params if crop_params is not None else (0, 0, 0, 0, 0, 0)
    if data_processing == "resample":
        meta_info["resampled2_subject_spacing"] = np.asarray(subject_resampled_longest_side.spacing)
        meta_info["resampled2_subject_spatial_shape"] = subject_resampled_longest_side.image.spatial_shape
    meta_info["roi_subject_affine"] = subject_cropped_padded.image.affine.copy()
    meta_info["roi_subject_spacing"] = np.asarray(subject_cropped_padded.spacing)
    meta_info["roi_subject_spatial_shape"] = tuple(roi_image.shape[2:])

    input_point_cropped = transform_3dpoint_crop_or_pad(input_point_resampled, meta_info["padding_params"],
                                                        meta_info["cropping_params"],
                                                        meta_info["roi_subject_spatial_shape"])

    return roi_image, input_point_cropped, meta_info


def invers_process_volume_like_sammed3d(pred_msk: np.ndarray, meta_info: dict, data_processing:str):
    roi_pred_tensor = torch.from_numpy(pred_msk.astype(np.float32)).unsqueeze(0)
    pred_label_map_roi_space = tio.Subject(
        label=tio.LabelMap(tensor=roi_pred_tensor, affine=meta_info["roi_subject_affine"]))

    # inverse pad to target_size in all dimensions
    inverse_transform = tio.Compose([
        tio.Crop(meta_info["padding_params"]),  # remove the padding you added
        tio.Pad(meta_info["cropping_params"]),  # restore the cropped-out parts
    ])
    subject_invert_pad = inverse_transform(pred_label_map_roi_space)

    # resample to sam med3d spacing
    ref_shape_resample = (1, *meta_info["resampled_sammed3d_subject_spatial_shape"])
    ref_image_resample = tio.ScalarImage(
        tensor=torch.zeros(ref_shape_resample),  # Content is irrelevant
        affine=meta_info["resampled_sammed3d_subject_affine"]
    )
    resample2 = tio.Resample(target=ref_image_resample, image_interpolation="nearest")  # preserves spacing
    subject_resampled2 = resample2(subject_invert_pad)

    # resample to original spacing
    ref_shape_ori = (1, *meta_info["original_subject_spatial_shape"])
    ref_image_ori = tio.ScalarImage(
        tensor=torch.zeros(ref_shape_ori),  # Content is irrelevant
        affine=meta_info["original_subject_affine"]
    )
    resample = tio.Resample(target=ref_image_ori, image_interpolation="nearest")
    subject_resampled = resample(subject_resampled2)

    final_pred_numpy = subject_resampled.label.data.squeeze(0).cpu().numpy().astype(np.uint8)

    return final_pred_numpy


def prompt_based_prediction_sammed3d_style_combination(predictor, input_tensor: torch.Tensor, input_prompts: np.ndarray,
                                                       prev_low_res_mask: Optional[torch.Tensor],
                                                       device: torch.device) -> tuple[np.ndarray, torch.Tensor]:
    if len(input_prompts) == 0:
        return None, None

    with torch.no_grad():
        image_embedding = predictor.image_encoder(input_tensor)  # (1, 384, 16, 16, 16)

        for input_prompt in input_prompts:
            # prepare prompts
            pc = torch.from_numpy(input_prompt[:3]).to(device).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 3)
            pl = torch.tensor(input_prompt[-1]).to(device).unsqueeze(0).unsqueeze(0)  # shape: (1, 1)
            if prev_low_res_mask is None:  # Initialize low_res_mask for the decoder
                prev_low_res_mask = torch.zeros(1, 1, input_tensor.shape[2] // 4, input_tensor.shape[3] // 4,
                                                input_tensor.shape[4] // 4, device=device, dtype=torch.float)
            # prompt embedding
            sparse_embeddings, dense_embeddings = predictor.prompt_encoder(
                points=[pc, pl],
                boxes=None,
                masks=prev_low_res_mask,
            )
            # mask encoder
            low_res_masks, _ = predictor.mask_decoder(
                image_embeddings=image_embedding.to(device),  # (B, 384, 64, 64, 64)
                image_pe=predictor.prompt_encoder.get_dense_pe(),  # (1, 384, 64, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 384)
                dense_prompt_embeddings=dense_embeddings,  # (B, 384, 64, 64, 64)
                multimask_output=False,
            )
            # postprocessing of prob to mask
            prev_low_res_mask = low_res_masks.detach()

        # Final high-resolution mask from the last low_res_masks
        final_masks_hr = F.interpolate(low_res_masks,  # Use the final low_res_masks
                                       size=input_tensor.shape[-3:],
                                       mode='trilinear',
                                       align_corners=False)

    medsam_seg_prob = torch.sigmoid(final_masks_hr)
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    medsam_seg_mask = (medsam_seg_prob > 0.5).astype(np.uint8)

    return medsam_seg_mask, low_res_masks.detach()
