import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.project_root import PROJECT_ROOT
SAM_MODULE_PATH = PROJECT_ROOT / "submodules" / "sam3"  # Get the absolute path to `submodules/sam2`
if str(SAM_MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(SAM_MODULE_PATH))

from src.inference.utils_3dprompts import transform_3dpoint_into_components, transform_3dbox_into_components
from submodules.sam3.sam3.model.sam3_image_processor import Sam3Processor
from submodules.sam3.sam3.model_builder import build_sam3_image_model, build_sam3_video_model




def create_sam3_predictor(device, video_predictor=False):
    if video_predictor:
        model = build_sam3_video_model(device=device)
        processor = model.tracker
        processor.backbone = model.detector.backbone
        processor.backbone.language_backbone = None # won't be used, and so the model is smaller
    else:
        bpe_path = f"{SAM_MODULE_PATH}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        model = build_sam3_image_model(bpe_path=bpe_path, enable_inst_interactivity=True, device=device)
        model = model.to(device=device)
        processor = Sam3Processor(model, confidence_threshold=0.5)
    return processor, model


def extract_and_process_slice_like_sam3(input_image: np.ndarray, slice_idx: int):
    input_array: np.ndarray = input_image[slice_idx]
    img: np.ndarray = np.uint8((input_array - input_array.min()) / (input_array.max() - input_array.min()) * 255)
    return img  # np.repeat(img[:, :, None], 3, axis=-1)


def create_png_video_dir(video_dir: Path, input_image: np.ndarray):
    os.makedirs(video_dir, exist_ok=True)
    for slice_idx in range(len(input_image)):
        input_array = input_image[slice_idx]
        input_array = np.uint8((input_array - input_array.min()) / (input_array.max() - input_array.min()) * 255)
        input_array = np.repeat(input_array[:, :, None], 3, axis=-1)
        im = Image.fromarray(input_array).convert("RGB")
        im.save(os.path.join(video_dir, f"{slice_idx}.jpg"))


def prompt_based_prediction_sam3_image_style_combination(predictor, inference_state,
                                                         prompt_points: np.ndarray, prompts_bbox: np.ndarray) -> \
        Optional[np.ndarray]:
    prediction: Optional[np.ndarray] = None
    if len(prompt_points) == 0 and len(prompts_bbox) == 0:  # no prompt
        return prediction
    elif len(prompts_bbox) > 0 and len(prompt_points) > 0:  # both bounding box and point
        pc: np.ndarray = prompt_points[:, :2]
        pl: np.ndarray = prompt_points[:, -1]
        for box in prompts_bbox:
            preds_single, scores, _ = predictor.predict_inst(
                inference_state,
                point_coords=pc,
                point_labels=pl,
                box=box,
                multimask_output=False,
            )
            # preds_single, scores, _ = predictor.predict(point_coords=pc, point_labels=pl, box=box,
            #                                             multimask_output=False)
            if prediction is None:
                prediction = preds_single
            else:
                prediction += preds_single
    elif len(prompts_bbox) == 0 and len(prompt_points) > 0:  # only point
        pc: np.ndarray = prompt_points[:, :2]
        pl: np.ndarray = prompt_points[:, -1]
        prediction, scores, _ = predictor.predict_inst(
            inference_state,
            point_coords=pc,
            point_labels=pl,
            box=None,
            multimask_output=False,
        )
        # prediction, scores, _ = predictor.predict(point_coords=pc, point_labels=pl, box=None,
        #                                           multimask_output=False)
    elif len(prompts_bbox) > 0 and len(prompt_points) == 0:  # only bounding box
        for box in prompts_bbox:
            preds_single, scores, _ = predictor.predict_inst(
                inference_state,
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False,
            )
            # preds_single, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=box,
            #                                             multimask_output=False)
            if prediction is None:
                prediction = preds_single
            else:
                prediction += preds_single
    # filter which mask to use -> default: take the first output
    prediction = prediction[0]  # take the first output -> no score_filtering or multi-mask output
    prediction = np.array(prediction > 0, dtype=np.uint8)
    return prediction


def prompt_based_prediction_sam3_video_style_combination(predictor, inference_state,
                                                         prompt_points: np.ndarray, prompt_bbox: np.ndarray,
                                                         img_width: int, img_height: int,
                                                         label: int, reverse: Optional[bool] = None,
                                                         start_idx: Optional[int] = None,
                                                         max_frame_numbers: Optional[int] = None):
    predictor.clear_all_points_in_video(inference_state)
    video_segments: dict[int: np.ndarray] = {}
    if len(prompt_points) == 0 and len(prompt_bbox) == 0:  # no prompt
        return video_segments
    elif len(prompt_points) > 0 and len(prompt_bbox) > 0:  # both bounding box and point
        point_coordinates, point_labels, point_frames = transform_3dpoint_into_components(prompt_points)
        box_coordinates, box_frames = transform_3dbox_into_components(prompt_bbox)
        rel_points = {k: [[x / img_width, y / img_height] for x, y in v] for k, v in point_coordinates.items()}
        rel_box = {k: [xmin / img_width, ymin / img_height, xmax / img_width, ymax / img_height] for k, v in
                   box_coordinates.items() for xmin, ymin, xmax, ymax in v}
        assert point_frames == box_frames, "init frames for box and point need to be the same"
        assert len(box_frames) > 0, "init needs at least one frame"
        for init_frame in point_frames:
            assert len(box_coordinates[init_frame]) <= 1, "not more than one box allowed"
            _, out_obj_ids, low_res_masks, video_res_masks = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=init_frame,
                obj_id=label,
                points=torch.tensor(rel_points[init_frame], dtype=torch.float32),
                labels=torch.tensor(point_labels[init_frame], dtype=torch.int32),
                box=np.array(rel_box[init_frame], dtype=np.float32),
            )
    elif len(prompt_bbox) == 0 and len(prompt_points) > 0:  # only point
        point_coordinates, point_labels, frames = transform_3dpoint_into_components(prompt_points)
        rel_points = {k: [[x / img_width, y / img_height] for x, y in v] for k, v in point_coordinates.items()}
        for init_frame in frames:
            _, out_obj_ids, low_res_masks, video_res_masks = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=init_frame,
                obj_id=label,
                points=torch.tensor(rel_points[init_frame], dtype=torch.float32),
                labels=torch.tensor(point_labels[init_frame], dtype=torch.int32),
            )
    elif len(prompt_bbox) > 0 and len(prompt_points) == 0:  # only bbox
        box_coordinates, frames = transform_3dbox_into_components(prompt_bbox)
        rel_box = {
            k: [[xmin / img_width, ymin / img_height, xmax / img_width, ymax / img_height] for xmin, ymin, xmax, ymax in
                v] for k, v in box_coordinates.items()}
        for init_frame in frames:
            assert len(box_coordinates[init_frame]) <= 1, "not more than one box allowed"
            _, out_obj_ids, low_res_masks, video_res_masks = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=init_frame,
                obj_id=label,
                box=np.array(rel_box[init_frame], dtype=np.float32),
            )

    # propagate through volume
    for out_frame_idx, out_obj_ids, low_res_masks, out_mask_logits, obj_scores in predictor.propagate_in_video(
            inference_state,
            reverse=reverse,
            start_frame_idx=start_idx,
            max_frame_num_to_track=max_frame_numbers,
            propagate_preflight=True):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    return video_segments
