import sys
from typing import Optional

import numpy as np
import torch

from src.inference.utils_3dprompts import transform_3dpoint_into_components, transform_3dbox_into_components
from src.project_root import PROJECT_ROOT

MEDICOSAM_MODULE_PATH = PROJECT_ROOT / "submodules" / "MedicoSAM"  # Get the absolute path
if str(MEDICOSAM_MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(MEDICOSAM_MODULE_PATH))

from micro_sam.util import get_sam_model
from micro_sam.multi_dimensional_segmentation import segment_mask_in_volume
from micro_sam.inference import batched_inference


def create_medicosam_predictor(device: torch.device, model_type: str = "vit_b_medical_imaging"):
    checkpoint = PROJECT_ROOT / "checkpoints" / "medico_sam" / "vit_b.pt"
    if not checkpoint.exists():
        raise FileExistsError(
            f"The checkpoint file {checkpoint} does not exist. Make sure the checkpoint file is downloaded into the correct folder.")
    predictor = get_sam_model(model_type=model_type,
                              device=device,
                              checkpoint_path=checkpoint)
    return predictor


def prompt_based_prediction_medicosam_style_combination(predictor, image: np.ndarray, prompt_points: np.ndarray,
                                                        prompts_bbox: np.ndarray):
    # https://github.com/computational-cell-analytics/micro-sam/blob/master/micro_sam/evaluation/inference.py#L94
    prediction: Optional[np.ndarray] = None
    boxes = None
    points = None
    point_labels = None
    if len(prompt_points) == 0 and len(prompts_bbox) == 0:  # no prompt
        return prediction
    elif len(prompts_bbox) > 0 and len(prompt_points) > 0:  # both bounding box and point
        points: np.ndarray = prompt_points[:, :2][:, np.newaxis, :]
        point_labels: np.ndarray = prompt_points[:, -1][..., np.newaxis]
        boxes = prompts_bbox
    elif len(prompts_bbox) == 0 and len(prompt_points) > 0:  # only point
        points: np.ndarray = prompt_points[:, :2][:, np.newaxis, :]
        point_labels: np.ndarray = prompt_points[:, -1][..., np.newaxis]
    elif len(prompts_bbox) > 0 and len(prompt_points) == 0:  # only bounding box
        boxes = prompts_bbox

    if boxes is not None and points is not None and len(boxes) != len(points):
        # print(boxes, points)
        if len(boxes) < len(points):
            points = points[:len(boxes)]
            point_labels = point_labels[:len(boxes)]
        else:
            boxes = boxes[:len(points)]
        # print(boxes, points)

    # # Use multi-masking only if we have a single positive point without box
    multimasking = False
    n_positives = 0
    n_negatives = 0
    if len(prompt_points) > 0:
        n_positives: int = len(points[0][point_labels[0] == 1])
        n_negatives: int = len(points[0][point_labels[0] == 0])
    if len(prompts_bbox) == 0 and (n_positives == 1 and n_negatives == 0):
        multimasking = True

    batch_size = 32
    prediction = batched_inference(
        predictor, image, batch_size,
        boxes=boxes, points=points, point_labels=point_labels,
        multimasking=multimasking, embedding_path=None,
        return_instance_segmentation=True, verbose_embeddings=False
    )
    return prediction  # (H, W)


def prompt_based_prediction_medicosam3d_style_combination(predictor, image: np.ndarray, embeddings: dict,
                                                          prompt_points: np.ndarray, prompt_bbox: np.ndarray):
    # https://github.com/computational-cell-analytics/micro-sam/blob/master/micro_sam/evaluation/multi_dimensional_segmentation.py#L59
    # extract slice_choice from 3d prompt
    if len(prompt_points) == 0 and len(prompt_bbox) == 0:  # no prompt
        return None
    elif len(prompt_points) > 0 and len(prompt_bbox) > 0:  # both bounding box and point
        point_coordinates, point_labels, point_frames = transform_3dpoint_into_components(prompt_points)
        box_coordinates, box_frames = transform_3dbox_into_components(prompt_bbox)
        assert point_frames == box_frames, "init frames for box and point need to be the same"
        # assert len(box_frames) > 0, "init needs at least one frame"
        frames = box_frames
    elif len(prompt_bbox) == 0 and len(prompt_points) > 0:  # only point
        point_coordinates, point_labels, frames = transform_3dpoint_into_components(prompt_points)
    elif len(prompt_bbox) > 0 and len(prompt_points) == 0:  # only bbox
        box_coordinates, frames = transform_3dbox_into_components(prompt_bbox)
    else:
        raise RuntimeError

    assert len(frames) == 1, f"only one init frame is allowed, not {len(frames)}"
    slice_choice = frames[0]
    if len(prompt_bbox) > 0:
        prompt_bbox = box_coordinates[slice_choice]

    # run 2d prediction
    output_slice = prompt_based_prediction_medicosam_style_combination(predictor, image[slice_choice], prompt_points,
                                                                       prompt_bbox)
    output_seg = np.zeros_like(image, dtype=output_slice.dtype)
    output_seg[slice_choice][output_slice == 1] = 1

    # Segment the object in the entire volume with the specified segmented slice
    prediction, _ = segment_mask_in_volume(
        segmentation=output_seg,
        predictor=predictor,
        image_embeddings=embeddings,
        segmented_slices=np.array(slice_choice),
        stop_lower=False, stop_upper=False,
        iou_threshold=0.7,
        projection="single_point",
        box_extension=0.0,
        verbose=False,
    )

    return prediction
