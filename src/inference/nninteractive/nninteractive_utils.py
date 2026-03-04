import os
import sys

import torch
import numpy as np

from src.inference.utils_3dprompts import reorder_point_coordinates
from src.project_root import PROJECT_ROOT

NNINTERACTIVE_MODULE_PATH = PROJECT_ROOT / "submodules" / "nnInteractive"  # Get the absolute path
if str(NNINTERACTIVE_MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(NNINTERACTIVE_MODULE_PATH))

from submodules.nnInteractive.nnInteractive.inference.inference_session import nnInteractiveInferenceSession


def create_nninteractive_predictor(device: torch.device):
    session = nnInteractiveInferenceSession(
        device=device,  # Set inference device
        use_torch_compile=False,  # Experimental: Not tested yet
        verbose=False,
        torch_n_threads=os.cpu_count(),  # Use available CPU cores
        do_autozoom=True,  # Enables AutoZoom for better patching
        use_pinned_memory=True,  # Optimizes GPU memory transfers
    )
    model_path = PROJECT_ROOT / "checkpoints" / "nninteractive" / "nnInteractive_v1.0"
    session.initialize_from_trained_model_folder(model_path)
    return session


def prompt_based_prediction_nninterative_style_combination(session, prompt_points: np.ndarray,
                                                           prompts_bbox: np.ndarray, ):
    # https://github.com/MIC-DKFZ/nnInteractive?tab=readme-ov-file#getting-started
    pos_points = []
    neg_points = []
    boxes = []

    if len(prompt_points) == 0 and len(prompts_bbox) == 0:  # no prompt
        return None
    elif len(prompts_bbox) > 0 and len(prompt_points) > 0:  # both bounding box and point
        # point convention: tuple (x, y, z)
        pc: np.ndarray = prompt_points[:, :3]
        pl: np.ndarray = prompt_points[:, -1]
        pos_points = pc[pl == 1]
        neg_points = pc[pl == 0]
        # box convention : [[x1, x2], [y1, y2], [z1, z1 + 1]]
        boxes = np.einsum("nij->nji", prompts_bbox.reshape(-1, 2, 3))
        boxes[:, 2, 1] += 1
    elif len(prompts_bbox) == 0 and len(prompt_points) > 0:  # only point
        pc: np.ndarray = prompt_points[:, :3]
        pl: np.ndarray = prompt_points[:, -1]
        pos_points = pc[pl == 1]
        neg_points = pc[pl == 0]
    elif len(prompts_bbox) > 0 and len(prompt_points) == 0:  # only bounding box
        boxes: np.ndarray = np.einsum("nij->nji", prompts_bbox.reshape(-1, 2, 3))
        boxes[:, 2, 1] += 1

    if len(pos_points) > 0:
        pos_points = reorder_point_coordinates(pos_points)  # (z, y, x)
        for pos_point in pos_points:
            session.add_point_interaction(pos_point, include_interaction=True)
    if len(neg_points) > 0:
        neg_points = reorder_point_coordinates(neg_points)  # (z, y, x)
        for neg_points in neg_points:
            session.add_point_interaction(neg_points, include_interaction=True)
    if len(boxes) > 0:
        for box in boxes:
            box = box[[2, 1, 0], :]
            session.add_bbox_interaction(box, include_interaction=True)

    results = session.target_buffer.clone()
    return results
