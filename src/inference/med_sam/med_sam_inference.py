########################################################################################################################
# Inference script for Med-SAM
# slice-by-slice inference of 1024x1024 input
# prompts are given as 2D prompts (bounding boxes) stored in json file
# settings: number of points
# masks are stored one-hot-encoded (pixel value = label value), stored per class (since Med-SAM has single-class outputs)
########################################################################################################################

import argparse
import json
import os
import time
from typing import Union, Optional

import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm

from src.inference.med_sam.med_sam_utils import create_medsam_model, extract_and_process_slice_like_medsam, \
    transform_bbox_medsam, bbox_based_prediction_medsam_style
from src.inference.utils_2dprompts import extract_all_slices_with_prompts2d, extract_original_prompt2d, \
    extract_all_slices_with_prompts_to_be_used
from src.inference.utils_filehandling import read_image_depth_first, write_output_masks_to_nii


def run_inference_med_sam(json_file: Union[Path, str], output_folder: Union[Path, str], number_prompts: int,
                          nnunet_suffix: bool = True, debug: bool = False):
    # read json file with prompts
    with open(json_file, "r") as f:
        data_prompt: dict = json.load(f)

    # get meta data
    file_ending: str = data_prompt["file_ending"]
    if nnunet_suffix:
        file_id_suffix = "_0000"
    else:
        file_id_suffix = ""
    path_images: Path = Path(data_prompt["image_path"])

    # get label information (lookup table, names and values)
    path_dataset_json = path_images.parent / "dataset.json"
    if not path_dataset_json.exists(): raise FileExistsError("dataset.json file does not exist. needs to be stored in ",
                                                             path_images.parent)
    with open(path_dataset_json, "r") as f:
        dataset: dict = json.load(f)
    labels_lookup: dict[str, Union[str, int]] = dataset["labels"]
    label_names: list[str] = [str(x) for x in labels_lookup.keys() if "background" not in x and "bg" not in x]
    labels_values: list[int] = [int(x) for x in data_prompt["labels"]]

    # create output folder if not existing
    output_dir: Path = Path(output_folder)
    if not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)
    for label_name in label_names:
        os.makedirs(output_dir / label_name, exist_ok=True)

    # get file ids
    file_ids: list[str] = [x for x in data_prompt.keys() if
                           x not in ["file_ending", "labels_path", "image_path", "labels"]]
    if debug:
        file_ids = file_ids[:1]

    # initialize model
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = create_medsam_model(device)

    print("")
    print(f"Med-SAM inference")

    # set up prompt strategy for experiment
    prompts_to_be_used: list[str] = ["bbox"]
    number_of_prompts: int = int(number_prompts)
    print(f"experiment setting: use {prompts_to_be_used} from up to {number_of_prompts} connected components.")

    # meta data json
    all_meta_data = {"model": "Med-SAM", "model_type": "vit_b",
                     "prompt": prompts_to_be_used, "json_file": str(json_file),
                     "output_folder": str(output_folder), "number_of_prompts": number_of_prompts,
                     "label_order": labels_values}

    # iterate through samples
    for file_id in file_ids:
        print(f"process {file_id}")

        # read image file
        input_image, affine = read_image_depth_first(path_images, file_id, file_ending, file_id_suffix)

        # prepare prediction
        output_msk: np.ndarray = np.zeros((len(labels_values), *input_image.shape), dtype=np.uint8)

        # iterate through slices (more efficient for inference)
        slice_meta_data = {}
        all_slices_with_prompts = extract_all_slices_with_prompts2d(data_prompt, file_id,
                                                                    prompts_to_be_used,
                                                                    labels_values)
        for slice_idx in tqdm(range(len(input_image)), desc="Slices"):
            # at least one prompt for this slice is available -> otherwise continue
            if str(slice_idx) in all_slices_with_prompts:
                start_time: float = time.time()
                input_array, H, W = extract_and_process_slice_like_medsam(input_image, slice_idx)
                with torch.no_grad():
                    image_embedding: torch.Tensor = predictor.image_encoder(input_array)  # (1, 256, 64, 64)

                # iterate through class labels
                classes_used: list[int] = []
                for idx, label in enumerate(labels_values):
                    prompts_2d: dict[str, dict] = data_prompt[file_id][str(label)]["2d_prompts"]
                    slices_with_prompts: list[str] = extract_all_slices_with_prompts_to_be_used(prompts_2d,
                                                                                                prompts_to_be_used)
                    # at least one prompt for this slice and label is available -> otherwise continue
                    if str(slice_idx) in slices_with_prompts:
                        input_prompts_bbox: np.ndarray = extract_original_prompt2d(prompts_2d, prompts_to_be_used[0],
                                                                                   slice_idx, number_of_prompts)
                        if len(input_prompts_bbox) > 0:
                            classes_used.append(label)
                            bbox_list: list[np.ndarray] = transform_bbox_medsam(input_prompts_bbox, H, W)

                            # create predictions
                            preds_msk: Optional[np.ndarray] = None
                            for box_1024 in bbox_list:
                                preds_single: np.ndarray = bbox_based_prediction_medsam_style(predictor,
                                                                                              image_embedding,
                                                                                              box_1024, H, W)
                                if preds_msk is None:
                                    preds_msk = preds_single
                                else:
                                    preds_msk += preds_single
                            if preds_msk is not None:
                                output_msk[idx][slice_idx][preds_msk > 0] = label

                elapsed_time: float = time.time() - start_time
                slice_meta_data[slice_idx] = {"time": elapsed_time, "classes": classes_used}

        write_output_masks_to_nii(output_msk, labels_lookup, output_dir, file_id, file_ending, affine, verbose=True)
        slice_meta_data["time_total"] = np.sum([float(slice_meta_data[k]["time"]) for k in slice_meta_data.keys()])
        all_meta_data[file_id] = slice_meta_data

    meta_data_json_file = output_dir / ("meta_data_combination_" + prompts_to_be_used[0] + ".json")
    with open(meta_data_json_file, "w") as file:
        json.dump(all_meta_data, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Med-SAM inference with bbox prompt from json file")
    parser.add_argument("--json_file", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--number_prompts", default=1, required=False)
    args = parser.parse_args()

    run_inference_med_sam(json_file=args.json_file,
                          output_folder=args.output_folder,
                          number_prompts=args.number_prompts)
