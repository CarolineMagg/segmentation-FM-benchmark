########################################################################################################################
# Inference script for MedicoSAM in 2D mode
# slice-by-slice inference of 1024x1024 input
# prompts are given as 2D prompts (points or bounding boxes) stored in json file
# settings: prompt type, number of points, number of random points, model_type (default: vit_b_medical_imaging)
# masks are stored one-hot-encoded (pixel value = label value), stored per class
########################################################################################################################
import argparse
import json
import os
import time
from pathlib import Path
from typing import Union, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.inference.medico_sam.medico_sam_utils import create_medicosam_predictor, \
    prompt_based_prediction_medicosam_style_combination
from src.inference.sam.sam_utils import extract_and_process_slice_like_sam
from src.inference.utils_2dprompts import extract_all_slices_with_prompts2d, extract_all_slices_with_prompts_to_be_used, \
    extract_original_prompt2d_combination
from src.inference.utils_filehandling import read_image_depth_first, write_output_masks_to_nii


def run_inference_medicosam2d(json_file: Union[Path, str], output_folder: Union[Path, str], prompt_type: list[str],
                              number_prompts: int, number_random_prompts: int,
                              model_type: str = "vit_b_medical_imaging",
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
    predictor = create_medicosam_predictor(device, model_type)

    print("")
    print(f"MedicoSAM2D inference with {model_type}")

    # set up prompt strategy for experiment
    prompts_to_be_used: list[str] = prompt_type
    number_of_prompts: int = int(number_prompts)
    number_random_prompts: int = int(number_random_prompts)
    print(f"experiment setting: use {prompts_to_be_used} from up to {number_of_prompts} connected components.")
    if "random" in prompts_to_be_used:
        print(f"{number_random_prompts} random points will be used.")
    if "negative" in prompts_to_be_used:
        print(f"{number_random_prompts} negative random points will be used.")

    # meta data json
    all_meta_data = {"model": "MedicoSAM2D", "model_type": model_type,
                     "prompt": prompts_to_be_used, "json_file": str(json_file),
                     "output_folder": str(output_folder), "number_of_prompts": number_of_prompts,
                     "random_number_prompts": number_random_prompts, "label_order": labels_values}

    # iterate through samples
    for file_id in file_ids:
        print(f"process {file_id}")

        # read image file
        # https://github.com/anwai98/tukra/blob/master/tukra/io/util.py#L46
        input_image, affine = read_image_depth_first(path_images, file_id, file_ending, file_id_suffix)

        # prepare prediction
        output_msk: np.ndarray = np.zeros((len(labels_values), *input_image.shape), dtype=np.uint8)

        # iterate through slices (more efficient for inference)
        slice_meta_data: dict = {}
        all_slices_with_prompts = extract_all_slices_with_prompts2d(data_prompt, file_id, prompts_to_be_used,
                                                                    labels_values)
        for slice_idx in tqdm(range(len(input_image)), desc="Slices"):
            # at least one prompt for this slice is available -> otherwise continue
            if str(slice_idx) in all_slices_with_prompts:
                start_time: float = time.time()  # track time per slice (for all label classes)
                input_array: np.ndarray = extract_and_process_slice_like_sam(input_image, slice_idx)

                # iterate through class labels
                classes_used: list[int] = []
                for idx, label in enumerate(labels_values):
                    prompts_2d: dict[str, dict] = data_prompt[file_id][str(label)]["2d_prompts"]
                    slices_with_prompts: list[str] = extract_all_slices_with_prompts_to_be_used(prompts_2d,
                                                                                                prompts_to_be_used)
                    # at least one prompt for this slice and label is available -> otherwise continue
                    if str(slice_idx) in slices_with_prompts:
                        input_prompt_points, input_prompt_bbox = extract_original_prompt2d_combination(prompts_2d,
                                                                                                       prompts_to_be_used,
                                                                                                       slice_idx,
                                                                                                       number_of_prompts,
                                                                                                       number_random_prompts)

                        # if prompts are not empty
                        if len(input_prompt_points) > 0 or len(input_prompt_bbox) > 0:
                            classes_used.append(label)
                            preds_msk: Optional[np.ndarray] = prompt_based_prediction_medicosam_style_combination(
                                predictor,
                                input_array,
                                input_prompt_points,
                                input_prompt_bbox)

                            # if prediction mask is not empty
                            if preds_msk is not None:
                                output_msk[idx][slice_idx][preds_msk > 0] = label

                    elapsed_time: float = time.time() - start_time
                    slice_meta_data[slice_idx] = {"time": elapsed_time, "classes": classes_used}

        # convert dict to segmentation mask & store
        write_output_masks_to_nii(output_msk, labels_lookup, output_dir, file_id, file_ending,
                                  affine, verbose=True)
        slice_meta_data["time_total"] = np.sum([float(slice_meta_data[k]["time"]) for k in slice_meta_data.keys()])
        all_meta_data[file_id] = slice_meta_data

    meta_data_json_file = output_dir / ("meta_data_combination_" + "_".join(prompts_to_be_used) + ".json")
    with open(meta_data_json_file, "w") as file:
        json.dump(all_meta_data, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MedicoSAM2D inference for combination of prompts from json file")
    parser.add_argument("--json_file", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--prompt_type", default=["bbox", "center"], nargs="*",
                        choices=["center", "centroid", "random", "bbox", "negative"])
    parser.add_argument("--number_prompts", default=1, required=False)
    parser.add_argument("--random_number_prompts", default=1, required=False)
    parser.add_argument("--model_type", default="vit_b_medical_imaging")
    args = parser.parse_args()

    run_inference_medicosam2d(json_file=args.json_file,
                              output_folder=args.output_folder,
                              prompt_type=args.prompt_type,
                              number_prompts=args.number_prompts,
                              number_random_prompts=args.random_number_prompts,
                              model_type=args.model_type)
