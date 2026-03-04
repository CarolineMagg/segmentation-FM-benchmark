########################################################################################################################
# Inference script for Med-SAM2
# slice-by-slice inference with memory bank and propagation of 1024x1024 input
# prompts are given as 2D prompts in initial frames (points or 1 bounding box) stored in json file
# settings: prompt type, prompt mode (2D or 3D), number of points, number of random points,
#           selection for initial frame, number of random frames,
#           number of additional frames, gap between additional frames, flag for equally distributed random frames
#           flag to use volume limits (top and bottom slice of object)
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

from src.inference.med_sam2.med_sam2_utils import create_med_sam2_predictor, process_volume_like_med_sam2, \
    prompt_based_prediction_med_sam2_video_style_combination
from src.inference.utils_3dprompts import convert_list_of_dicts_to_mask, extract_original_3d_prompt, get_min_max_frames, \
    get_frames_for_propagation
from src.inference.utils_filehandling import read_image_depth_first, write_output_masks_to_nii


def run_inference_med_sam2(json_file: Union[Path, str], output_folder: Union[Path, str], prompt_type: list[str],
                           prompt_mode: str, number_prompts: int, number_random_prompts: int,
                           initial_frame_selection: list[str], number_additional_frames: int, gap_between_frames: int,
                           use_volume_limits: bool = False, model_type: str = "MedSAM2_latest",
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = create_med_sam2_predictor(device)

    print("")
    print(f"Med-SAM2 video inference with {model_type}")

    # set up prompt strategy for experiment
    prompts_to_be_used: list[str] = prompt_type
    number_of_prompts: int = int(number_prompts)
    number_random_prompts: int = int(number_random_prompts)
    print(f"experiment setting: use {prompts_to_be_used} from up to {number_of_prompts} connected components.")
    if "random" in prompts_to_be_used:
        print(f"{number_random_prompts} random points will be used.")
    if "negative" in prompts_to_be_used:
        print(f"{number_random_prompts} negative random points will be used.")
    if number_additional_frames == 0 and gap_between_frames == 0:
        print(f"the initial frames will be {initial_frame_selection}.")
    else:
        print(
            f"the initial frames will be {initial_frame_selection} with {number_additional_frames} additional frames with {gap_between_frames} frames inbetween.")
    print(f"propagation will use volume limits: {use_volume_limits}")

    # meta data json
    all_meta_data = {"model": "Med-SAM2 video", "model_type": model_type,
                     "prompt": prompts_to_be_used, "prompt_mode": prompt_mode, "json_file": str(json_file),
                     "output_folder": str(output_folder), "number_of_prompts": number_of_prompts,
                     "random_number_prompts": number_random_prompts, "label_order": labels_values,
                     "initial_frame_selection": initial_frame_selection, "number_of_frames": number_additional_frames,
                     "gap_between_frames": gap_between_frames, "use_volume_limits": use_volume_limits}

    # iterate through samples
    for file_id in file_ids:
        print(f"process {file_id}")

        # read image file
        input_image, affine = read_image_depth_first(path_images, file_id, file_ending, file_id_suffix)
        img_resized: torch.Tensor = process_volume_like_med_sam2(input_image)
        D, H, W = input_image.shape

        # set init state
        inference_state = predictor.init_state(img_resized, H, W)

        # set prompts separate for each class label & run propagation for each class separately
        sample_meta_data_dict = {}
        areas_per_class = None
        volume_segments_all: list[dict[int: np.ndarray]] = []
        for idx, label in enumerate(labels_values):
            label_name: str = next(key for key, value in labels_lookup.items() if value == str(label))
            prompts = data_prompt[file_id][str(label)]

            output = extract_original_3d_prompt(prompts=prompts,
                                                prompt_mode=prompt_mode,
                                                prompts_to_be_used=prompts_to_be_used,
                                                label=label,
                                                number_of_prompts=number_of_prompts,
                                                number_random_points=number_random_prompts,
                                                areas_per_class=areas_per_class,
                                                labels_values=labels_values,
                                                data_prompt=data_prompt[file_id],
                                                initial_frame_selection=initial_frame_selection,
                                                gap_between_frames=gap_between_frames,
                                                number_additional_frames=number_additional_frames)
            input_point_original, input_bbox_original, areas_per_class = output

            start_frame_idx: list[Optional[int]] = [None] * 2
            max_frame_num_to_track: list[Optional[int]] = [None] * 2
            if use_volume_limits and (len(input_point_original) > 0 or len(input_bbox_original) > 0):
                min_max_frames = get_min_max_frames(prompts=prompts,
                                                    prompts_to_be_used=prompts_to_be_used,
                                                    prompt_mode=prompt_mode,
                                                    areas_per_class=areas_per_class, label=label)
                # define volume borders (for bidirectional: first revers=False, then True)
                start_frame_idx, max_frame_num_to_track = get_frames_for_propagation(input_point_original,
                                                                                     input_bbox_original,
                                                                                     min_max_frames)

            # prediction
            start_time: float = time.time()
            volume_segments: dict[int: np.ndarray] = {}
            for reverse, s, f_n in zip([False, True], start_frame_idx, max_frame_num_to_track):
                tmp = prompt_based_prediction_med_sam2_video_style_combination(predictor, inference_state,
                                                                               input_point_original,
                                                                               input_bbox_original,
                                                                               label, reverse=reverse,
                                                                               start_idx=s, max_frame_numbers=f_n)
                volume_segments.update(tmp)

            elapsed_time: float = time.time() - start_time
            sample_meta_data_dict[str(label_name)] = {"time": elapsed_time, "classes": int(label)}

            volume_segments_all.append(volume_segments)

        # convert dict to segmentation mask & store
        output_msk: np.ndarray = convert_list_of_dicts_to_mask(volume_segments_all, labels_values, input_image.shape)
        write_output_masks_to_nii(output_msk, labels_lookup, output_dir, file_id, file_ending, affine, verbose=True)

        sample_meta_data_dict["time_total"] = np.sum(
            [float(sample_meta_data_dict[k]["time"]) for k in sample_meta_data_dict.keys()])
        all_meta_data[file_id] = sample_meta_data_dict

    with open(os.path.join(output_dir, "meta_data_combination_" + "_".join(prompts_to_be_used) + ".json"),
              "w") as file:
        json.dump(all_meta_data, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Med-SAM2 inference for combination of prompts from json file")
    parser.add_argument("--json_file", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--prompt_type", default=["bbox", "center"], nargs="*",
                        choices=["center", "centroid", "random", "bbox", "negative", "bbox", "center"])
    parser.add_argument("--prompt_mode", default="3d_prompts", required=False)
    parser.add_argument("--number_prompts", default=1, required=False)
    parser.add_argument("--random_number_prompts", default=1, required=False)
    parser.add_argument("--initial_frame_selection", default=["none"], nargs="*",
                        choices=["center", "largest", "first", "last", "random", "none"])
    parser.add_argument("--number_additional_frames", default=0, required=False)
    parser.add_argument("--gap_between_frames", default=1, required=False)
    parser.add_argument("--use_volume_limits", action='store_true')
    parser.add_argument("--model_type", default="sam2_new_hiera_base_plus",
                        choices=["MedSAM2_latest", "MedSAM2_2411"])
    args = parser.parse_args()

    run_inference_med_sam2(json_file=args.json_file,
                           output_folder=args.output_folder,
                           prompt_type=args.prompt_type,
                           prompt_mode=args.prompt_mode,
                           number_prompts=args.number_prompts,
                           number_random_prompts=args.random_number_prompts,
                           initial_frame_selection=args.initial_frame_selection,
                           number_additional_frames=args.number_additiona_frames,
                           gap_between_frames=args.gap_between_frames,
                           use_volume_limits=args.use_volume_limits,
                           model_type=args.model_type)
