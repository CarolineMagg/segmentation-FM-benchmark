########################################################################################################################
# Inference script for SegVol
# full-resolution inference with zoom-in-zoom-out mechanism (32x256x256)
# prompts are given as 3D points (points and bounding box) stored in json file
# settings: prompt type, prompt mode (2D or 3D), number of points, number of random points,
#           selection for initial frame, number of random frames,
#           number of additional frames, gap between additional frames, flag for equally distributed random frames
#           flag to use zoom-in-zoom-out mechanism (default: True)
# masks are stored one-hot-encoded (pixel value = label value), stored per class
########################################################################################################################
import argparse
import json
import os
import time
from pathlib import Path
from typing import Union

import numpy as np
import torch

from src.inference.seg_vol.seg_vol_utils import create_seg_vol_predictor, process_volume_like_segvol, \
    prompt_based_prediction_segvol_style_combination, inverse_process_volume_like_segvol
from src.inference.utils_3dprompts import extract_original_3d_prompt, crop_3dpoint_to_foreground, \
    crop_3dbox_to_foreground, transform_3dpoint_resize, \
    transform_3dbox_resize, convert_list_of_masks_to_mask
from src.inference.utils_filehandling import read_image_depth_first, write_output_masks_to_nii


def run_inference_segvol(json_file: Union[Path, str], output_folder: Union[Path, str], prompt_type: list[str],
                         prompt_mode: str, number_prompts: int, number_random_prompts: int,
                         initial_frame_selection: list[str], number_additional_frames: int, gap_between_frames: int,
                         use_zoom_in: bool = True, nnunet_suffix: bool = True, debug: bool = False):
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
    spatial_size = (32, 256, 256)
    patch_size = (4, 16, 16)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = create_seg_vol_predictor(device, spatial_size, patch_size)

    print("")
    print(f"SegVol inference")

    # set up prompt strategy for experiment
    prompts_to_be_used: list[str] = prompt_type
    number_of_prompts: int = int(number_prompts)
    number_random_prompts: int = int(number_random_prompts)
    print(f"experiment setting: use {prompts_to_be_used} from up to {number_of_prompts} connected components.")
    print(f"the prompt mode is {prompt_mode}")
    if "random" in prompts_to_be_used:
        print(f"{number_random_prompts} random points will be used.")
    if "negative" in prompts_to_be_used:
        print(f"{number_random_prompts} negative random points will be used.")
    if number_additional_frames == 0 and gap_between_frames == 0:
        print(f"the initial frames will be {initial_frame_selection}.")
    else:
        print(
            f"the initial frames will be {initial_frame_selection} with {number_additional_frames} additional frames with {gap_between_frames} frames inbetween.")
    print(f"zoom-in-zoom-out mechanism will be used: {use_zoom_in}")

    if prompt_mode == "2d_prompting" and "bbox" in prompts_to_be_used:
        raise RuntimeError("2d bbox is not supported")
    if len(prompts_to_be_used) >= 2 and "bbox" in prompts_to_be_used:
        raise RuntimeError("combination with bbox is not supported")

    # meta data json
    all_meta_data = {"model": "SegVol", "model_type": "default",
                     "prompt": prompts_to_be_used, "json_file": str(json_file),
                     "output_folder": str(output_folder), "number_of_prompts": number_of_prompts,
                     "random_number_prompts": number_random_prompts, "label_order": labels_values,
                     "initial_frame_selection": initial_frame_selection, "number_of_frames": number_additional_frames,
                     "gap_between_frames": gap_between_frames, "use_zoom_in": use_zoom_in}

    # iterate through samples
    for file_id in file_ids:
        print(f"process {file_id}")

        # read image file
        input_image, affine = read_image_depth_first(path_images, file_id, file_ending, file_id_suffix)
        data_item = process_volume_like_segvol(input_image, spatial_size)
        input_tensor = data_item["image"]
        input_tensor_zoom_out = data_item["zoom_out_image"]
        foreground_start_coord = data_item["foreground_start_coord"]
        foreground_end_coord = data_item["foreground_end_coord"]

        # set prompts separate for each class label & run propagation for each class separately
        sample_meta_data_dict = {}
        volume_segments_all = []
        areas_per_class = None
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
            img_shape = input_tensor.shape
            img_zoom_shape = input_tensor_zoom_out.shape
            input_point_original3d = crop_3dpoint_to_foreground(input_point_original, foreground_start_coord, img_shape)
            input_bbox_original3d = crop_3dbox_to_foreground(input_bbox_original, foreground_start_coord, img_shape)
            input_point_zoom3d = transform_3dpoint_resize(input_point_original3d, img_shape, img_zoom_shape)
            input_box_zoom3d = transform_3dbox_resize(input_bbox_original3d, img_shape, img_zoom_shape)

            # prediction
            start_time: float = time.time()
            volume_segments, _ = prompt_based_prediction_segvol_style_combination(predictor, input_tensor.unsqueeze(0),
                                                                                  input_tensor_zoom_out.unsqueeze(0),
                                                                                  input_point_zoom3d,
                                                                                  input_box_zoom3d,
                                                                                  spatial_size, use_zoom_in)

            elapsed_time: float = time.time() - start_time
            sample_meta_data_dict[str(label_name)] = {"time": elapsed_time, "classes": int(label)}

            volume_segments_all.append(volume_segments)

        # convert dict to segmentation mask & store
        output_msk = convert_list_of_masks_to_mask(volume_segments_all, labels_values, input_tensor.shape[1:])
        output_msk = inverse_process_volume_like_segvol(output_msk, foreground_start_coord, foreground_end_coord,
                                                        input_image.shape, spatial_size)
        write_output_masks_to_nii(output_msk, labels_lookup, output_dir, file_id, file_ending, affine, verbose=True)
        sample_meta_data_dict["time_total"] = np.sum(
            [float(sample_meta_data_dict[k]["time"]) for k in sample_meta_data_dict.keys()])
        all_meta_data[file_id] = sample_meta_data_dict

    meta_data_json_file = output_dir / ("meta_data_combination_" + "_".join(prompts_to_be_used) + ".json")
    with open(meta_data_json_file, "w") as file:
        json.dump(all_meta_data, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SegVol inference for combination of prompts from json file")
    parser.add_argument("--json_file", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--prompt_type", default=["bbox", "center"], nargs="*",
                        choices=["center", "centroid", "random", "bbox", "negative"])
    parser.add_argument("--prompt_mode", default="3d_prompts", required=False)
    parser.add_argument("--number_prompts", default=1, required=False)
    parser.add_argument("--random_number_prompts", default=1, required=False)
    parser.add_argument("--initial_frame_selection", default=["none"], nargs="*",
                        choices=["center", "largest", "first", "last", "random", "none"])
    parser.add_argument("--number_additional_frames", default=0, required=False)
    parser.add_argument("--gap_between_frames", default=1, required=False)
    parser.add_argument("--use_zoom_in", default=True, required=False)
    args = parser.parse_args()

    run_inference_segvol(json_file=args.json_file,
                         output_folder=args.output_folder,
                         prompt_type=args.prompt_type,
                         prompt_mode=args.prompt_mode,
                         number_prompts=args.number_prompts,
                         number_random_prompts=args.random_number_prompts,
                         initial_frame_selection=args.initial_frame_selection,
                         number_additional_frames=args.number_additional_frames,
                         use_zoom_in=args.use_zoom_in,
                         gap_between_frames=args.gap_between_frames)
