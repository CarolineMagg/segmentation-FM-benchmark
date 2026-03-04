########################################################################################################################
# Inference script for MedicoSAM in 3D mode
# slice-by-slice inference with propagation of 1024x1024 input
# prompts are given as 2D prompts in 1 initial frame (points or bounding boxes) stored in json file
# settings: prompt type, prompt mode (2D or 3D), number of points, number of random points,
#           selection for initial frame, number of additional frames, gap between additional frames
#           model_type (default: vit_b_medical_imaging)
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
from micro_sam.util import precompute_image_embeddings

from src.inference.medico_sam.medico_sam_utils import create_medicosam_predictor, \
    prompt_based_prediction_medicosam3d_style_combination
from src.inference.utils_3dprompts import extract_original_3d_prompt, convert_list_of_masks_to_mask
from src.inference.utils_filehandling import read_image_depth_first, write_output_masks_to_nii


def run_inference_medicosam3d(json_file: Union[Path, str], output_folder: Union[Path, str], prompt_type: list[str],
                              prompt_mode: str, number_prompts: int, number_random_prompts: int,
                              initial_frame_selection: list[str], number_additional_frames: int,
                              gap_between_frames: int, model_type: str = "vit_b_medical_imaging",
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
    print(f"MedicoSAM3D inference with {model_type}")

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
    all_meta_data = {"model": "MedicoSAM3D", "model_type": model_type,
                     "prompt": prompts_to_be_used, "json_file": str(json_file),
                     "output_folder": str(output_folder), "number_of_prompts": number_of_prompts,
                     "random_number_prompts": number_random_prompts, "label_order": labels_values}

    # iterate through samples
    for file_id in file_ids:
        print(f"process {file_id}")

        # read image file
        # https://github.com/anwai98/tukra/blob/master/tukra/io/util.py#L46
        input_image, affine = read_image_depth_first(path_images, file_id, file_ending, file_id_suffix)
        embeddings: dict = precompute_image_embeddings(
            predictor=predictor, input_=input_image, save_path=None, ndim=3, verbose=True,
        )

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

            # prediction
            start_time: float = time.time()
            volume_segments = prompt_based_prediction_medicosam3d_style_combination(predictor,
                                                                                    input_image, embeddings,
                                                                                    input_point_original,
                                                                                    input_bbox_original)

            elapsed_time: float = time.time() - start_time
            sample_meta_data_dict[str(label_name)] = {"time": elapsed_time, "classes": int(label)}

            volume_segments_all.append(volume_segments)

        # convert dict to segmentation mask & store
        output_msk = convert_list_of_masks_to_mask(volume_segments_all, labels_values, input_image.shape)
        # convert dict to segmentation mask & store
        write_output_masks_to_nii(output_msk, labels_lookup, output_dir, file_id, file_ending,
                                  affine, verbose=True)
        sample_meta_data_dict["time_total"] = np.sum(
            [float(sample_meta_data_dict[k]["time"]) for k in sample_meta_data_dict.keys()])
        all_meta_data[file_id] = sample_meta_data_dict

    meta_data_json_file = output_dir / ("meta_data_combination_" + "_".join(prompts_to_be_used) + ".json")
    with open(meta_data_json_file, "w") as file:
        json.dump(all_meta_data, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MedicoSAM3D inference for combination of prompts from json file")
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
    parser.add_argument("--model_type", default="vit_b_medical_imaging")
    args = parser.parse_args()

    run_inference_medicosam3d(json_file=args.json_file,
                              output_folder=args.output_folder,
                              prompt_type=args.prompt_type,
                              prompt_mode=args.prompt_mode,
                              number_prompts=args.number_prompts,
                              number_random_prompts=args.random_number_prompts,
                              initial_frame_selection=args.initial_frame_selection,
                              number_additional_frames=args.number_additional_frames,
                              gap_between_frames=args.gap_between_frames,
                              model_type=args.model_type)
