from typing import Union, Tuple

import numpy as np

from natsort import natsorted


def extract_all_slices_with_prompts2d(data: dict, file_id: str, prompts_to_be_used: list[str], labels: list[int]) -> \
        list[str]:
    all_slices_with_prompts: set[str] = set()
    for prompt in prompts_to_be_used:
        for idx in labels:
            all_slices_with_prompts.update(data[file_id][str(idx)]["2d_prompts"][prompt].keys())
        # all_slices_with_prompts_: list[set[str]] = [set(data[file_id][str(idx)]["2d_prompts"][prompt].keys()) for idx
        #                                             in labels]
        # all_slices_with_prompts.append([x for xs in all_slices_with_prompts_ for x in xs])
    return natsorted(all_slices_with_prompts)


def extract_all_slices_with_prompts_to_be_used(prompts2d: dict, prompts_to_be_used: list[str]) -> list[str]:
    all_slices_with_prompts: set[str] = set()
    for prompt in prompts_to_be_used:
        all_slices_with_prompts.update(set(prompts2d[prompt].keys()))
    return natsorted(all_slices_with_prompts)


def extract_original_prompt2d(prompts_dict: dict, prompt_to_be_used: str, slice_idx: int,
                              number_of_prompts: int = 1, random_number_prompts: int = 1) -> np.ndarray:
    prompts: dict[str, list] = prompts_dict[prompt_to_be_used]
    if prompt_to_be_used == "random":
        input_prompt: np.ndarray = np.array([x for x in prompts[str(slice_idx)][:number_of_prompts] if len(x) > 0])
        if len(input_prompt) > 0:
            input_prompt = input_prompt[:, :random_number_prompts, :].reshape(
                input_prompt.shape[0] * random_number_prompts, 3)
    else:
        input_prompt: np.ndarray = np.array(prompts[str(slice_idx)][:number_of_prompts])
    return input_prompt


def extract_original_prompt2d_combination(prompts_2d, prompts_to_be_used, slice_idx, number_of_prompts,
                                          number_random_prompts) -> Tuple[np.ndarray, np.ndarray]:
    input_prompt_points_tmp: list[np.ndarray] = []
    input_prompt_bbox: np.ndarray = np.array([])
    input_prompt_points: np.ndarray = np.array([])
    # get point prompts
    if "random" in prompts_to_be_used:
        input_prompt: np.ndarray = np.array(
            [x for x in prompts_2d["random"][str(slice_idx)][:number_of_prompts] if len(x) > 0])
        if len(input_prompt) == 0 and len(prompts_2d["random"][str(slice_idx)]) > 1:
            # if first component did not result in random points (too small)
            input_prompt: np.ndarray = np.array(
                [x for x in prompts_2d["random"][str(slice_idx)][1:number_of_prompts + 1] if len(x) > 0])
        if len(input_prompt) > 0:
            input_prompt: np.ndarray = input_prompt[:, :number_random_prompts, :].reshape(
                input_prompt.shape[0] * number_random_prompts, 3)
        input_prompt_points_tmp.append(input_prompt)
    if "negative" in prompts_to_be_used:
        input_prompt: np.ndarray = np.array(
            [x for x in prompts_2d["negative"][str(slice_idx)][:number_of_prompts] if len(x) > 0])
        if len(input_prompt) == 0 and len(prompts_2d["negative"][str(slice_idx)]) > 1:
            # if first component did not result in random points (too small)
            input_prompt: np.ndarray = np.array(
                [x for x in prompts_2d["negative"][str(slice_idx)][1:number_of_prompts + 1] if len(x) > 0])
        if len(input_prompt) > 0:
            input_prompt: np.ndarray = input_prompt[:, :number_random_prompts, :].reshape(
                input_prompt.shape[0] * number_random_prompts, 3)
        input_prompt_points_tmp.append(input_prompt)
    if "centroid" in prompts_to_be_used:
        input_prompt_points_tmp.append(np.array(prompts_2d["centroid"][str(slice_idx)][:number_of_prompts]))
    if "center" in prompts_to_be_used:
        input_prompt_points_tmp.append(np.array(prompts_2d["center"][str(slice_idx)][:number_of_prompts]))
    # get bbox prompts
    if "bbox" in prompts_to_be_used:
        input_prompt_bbox: np.ndarray = np.array(prompts_2d["bbox"][str(slice_idx)][:number_of_prompts])
    # get unique point prompts
    input_prompt_points_non_empty: list[list[np.ndarray]] = [x for x in input_prompt_points_tmp if len(x) > 0]
    if len(input_prompt_points_non_empty) > 0:
        input_prompt_points_concat: np.ndarray = np.concatenate(input_prompt_points_non_empty, axis=0)
        unique_array, indices = np.unique(input_prompt_points_concat, axis=0, return_index=True)
        sorted_indices: np.ndarray = np.sort(indices)
        input_prompt_points = input_prompt_points_concat[sorted_indices]
    return input_prompt_points, input_prompt_bbox
