from collections import defaultdict
from typing import Tuple, Union, Optional

import numpy as np
from natsort import natsorted

from src.inference.utils_2dprompts import extract_all_slices_with_prompts_to_be_used, \
    extract_original_prompt2d_combination


def collect_area_information_per_class(labels: list[int], data: dict) -> dict[int, dict[str: int]]:
    areas_per_class: dict[int, dict[str: int]] = {}
    for label in labels:
        areas_per_class[label] = {k: np.sum(v) for k, v in data[str(label)]["area"].items()}
    return areas_per_class


def get_min_frame(prompts: dict, prompts_to_be_used: list[str], areas_per_class_per_label: dict) -> int:
    min_idx: int = np.min([int(k) for k, v in areas_per_class_per_label.items() if v > 0])
    empty_prompt: bool = True
    while empty_prompt:
        for p in prompts_to_be_used:
            if str(min_idx) not in prompts[p].keys() or len(prompts[p][str(min_idx)][0]) == 0:
                min_idx += 1
            else:
                empty_prompt = False
    return min_idx


def get_max_frame(prompts: dict, prompts_to_be_used: list[str], areas_per_class_per_label: dict) -> int:
    max_idx: int = np.max([int(k) for k, v in areas_per_class_per_label.items() if v > 0])
    empty_prompt: bool = True
    while empty_prompt:
        for p in prompts_to_be_used:
            if str(max_idx) not in prompts[p].keys() or len(prompts[p][str(max_idx)][0]) == 0:
                max_idx -= 1
            else:
                empty_prompt = False
    return max_idx


def get_initial_frame_original_2d_prompt(prompts: dict, prompts_to_be_used: list[str],
                                         areas_per_class: Optional[dict[int, dict[str, int]]], label: int,
                                         frame_selections: list[str], number_of_frames: int = 0,
                                         gap_between_frames: int = 1, number_of_random_frames: int = 0,
                                         equally_distributed_random_frames: bool = False) -> list[int]:
    # add all initial frames to the list
    all_init_frames: list[int] = []
    for initial_frame_selection in frame_selections:
        if initial_frame_selection == "largest":
            all_init_frames.append(int(max(areas_per_class[label], key=areas_per_class[label].get)))
        elif initial_frame_selection == "center":
            all_init_frames.append(np.median([int(k) for k, v in areas_per_class[label].items() if v > 0]).astype(int))
        elif initial_frame_selection == "first":
            # get first frame with a prompt
            min_idx = get_min_frame(prompts, prompts_to_be_used, areas_per_class[label])
            all_init_frames.append(min_idx)
        elif initial_frame_selection == "last":
            # get last frame with a prompt
            max_idx = get_max_frame(prompts, prompts_to_be_used, areas_per_class[label])
            all_init_frames.append(max_idx)
        # elif initial_frame_selection == "random":
        #     if not equally_distributed_random_frames:
        #         prompts_random = prompts["random_slices"]
        #         random_frames = np.array(prompts_random)[:random_number_frames]
        #         all_init_frames.extend(random_frames)
        #     else:
        #         prompts_random = prompts["random_slices_blocks"]
        #         random_frames = np.array(prompts_random)[:, :random_number_frames]
        #         all_init_frames.extend(random_frames)
        elif initial_frame_selection == "all":
            all_idx = extract_all_slices_with_prompts_to_be_used(prompts, prompts_to_be_used)
            all_init_frames.extend(int(x) for x in all_idx)
        elif "slice_" in initial_frame_selection:  # take the overall nth slice of the volume
            all_idx = extract_all_slices_with_prompts_to_be_used(prompts, prompts_to_be_used)
            number_slice = int(initial_frame_selection.replace("slice_", ""))
            if str(number_slice) in all_idx:
                all_init_frames.append(number_slice)
            else:
                print(f"slice {number_slice} has not prompt")
        elif "sliceR_" in initial_frame_selection:  # take the nth slice where prompts exist
            all_idx = extract_all_slices_with_prompts_to_be_used(prompts, prompts_to_be_used)
            number_slice = int(initial_frame_selection.replace("sliceR_", ""))
            all_init_frames.append(int(all_idx[number_slice]))
        else:
            raise ValueError(f"frame selection {initial_frame_selection} not implemented")
    # get adjacent frames if requested
    if number_of_frames > 0:
        init_frames = all_init_frames.copy()
        min_idx = get_min_frame(prompts, prompts_to_be_used, areas_per_class[label])
        max_idx = get_max_frame(prompts, prompts_to_be_used, areas_per_class[label])
        for init_frame in init_frames:
            for idx in range(number_of_frames + 1):
                if init_frame + idx * gap_between_frames <= max_idx:
                    all_init_frames.append(init_frame + idx * gap_between_frames)
                if init_frame - idx * gap_between_frames >= min_idx:
                    all_init_frames.append(init_frame - idx * gap_between_frames)
    return natsorted(set(all_init_frames))


def get_min_max_frames(prompts: dict, prompt_mode: str, prompts_to_be_used: list[str],
                       areas_per_class: Optional[dict], label: int) -> list[int]:
    if prompt_mode == "2d_prompts" and "2d_prompts" in prompts.keys():
        min_max_frames: list[int] = get_initial_frame_original_2d_prompt(prompts["2d_prompts"], prompts_to_be_used,
                                                                         areas_per_class, label,
                                                                         frame_selections=["first", "last"],
                                                                         number_of_frames=0,
                                                                         gap_between_frames=0,
                                                                         number_of_random_frames=0)
    elif prompt_mode == "3d_prompts" or "3d_prompts" in prompts.keys():
        box_3d = prompts["3d_prompts"]["bbox"][0]
        min_max_frames: list[int] = [box_3d[2], box_3d[5]]
    elif prompt_mode == "3d_prompts_from_2d" and "3d_prompts" in prompts.keys():
        box_3d = prompts["3d_prompts"]["bbox"][0]
        min_max_frames: list[int] = [box_3d[2], box_3d[5]]
    else:
        raise KeyError(f"prompt dict has no prompt key, must have '2d_prompts' or '3d_prompts' key.")
    return min_max_frames


def get_frames_for_propagation(input_point_original: np.ndarray, input_bbox_original: np.ndarray,
                               min_max_frames: list[int]) -> tuple[list, list]:
    assert len(min_max_frames) == 2
    if len(input_point_original) == 0 and len(input_bbox_original) == 0:
        return [None] * 2, [None] * 2
    first_slice_with_prompt: int = extract_first_slice_with_prompt(input_point_original,
                                                                   input_bbox_original)
    max_frame_num_to_track = [min_max_frames[1] - first_slice_with_prompt,
                              first_slice_with_prompt - min_max_frames[0]]
    start_frame_idx = [first_slice_with_prompt, first_slice_with_prompt]
    return start_frame_idx, max_frame_num_to_track


def extract_first_slice_with_prompt(prompt_points: np.ndarray, prompt_bbox: np.ndarray) -> int:
    point_frames = []
    box_frames = []
    if len(prompt_points) > 0:
        _, _, point_frames = transform_3dpoint_into_components(prompt_points)
    if len(prompt_bbox) > 0:
        _, box_frames = transform_3dbox_into_components(prompt_bbox)
    if len(point_frames) > 0 and len(box_frames) > 0:
        assert point_frames == box_frames, "init frames for box and point need to be the same"
    if len(point_frames) > 0:
        return point_frames[0]
    elif len(box_frames) > 0:
        return box_frames[0]
    else:
        raise ValueError("init needs at least one frame")


def extract_original_3d_prompt_from_2d(prompts: dict, prompts_to_be_used: list[str],
                                       init_frames: list[int], number_of_prompts: int,
                                       number_random_prompts: int, already_3d: bool) -> Tuple[np.ndarray, np.ndarray]:
    # extract original prompts and transform to 3D prompt
    input_point_tmp_list: list[np.ndarray] = []
    input_bbox_tmp_list: list[np.ndarray] = []
    for idx in init_frames:
        input_point_3d, input_bbox_3d = extract_original_prompt2d_combination(
            prompts, prompts_to_be_used, idx, number_of_prompts, number_random_prompts)
        if not already_3d:
            input_point_3d = transform_original_point_to_3d(input_point_3d, [idx] * len(input_point_3d))
            input_bbox_3d = transform_original_bbox_to_3d(input_bbox_3d, [idx] * len(input_bbox_3d))
        if len(input_point_3d) > 0:
            input_point_tmp_list.append(input_point_3d)
        if len(input_bbox_3d) > 0:
            input_bbox_tmp_list.append(input_bbox_3d)
    input_point_original3d: np.ndarray = np.concatenate(input_point_tmp_list) if len(input_point_tmp_list) > 0 else []
    input_bbox_original3d: np.ndarray = np.concatenate(input_bbox_tmp_list) if len(input_bbox_tmp_list) > 0 else []
    # make unique
    if len(input_point_original3d) > 0:
        unique_array, indices = np.unique(input_point_original3d, axis=0, return_index=True)
        sorted_indices = np.sort(indices)
        if len(sorted_indices) > 0:
            input_point_original3d = input_point_original3d[sorted_indices]
    if len(input_bbox_original3d) > 0:
        unique_array, indices = np.unique(input_bbox_original3d, axis=0, return_index=True)
        sorted_indices = np.sort(indices)
        if len(sorted_indices) > 0:
            input_bbox_original3d = input_bbox_original3d[sorted_indices]
    return input_point_original3d, input_bbox_original3d


def extract_original_3d_prompt_from_3d(prompts: dict, prompts_to_be_used: list[str],
                                       number_of_prompts: int) -> Tuple[np.ndarray, np.ndarray]:
    # extract original prompts and transform to 3D prompt
    input_point_tmp_list: list[np.ndarray] = []
    input_bbox_tmp_list: list[np.ndarray] = []
    if "centroid" in prompts_to_be_used:
        input_point_tmp_list.append(np.array(prompts["centroid"][:number_of_prompts]))
    if "center" in prompts_to_be_used:
        input_point_tmp_list.append(np.array(prompts["center"][:number_of_prompts]))
    # get bbox prompts
    if "bbox" in prompts_to_be_used:
        input_bbox_tmp_list: np.ndarray = np.array(prompts["bbox"][:number_of_prompts])
    input_point_original3d: np.ndarray = np.concatenate(input_point_tmp_list) if len(input_point_tmp_list) > 0 else []
    input_bbox_original3d: np.ndarray = input_bbox_tmp_list if len(input_bbox_tmp_list) > 0 else []
    # make unique
    if len(input_point_original3d) > 0:
        unique_array, indices = np.unique(input_point_original3d, axis=1, return_index=True)
        sorted_indices = np.sort(indices)
        if len(sorted_indices) > 0:
            input_point_original3d = input_point_original3d[:, sorted_indices]
    if len(input_bbox_original3d) > 0:
        unique_array, indices = np.unique(input_bbox_original3d, axis=0, return_index=True)
        sorted_indices = np.sort(indices)
        if len(sorted_indices) > 0:
            input_bbox_original3d = input_bbox_original3d[sorted_indices]
    return input_point_original3d, input_bbox_original3d


def extract_original_3d_prompt(prompts: dict, prompt_mode: str, prompts_to_be_used: list[str], label: int,
                               number_of_prompts: int, number_random_points: int,
                               areas_per_class: Optional[dict], labels_values: list[int], data_prompt: dict,
                               initial_frame_selection: list[str] = ["largest"],
                               number_additional_frames: int = 0, gap_between_frames: int = 0,
                               number_random_frames: int = 0, equally_distributed_random_frames: bool = False) -> Tuple[
    np.ndarray, np.ndarray, dict]:
    if "2d_prompts" in prompts.keys() and prompt_mode == "2d_prompts":
        if areas_per_class is None:
            areas_per_class = collect_area_information_per_class(labels_values, data_prompt)
        # set initial frame from original prompt
        all_init_frames = get_initial_frame_original_2d_prompt(prompts["2d_prompts"], prompts_to_be_used,
                                                               areas_per_class, label, initial_frame_selection,
                                                               number_of_frames=number_additional_frames,
                                                               gap_between_frames=gap_between_frames,
                                                               number_of_random_frames=number_random_frames,
                                                               equally_distributed_random_frames=equally_distributed_random_frames)

        input_point_original, input_bbox_original = extract_original_3d_prompt_from_2d(prompts["2d_prompts"],
                                                                                       prompts_to_be_used,
                                                                                       all_init_frames,
                                                                                       number_of_prompts,
                                                                                       number_random_points,
                                                                                       already_3d=False)

    elif "3d_prompts" in prompts.keys() and prompt_mode == "3d_prompts":
        input_point_original, input_bbox_original = extract_original_3d_prompt_from_3d(prompts["3d_prompts"],
                                                                                       prompts_to_be_used,
                                                                                       number_of_prompts)
    elif "3d_prompts_from_2d" in prompts.keys() and prompt_mode == "3d_prompts_from_2d":
        all_init_frames = get_initial_frame_original_2d_prompt(prompts["3d_prompts_from_2d"], prompts_to_be_used,
                                                               None, label, initial_frame_selection,
                                                               number_of_frames=0,
                                                               gap_between_frames=1,
                                                               number_of_random_frames=number_random_frames,
                                                               equally_distributed_random_frames=equally_distributed_random_frames)
        input_point_original, input_bbox_original = extract_original_3d_prompt_from_2d(prompts["3d_prompts_from_2d"],
                                                                                       prompts_to_be_used,
                                                                                       all_init_frames,
                                                                                       number_of_prompts,
                                                                                       number_random_prompts=0,
                                                                                       already_3d=True)
    elif "2d_prompts" in prompts.keys() and prompt_mode == "3d_prompts_from_2d":
        all_init_frames = get_initial_frame_original_2d_prompt(prompts["2d_prompts"], prompts_to_be_used,
                                                               None, label, initial_frame_selection,
                                                               number_of_frames=0,
                                                               gap_between_frames=1,
                                                               number_of_random_frames=number_random_frames,
                                                               equally_distributed_random_frames=equally_distributed_random_frames)
        input_point_original, input_bbox_original = extract_original_3d_prompt_from_2d(prompts["2d_prompts"],
                                                                                       prompts_to_be_used,
                                                                                       all_init_frames,
                                                                                       number_of_prompts,
                                                                                       number_random_prompts=0,
                                                                                       already_3d=True)

    else:
        raise KeyError(f"prompt dict has no prompt key, must have {prompt_mode}.")
    return input_point_original, input_bbox_original, areas_per_class


def transform_original_point_to_3d(input_prompt: np.ndarray, slice_idx: list[int]) -> np.ndarray:
    input_prompt_new = []
    for point, idx in zip(input_prompt, slice_idx):
        input_prompt_new.append([point[0], point[1], idx, point[2]])
    input_prompt_new = np.array(input_prompt_new)
    return input_prompt_new


def transform_original_bbox_to_3d(input_prompt: np.ndarray, slice_idx: list[int]) -> np.ndarray:
    input_prompt_new = []
    for bbox, idx in zip(input_prompt, slice_idx):
        input_prompt_new.append([bbox[0], bbox[1], idx, bbox[2], bbox[3], idx])
    input_prompt_new = np.array(input_prompt_new)
    return input_prompt_new


def transform_3dpoint_into_components(input_point_original3d: np.ndarray) -> Tuple[
    dict[int, np.ndarray], dict[int, list[int]], list[int]]:
    point_labels: dict[int, list[int]] = defaultdict(list)
    temp_coordinates: dict[int, list] = defaultdict(list)
    for row in input_point_original3d:
        init_frame: int = int(row[2])
        pc: np.ndarray = row[:2]
        pl: int = row[-1]
        temp_coordinates[init_frame].append(pc)
        point_labels[init_frame].append(pl)
    point_coordinates: dict[int, np.ndarray] = {k: np.vstack(v) for k, v in temp_coordinates.items()}
    return point_coordinates, dict(point_labels), list(point_coordinates.keys())


def transform_3dbox_into_components(input_box_original3d: np.ndarray) -> Tuple[
    dict[int, np.ndarray], list[int]]:
    temp_coordinates: dict[int, list] = defaultdict(list)
    for row in input_box_original3d:
        init_frame: int = int(row[2])
        end_frame: int = int(row[5])
        if init_frame != end_frame:  # for 3D boxes, we take the center slice as prompts
            init_frame = (init_frame + end_frame) // 2
        bc = [row[i] for i in [0, 1, 3, 4]]
        temp_coordinates[init_frame].append(bc)
    box_coordinates: dict[int, np.ndarray] = {k: np.vstack(v) for k, v in temp_coordinates.items()}
    return box_coordinates, list(box_coordinates.keys())


def transform_3dbox_resize(input_box_original: np.ndarray, orig_shape: np.ndarray, new_shape: np.ndarray) -> Optional[
    np.ndarray]:
    if len(orig_shape) > 3:
        orig_shape = orig_shape[-3:]
    if len(new_shape) > 3:
        new_shape = new_shape[-3:]
    assert len(orig_shape) == len(new_shape)
    if input_box_original is None or len(input_box_original) == 0:
        return None
    scaling_factor = [new / original for original, new in zip(orig_shape, new_shape)]
    input_box_new = []
    for input_box in input_box_original:
        new_box_resized = [*scaling_factor, *scaling_factor] * input_box[::-1]
        new_box_resized = np.round(new_box_resized[::-1]).astype(int)
        assert is_3dbox_inside_volume(new_box_resized,
                                      new_shape), f"box outside of volume: {new_box_resized}, {new_shape}"
        input_box_new.append(new_box_resized)
    return np.asarray(input_box_new)


def transform_2dbox_resize(input_box_original: np.ndarray, orig_shape: np.ndarray, new_shape: np.ndarray) -> np.ndarray:
    if len(orig_shape) > 2:
        orig_shape = orig_shape[-2:]
    if len(new_shape) > 2:
        new_shape = new_shape[-2:]
    assert len(orig_shape) == len(new_shape)
    if input_box_original is None or len(input_box_original) == 0:
        return np.asarray([])
    scaling_factor = [new / original for original, new in zip(orig_shape, new_shape)]
    input_box_new = []
    for input_box in input_box_original:
        new_box_resized = [*scaling_factor, *scaling_factor] * input_box[::-1]
        new_box_resized = np.round(new_box_resized[::-1]).astype(int)
        assert is_2dbox_inside_plane(new_box_resized,
                                      new_shape), f"box outside of plane: {new_box_resized}, {new_shape}"
        input_box_new.append(new_box_resized)
    return np.asarray(input_box_new)

def transform_3dpoint_resize(input_point_original: np.ndarray, orig_shape: np.ndarray, new_shape: np.ndarray) -> \
        Optional[np.ndarray]:
    if len(orig_shape) > 3:
        orig_shape = orig_shape[-3:]
    if len(new_shape) > 3:
        new_shape = new_shape[-3:]
    assert len(orig_shape) == len(new_shape)
    if input_point_original is None or len(input_point_original) == 0:
        return None
    scaling_factor = [new / original for original, new in zip(orig_shape, new_shape)]
    input_point_new = []
    for input_point in input_point_original:
        new_point_resized = scaling_factor * input_point[:-1][::-1]
        new_point_resized = np.append(np.round(new_point_resized[::-1]).astype(int), input_point[-1])
        assert is_3dpoint_inside_volume(new_point_resized,
                                        new_shape), f"point outside of volume: {new_point_resized}, {new_shape}"
        input_point_new.append(new_point_resized)
    return np.asarray(input_point_new)


def transform_2dpoint_resize(input_point_original: np.ndarray, orig_shape: np.ndarray, new_shape: np.ndarray) -> \
        Optional[np.ndarray]:
    if len(orig_shape) > 2:
        orig_shape = orig_shape[-2:]
    if len(new_shape) > 2:
        new_shape = new_shape[-2:]
    assert len(orig_shape) == len(new_shape)
    if input_point_original is None or len(input_point_original) == 0:
        return np.asarray([])
    scaling_factor = [new / original for original, new in zip(orig_shape, new_shape)]
    input_point_new = []
    for input_point in input_point_original:
        new_point_resized = scaling_factor * input_point[:-1][::-1]
        new_point_resized = np.append(np.round(new_point_resized[::-1]).astype(int), input_point[-1])
        assert is_2dpoint_inside_plane(new_point_resized,
                                        new_shape), f"point outside of plane: {new_point_resized}, {new_shape}"
        input_point_new.append(new_point_resized)
    return np.asarray(input_point_new)


def transform_3dpoint_resample(input_point_original: np.ndarray, ori_affine: np.ndarray,
                               new_affine: Union[list[float], np.ndarray], new_shape: np.ndarray) -> np.ndarray:
    input_prompt_new = []
    for input_point in input_point_original:
        point_in_world = ori_affine @ np.append(input_point[:-1][::-1], 1)
        new_point_resampled = np.linalg.inv(new_affine) @ point_in_world
        new_point_resampled = np.round(new_point_resampled[:-1])[::-1].astype(int)
        new_point_resampled = np.append(new_point_resampled, input_point[-1])
        assert is_3dpoint_inside_volume(new_point_resampled,
                                        new_shape), f"point outside of volume: {new_point_resampled}, {new_shape}"
        input_prompt_new.append(new_point_resampled)
    return np.asarray(input_prompt_new)


def transform_3dpoint_crop_or_pad(input_prompt: Union[list, np.ndarray], padding: Optional[np.ndarray],
                                  cropping: Optional[np.ndarray], new_shape: np.ndarray) -> np.ndarray:
    input_prompt_new = []
    if padding is None:
        padding = (0, 0, 0, 0, 0, 0)
    if cropping is None:
        cropping = (0, 0, 0, 0, 0, 0)
    point_offset: np.ndarray = np.asarray(padding) - np.asarray(cropping)
    point_offset = point_offset[[4, 2, 0]]
    for input_point in input_prompt:
        new_point = input_point[:3] + point_offset
        new_point = np.append(new_point.astype(int), input_point[-1])
        assert is_3dpoint_inside_volume(new_point, new_shape), f"point outside of volume: {new_point}, {new_shape}"
        input_prompt_new.append(new_point)
    return np.asarray(input_prompt_new)


def crop_3dbox_to_foreground(input_box_original: np.ndarray, foreground_start_coordinates: np.ndarray,
                             img_shape: np.ndarray) -> Optional[np.ndarray]:
    if len(input_box_original) == 0:
        return None
    else:
        input_box_new = []
        z_start, y_start, x_start = foreground_start_coordinates
        for input_box in input_box_original:
            box = [input_box[0] - x_start, input_box[1] - y_start, input_box[2] - z_start, input_box[3] - x_start,
                   input_box[4] - y_start, input_box[5] - z_start]
            assert is_3dbox_inside_volume(box, img_shape), f"box outside of volume: {box}, {img_shape}"
            input_box_new.append(box)
        return np.array(input_box_new)


def crop_3dpoint_to_foreground(input_point_original: np.ndarray, foreground_start_coordinates: np.ndarray,
                               img_shape: np.ndarray) -> Optional[np.ndarray]:
    if len(input_point_original) == 0:
        return None
    else:
        input_points_new = []
        z_start, y_start, x_start = foreground_start_coordinates
        for input_point in input_point_original:
            point = [input_point[0] - x_start, input_point[1] - y_start, input_point[2] - z_start, input_point[3]]
            assert is_3dpoint_inside_volume(point, img_shape), f"point outside of volume: {point}, {img_shape}"
            input_points_new.append(point)
        return np.array(input_points_new)


def reorder_bbox_coordinates(input_bbox_original):
    input_bbox_new = []
    for input_bbox in input_bbox_original:
        input_bbox_new.append(
            [input_bbox[2], input_bbox[1], input_bbox[0], input_bbox[5], input_bbox[4], input_bbox[3]])
    return np.array(input_bbox_new)


def reorder_point_coordinates(input_point_original):
    input_point_new = []
    for input_point in input_point_original:
        if len(input_point) == 4:
            input_point_new.append([input_point[2], input_point[1], input_point[0], input_point[3]])
        else:
            input_point_new.append([input_point[2], input_point[1], input_point[0]])
    return np.array(input_point_new)


def is_3dbox_inside_volume(box: Union[np.ndarray, list], shape: Union[np.ndarray, list]) -> bool:
    x_min, y_min, z_min, x_max, y_max, z_max = box
    if len(shape) == 3:
        D, H, W = shape
    elif len(shape) == 4:
        _, D, H, W = shape
    else:
        raise ValueError("'shape' should have length 3 or 4.")
    return (
            0 <= y_min < y_max <= H and
            0 <= x_min < x_max <= W and
            0 <= z_min < z_max <= D
    )


def is_2dbox_inside_plane(box: Union[np.ndarray, list], shape: Union[np.ndarray, list]) -> bool:
    x_min, y_min, x_max, y_max = box
    if len(shape) == 3:
        _, H, W = shape
    elif len(shape) == 2:
        H, W = shape
    else:
        raise ValueError("'shape' should have length 2 or 3.")
    return (
            0 <= y_min < y_max <= H and
            0 <= x_min < x_max <= W
    )


def is_3dpoint_inside_volume(point: Union[np.ndarray, list], shape: Union[np.ndarray, list]) -> bool:
    x, y, z, _ = point
    if len(shape) == 3:
        D, H, W = shape
    elif len(shape) == 4:
        _, D, H, W = shape
    else:
        raise ValueError("'shape' should have length 3 or 4.")
    return (0 <= y < H) and (0 <= x < W) and (0 <= z < D)


def is_2dpoint_inside_plane(point: Union[np.ndarray, list], shape: Union[np.ndarray, list]) -> bool:
    x, y, _ = point
    if len(shape) == 3:
        _, H, W = shape
    elif len(shape) == 2:
        H, W = shape
    else:
        raise ValueError("'shape' should have length 2 or 3.")
    return (0 <= y < H) and (0 <= x < W)


def convert_list_of_dicts_to_mask(volume_segments_all: list[dict[int, np.ndarray]],
                                  labels_values: list[int], input_image_shape: np.ndarray) -> np.ndarray:
    output_msk: np.ndarray = np.zeros((len(labels_values), *input_image_shape), dtype=np.uint8)
    for volume_segments in volume_segments_all:
        for slice_idx, output_msks in volume_segments.items():
            for idx, label in enumerate(labels_values):
                if label in output_msks.keys():
                    output_msk[idx][slice_idx][output_msks[label][0] > 0] = label
    return output_msk


def convert_list_of_masks_to_mask(volume_segments_all: list[np.ndarray], labels_values: list[int],
                                  input_image_shape: np.ndarray) -> np.ndarray:
    output_msk: np.ndarray = np.zeros((len(labels_values), *input_image_shape), dtype=np.uint8)
    for idx, msk in enumerate(volume_segments_all):
        if msk is not None:
            output_msk[idx][msk > 0] = labels_values[idx]
    return output_msk
