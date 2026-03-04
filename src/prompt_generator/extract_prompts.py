from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union, Tuple

import cv2
import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.ndimage import distance_transform_edt, generate_binary_structure, label
from scipy.spatial.distance import cdist
from skimage.measure import centroid
import nibabel as nib
from stringzilla import index

from src.inference.utils_3dprompts import transform_original_point_to_3d, transform_original_bbox_to_3d


def read_masks(input_file: Path) -> np.ndarray:
    nii_mask = nib.load(input_file)
    input_mask = np.asarray(nii_mask.dataobj)  # preserve original channel order
    return input_mask


def mask_one_hot_encoding(input_mask: np.ndarray, labels: list[int]) -> np.ndarray:
    num_class = len(labels) + 1
    if num_class > 1:
        mask_one_hot = (np.arange(1, num_class) == input_mask[..., None]).astype(int)
        mask_one_hot = np.moveaxis(mask_one_hot, [-1, -2], [0, 1])
    else:
        mask_one_hot = np.array(input_mask > 0, dtype=int)
        mask_one_hot = np.moveaxis(mask_one_hot, -1, 0)

    if len(mask_one_hot.shape) < 3:
        mask_one_hot = mask_one_hot[np.newaxis, :, :]  # 1*height*depth, to consistent with multi-class setting

    return mask_one_hot


def get_list_of_non_empty_slice(input_mask: np.ndarray, cls: Optional[list[int]] = None) -> list[int]:
    """
    :param input_mask:
    :param cls: if None, input_mask is supposed to be 3D array with D, H, W; otherwise 4D array with N, D, H, W
    :return:
    """
    if cls is None:
        return [idx for idx, m in enumerate(input_mask) if np.max(m) > 0]
    else:
        input_mask = np.uint8(input_mask[cls])
        return [idx for idx, m in enumerate(input_mask) if np.max(m) > 0]


def find_connected_components_2d_with_stats(input_mask: np.ndarray) -> Union[
    Tuple[Tuple[int, np.ndarray, np.ndarray, np.ndarray]], list[float], list[int], list[int]]:
    connectivity = 4
    output: Tuple[int, np.ndarray, np.ndarray, np.ndarray] = cv2.connectedComponentsWithStats(input_mask, connectivity,
                                                                                              cv2.CV_32S)
    ratio_list, regionid_list, area_list = [], [], []
    for region_id in range(1, output[0]):
        # find coordinates of points in the region
        binary_msk: np.ndarray = (output[1] == region_id).astype(np.uint8)
        r: float = np.sum(binary_msk) / np.sum(input_mask)
        ratio_list.append(r)
        regionid_list.append(region_id)
        area_list.append(int(np.sum(binary_msk)))
    return output, ratio_list, regionid_list, area_list


def find_connected_components_3d_with_stats(input_mask: np.ndarray) -> Tuple[
    np.ndarray, list[float], list[int], list[float]]:
    structure = generate_binary_structure(3, 1)  # connectivity=1 (6-connectivity)
    labeled_array, num_labels = label(input_mask > 0, structure=structure)
    ratio_list, regionid_list, area_list = [], [], []
    for region_id in range(1, num_labels + 1):
        # find coordinates of points in the region
        binary_msk: np.ndarray = (labeled_array == region_id).astype(np.uint8)
        r: float = np.sum(binary_msk) / np.sum(input_mask)
        ratio_list.append(r)
        regionid_list.append(region_id)
        area_list.append(int(np.sum(binary_msk)))
    return labeled_array, ratio_list, regionid_list, area_list


def create_point_prompt_2d_center(label_msk: np.ndarray, regionid_list: list[int]) -> list[list[int]]:
    # always inside of volume
    prompt: list = []
    for mask_idx in regionid_list:
        binary_msk: np.ndarray = (label_msk == mask_idx).astype(np.uint8)
        padded_mask: np.ndarray = np.uint8(np.pad(binary_msk, ((1, 1), (1, 1)), 'constant'))
        dist_img: np.ndarray = cv2.distanceTransform(padded_mask, distanceType=cv2.DIST_L2, maskSize=5).astype(
            np.float32)[1:-1, 1:-1]
        max_val: float = dist_img.max()
        coords: np.ndarray = np.argwhere(dist_img == max_val)
        selected: np.ndarray = coords[0]
        if coords.shape[0] > 1:
            selected = coords[np.random.randint(len(coords))]
        y, x = map(int, selected)
        prompt.append([x, y, 1])
    return prompt


def create_point_prompt_3d_center(label_msk: np.ndarray, regionid_list: list[int]):
    prompt: list = []
    for mask_idx in regionid_list:
        binary_msk: np.ndarray = (label_msk == mask_idx).astype(np.uint8)
        padded_mask = np.pad(binary_msk, ((1, 1), (1, 1), (1, 1)), mode='constant')
        dist_img: np.ndarray = distance_transform_edt(padded_mask)[1:-1, 1:-1, 1:-1]
        max_coords: np.ndarray = np.argwhere(dist_img == dist_img.max())
        random_idx: int = np.random.randint(len(max_coords))
        z, y, x = map(int, max_coords[random_idx])
        prompt.append([x, y, z, 1])
    return prompt


def create_point_prompt_2d_random(label_msk: np.ndarray, regionid_list: list[int], size: int = 10) -> list[
    list[list[int]]]:
    # always inside of volume
    prompt: list = []
    for mask_idx in regionid_list:
        binary_msk: np.ndarray = np.where(label_msk == mask_idx, 1, 0).astype(np.uint8)
        kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded_msk: np.ndarray = cv2.erode(binary_msk, kernel, iterations=1)
        cY, cX = np.where(eroded_msk == 1)
        if len(cX) > 15:
            random_idx: list[int] = np.random.randint(0, len(cX), size=size)
            prompt_tmp: list[list[int]] = []
            for ri in random_idx:
                prompt_tmp.append([int(cX[ri]), int(cY[ri]), 1])
            prompt.append(prompt_tmp)
        else:
            prompt.append([])
    return prompt


def create_point_prompt_2d_negative(label_msk: np.ndarray, regionid_list: list[int], size: int = 10) -> list[
    list[list[int]]]:
    # always outside of volume
    prompt: list = []
    for mask_idx in regionid_list:
        binary_msk: np.ndarray = np.where(label_msk == mask_idx, 1, 0).astype(np.uint8)
        kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_msk: np.ndarray = cv2.dilate(binary_msk, kernel, iterations=1)
        kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_msk2: np.ndarray = cv2.dilate(dilated_msk, kernel, iterations=1)
        boundary_msk: np.ndarray = dilated_msk2.copy()
        boundary_msk[dilated_msk == 1] = 0
        boundary_msk[label_msk > 0] = 0
        cY, cX = np.where(boundary_msk == 1)
        if len(cX) > 15:
            random_idx: list[int] = np.random.randint(0, len(cX), size=size)
            prompt_tmp: list[list[int]] = []
            for ri in random_idx:
                prompt_tmp.append([int(cX[ri]), int(cY[ri]), 0])
            prompt.append(prompt_tmp)
        else:
            prompt.append([])
    return prompt


def create_point_prompt_2d_centroid_from_stats(centroids: list[int]) -> list[list[int]]:
    # Note: can land outside of volume (not a very good prompt!)
    # has to be already filtered for background
    prompt = []
    for x in centroids:
        prompt.append([int(x[0]), int(x[1]), 1])
    return prompt  # list with [x, y, 1] entries


def create_point_prompt_3d_centroid_from_mask(label_msk: np.ndarray) -> list[list[int]]:
    # Note: can land outside of volume (not a very good prompt!)
    coords: np.ndarray = np.argwhere(label_msk)
    if coords.size == 0:
        return []
    else:
        centroid = coords.mean(axis=0, keepdims=True)
        z, y, x = np.round(centroid).astype(int)[0]
    return [[int(x), int(y), int(z), 1]]  # [x, y, z, 1]


def create_box_prompt_2d_from_stats(stats) -> list[list[int]]:
    # has to be already filtered for background
    prompt = []
    for x in stats:
        prompt.append([int(x[0]), int(x[1]), int(x[0] + x[2]), int(x[1] + x[3])])
    return prompt


def create_box_prompt_3d_from_mask(mask_cls: np.ndarray) -> list[list[int]]:
    bbox = mask_to_3d_box(mask_cls)
    return [bbox]


def mask_to_box_2d(mask: np.ndarray) -> list[int]:
    mask = mask.squeeze()
    # find coordinates of points in the region
    row, col = np.argwhere(mask).T
    # find the four corner coordinates
    y0, x0 = row.min(), col.min()
    y1, x1 = row.max(), col.max()
    return [int(x0), int(y0), int(x1), int(y1)]


def mask_to_3d_box(mask: np.ndarray) -> list[int]:
    mask = mask.squeeze()
    # find coordinates of points in the region
    slices, row, col = np.argwhere(mask).T
    # find the four corner coordinates
    z0, y0, x0 = slices.min(), row.min(), col.min()
    z1, y1, x1 = slices.max(), row.max(), col.max()
    return [int(x0), int(y0), int(z0), int(x1), int(y1), int(z1)]


def sort_ratio_regionid_area(index_resort: Union[np.ndarray, list[int]], ratio_list: list[float],
                             regionid_list: list[int], area_list: list[float]) -> Tuple[
    list[float], list[int], list[float]]:
    ratio: list[float] = [ratio_list[i] for i in index_resort]
    regionid: list[int] = [regionid_list[i] for i in index_resort]
    area: list[float] = [area_list[i] for i in index_resort]
    return ratio, regionid, area


def remove_index_ratio_regionid_area(index_to_remove: Union[np.ndarray, list[int]], ratio_list: list[float],
                                     regionid_list: list[int], area_list: list[float]) -> Tuple[
    list[float], list[int], list[float]]:
    ratio: list[float] = [ratio_list[i] for i in range(len(ratio_list)) if i not in index_to_remove]
    regionid: list[int] = [regionid_list[i] for i in range(len(regionid_list)) if i not in index_to_remove]
    area: list[float] = [area_list[i] for i in range(len(area_list)) if i not in index_to_remove]
    return ratio, regionid, area


def create_2d_prompts_for_array(input_mask: np.ndarray, labels: Optional[list[int]] = None) -> dict:
    # extract labels if not given
    if labels is None:
        labels = [x for x in np.unique(input_mask) if x > 0]
    # create one hot encoding
    mask_one_hot = mask_one_hot_encoding(input_mask, labels)
    # iterate through classes
    prompts_per_class: dict = {}
    for i, cls in enumerate(labels):
        mask_cls = np.uint8(mask_one_hot[i])

        # init 2D prompts - centroid, center, 2d tight box, random points 2d (pos + neg) #
        centroid_point_2d, box_2d, center_point_2d, random_points_2d, negative_points_2d = {}, {}, {}, {}, {}
        ratio_dict, regionid_dict, area_dict = {}, {}, {}

        # find non empty slices to save computations
        idx_non_empty: list[int] = get_list_of_non_empty_slice(mask_cls)
        if len(idx_non_empty) == 0:
            print(f"label class {cls} not present for current array. continue ...")
        else:
            # iterate through slices for 2D prompts
            for idx in idx_non_empty:
                # find connected components
                mask_cls_slice: np.ndarray = mask_cls[idx]  # H, W
                output_slice, ratio_list, regionid_list, area_list = find_connected_components_2d_with_stats(
                    mask_cls_slice)
                num_labels, label_msk, stats, centroids = output_slice

                # sort based on area
                index_resort: np.ndarray = np.argsort(stats[1:, -1])[::-1]
                stats: np.ndarray = stats[1:][index_resort]
                centroids: np.ndarray = centroids[1:][index_resort]
                ratio_list, regionid_list, area_list = sort_ratio_regionid_area(index_resort,
                                                                                ratio_list,
                                                                                regionid_list,
                                                                                area_list)

                # filter based on area
                index_to_remove: list[int] = [i for i, a in enumerate(area_list) if a < 15] + [i for i, a in
                                                                                               enumerate(ratio_list)
                                                                                               if
                                                                                               a < 0.05]
                index_to_remove = list(np.unique(index_to_remove))
                if len(index_to_remove) > 0:
                    # print(f"remove {index_to_remove} since area criteria is not fullfilled")
                    ratio_list, regionid_list, area_list = remove_index_ratio_regionid_area(index_to_remove,
                                                                                            ratio_list,
                                                                                            regionid_list,
                                                                                            area_list)
                    stats = np.delete(stats, index_to_remove, axis=0)
                    centroids = np.delete(centroids, index_to_remove, axis=0)

                if len(area_list) == 0:
                    continue

                ratio_dict[idx] = ratio_list
                regionid_dict[idx] = regionid_list
                area_dict[idx] = area_list

                # centroid point
                centroid_point_2d[idx]: list[list[int]] = create_point_prompt_2d_centroid_from_stats(centroids)

                # center point
                center_point_2d[idx]: list[list[int]] = create_point_prompt_2d_center(label_msk, regionid_list)

                # 10 random positive points
                random_points_2d[idx]: list[list[list[int]]] = create_point_prompt_2d_random(label_msk, regionid_list,
                                                                                             size=10)

                # bbox
                box_2d[idx]: list[list[list[int]]] = create_box_prompt_2d_from_stats(stats)

                # 10 negative points
                negative_points_2d[idx]: list[list[list[int]]] = create_point_prompt_2d_negative(label_msk,
                                                                                                 regionid_list,
                                                                                                 size=10)

        prompts_per_class[cls] = {"2d_prompts": {"centroid": centroid_point_2d, "bbox": box_2d,
                                                 "center": center_point_2d, "random": random_points_2d,
                                                 "negative": negative_points_2d},
                                  "ratio": ratio_dict, "area": area_dict}

    return prompts_per_class


def create_3d_prompts_for_array(input_mask: np.ndarray, labels: Optional[list[int]] = None,
                                df: Optional[pd.DataFrame] = None, json_2d_prompts: Optional[dict] = None) -> dict:
    # extract labels if not given
    if labels is None:
        labels = [x for x in np.unique(input_mask) if x > 0]
    # create one hot encoding
    mask_one_hot = mask_one_hot_encoding(input_mask, labels)
    # iterate through classes
    prompts_per_class: dict = {}
    for i, cls in enumerate(labels):
        mask_cls = np.uint8(mask_one_hot[i])

        label_msk, ratio_list, regionid_list, area_list = find_connected_components_3d_with_stats(mask_cls)

        # sort based on area
        index_resort: np.ndarray = np.argsort(area_list)[::-1]
        ratio_list, regionid_list, area_list = sort_ratio_regionid_area(index_resort,
                                                                        ratio_list,
                                                                        regionid_list,
                                                                        area_list)

        # filter based on area
        index_to_remove: list[int] = [i for i, a in enumerate(area_list) if a < 15] + [i for i, a in
                                                                                       enumerate(ratio_list)
                                                                                       if
                                                                                       a < 0.05]
        index_to_remove = list(set(index_to_remove))
        if len(index_to_remove) > 0:
            ratio_list, regionid_list, area_list = remove_index_ratio_regionid_area(index_to_remove,
                                                                                    ratio_list,
                                                                                    regionid_list,
                                                                                    area_list)

        if len(area_list) == 0:
            continue

        box_3d: list[list[int]] = create_box_prompt_3d_from_mask(mask_cls)
        centroid_3d: list[list[int]] = create_point_prompt_3d_centroid_from_mask(mask_cls)
        center_3d: list[list[int]] = create_point_prompt_3d_center(label_msk, regionid_list)

        box_2d, center_2d, centroid_2d, random_2d = {}, {}, {}, {}
        if df is not None and json_2d_prompts is not None:
            all_slices = df["slice_id"].tolist()
            json_2d_prompts_class = json_2d_prompts[str(cls)]["2d_prompts"]
            for slice_idx in all_slices:
                if str(slice_idx) in json_2d_prompts_class["bbox"]:
                    input_boxes = json_2d_prompts_class["bbox"][str(slice_idx)]
                    box_2d[str(slice_idx)] = [x.tolist() for x in
                                              transform_original_bbox_to_3d(input_boxes,
                                                                            [slice_idx] * len(input_boxes))]
                if str(slice_idx) in json_2d_prompts_class["center"]:
                    input_points = json_2d_prompts_class["center"][str(slice_idx)]
                    center_2d[str(slice_idx)] = [x.tolist() for x in transform_original_point_to_3d(input_points,
                                                                                                    [slice_idx] * len(
                                                                                                        input_points))]
                if str(slice_idx) in json_2d_prompts_class["centroid"]:
                    input_points = json_2d_prompts_class["centroid"][str(slice_idx)]
                    centroid_2d[str(slice_idx)] = [x.tolist() for x in transform_original_point_to_3d(input_points,
                                                                                                      [slice_idx] * len(
                                                                                                          input_points))]
                    if str(slice_idx) in json_2d_prompts_class["random"]:
                        input_points = json_2d_prompts_class["random"][str(slice_idx)]
                    random_2d[str(slice_idx)] = [[x.tolist() for x in transform_original_point_to_3d(input_point_list,
                                                                                                     [slice_idx] * len(
                                                                                                         input_points))]
                                                 for input_point_list in input_points]

        prompts_per_class[cls] = {"3d_prompts": {"bbox": box_3d, "center": center_3d,
                                                 "centroid": centroid_3d},
                                  "3d_prompts_from_2d": {"bbox": box_2d, "center": center_2d,
                                                         "centroid": centroid_2d, "random": random_2d},
                                  "ratio": ratio_list, "area": area_list}

    return prompts_per_class

def generate_2d_prompts_for_file(input_file: Path, labels: Optional[list[int]] = None) -> dict:
    # read input file
    input_mask: np.ndarray = read_masks(input_file)
    # generate prompts for array
    prompts_2d_per_class: dict = create_2d_prompts_for_array(input_mask, labels)
    return prompts_2d_per_class

def generate_3d_prompts_for_file(input_file: Path, labels: Optional[list[int]] = None,
                                 df: Optional[pd.DataFrame] = None, json_2d_prompts: Optional[dict] = None) -> dict:
    # read input file
    input_mask: np.ndarray = read_masks(input_file)
    # generate prompts for array
    prompts_3d_per_class: dict = create_3d_prompts_for_array(input_mask, labels, df, json_2d_prompts)
    return prompts_3d_per_class

def generate_2d_prompts_for_folder(path_labels: Path, labels: Optional[list[int]] = None,
                                   file_ending: str = ".nii.gz") -> dict:
    # extract all files
    file_ids = natsorted(
        list(map(lambda p: str(p.name).removesuffix(file_ending), path_labels.glob(f'*{file_ending}'))))
    # generate all prompts (2d, 3d)
    all_prompts: dict = {}
    all_prompts["labels_path"] = str(path_labels)
    all_prompts["file_ending"] = file_ending
    all_prompts["labels"] = labels
    for file_id in file_ids:
        print(f"processing {file_id}")
        file_name = path_labels / (file_id + file_ending)
        prompts = generate_2d_prompts_for_file(file_name, labels=labels)
        all_prompts[file_id] = prompts
    return all_prompts

def generate_3d_prompts_for_folder(path_labels: Path, labels: Optional[list[int]] = None,
                                   file_ending: str = ".nii.gz", df: Optional[pd.DataFrame] = None,
                                   json_2d_prompts: Optional[dict] = None):
    # extract all files
    file_ids = natsorted(
        list(map(lambda p: str(p.name).removesuffix(file_ending), path_labels.glob(f'*{file_ending}'))))
    # generate all prompts (2d, 3d)
    all_prompts: dict = {}
    all_prompts["labels_path"] = str(path_labels)
    all_prompts["file_ending"] = file_ending
    all_prompts["labels"] = labels
    for file_id in file_ids:
        print(f"processing {file_id}")
        file_name = path_labels / (file_id + file_ending)
        df_ = df[df["file_id"] == file_id] if df is not None else None
        json_2d_prompts_ = json_2d_prompts[file_id] if json_2d_prompts is not None else None
        prompts = generate_3d_prompts_for_file(file_name, labels=labels, df=df_, json_2d_prompts=json_2d_prompts_)
        all_prompts[file_id] = prompts
    return all_prompts
