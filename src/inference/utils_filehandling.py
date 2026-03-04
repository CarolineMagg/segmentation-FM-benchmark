from pathlib import Path
from typing import Union, Optional

import nibabel as nib
import numpy as np
import SimpleITK as sitk


def read_image_depth_first(path_image: Path, file_id: str, file_ending: str = ".nii.gz", file_id_suffix: str = "_0000"):
    nii_image, nii_image_affine = read_nii_file(path_image, file_id, file_ending, file_id_suffix)
    input_image = np.asarray(nii_image.dataobj)
    if len(input_image.shape) == 2:
        input_image = input_image[:, :, np.newaxis]
    input_image = np.einsum("hwd->dhw", input_image)  # depth is on first place
    return input_image, nii_image_affine


def read_nii_file(path_image: Path, file_id: str, file_ending: str = ".nii.gz",
                  file_id_suffix: Optional[str] = "_0000"):
    if file_id_suffix is None:
        img_file_name = path_image / (file_id + file_ending)
    else:
        img_file_name = path_image / (file_id + file_id_suffix + file_ending)
    nii_image = nib.load(img_file_name)
    return nii_image, nii_image.affine


def read_nii_file_with_sitk(path_image: Path, file_id: str, file_ending: str = ".nii.gz",
                            file_id_suffix: Optional[str] = "_0000"):
    if file_id_suffix is None:
        img_file_name = path_image / (file_id + file_ending)
    else:
        img_file_name = path_image / (file_id + file_id_suffix + file_ending)
    sitk_image = sitk.ReadImage(img_file_name)
    nii_image = nib.load(img_file_name)
    return sitk_image, nii_image.affine


def read_image_depth_last(path_image: Path, file_id: str, file_ending: str = ".nii.gz",
                          file_id_suffix: str = "_0000") -> tuple[np.ndarray, np.ndarray]:
    nii_image, nii_image_affine = read_nii_file(path_image, file_id, file_ending, file_id_suffix)
    input_image = np.asarray(nii_image.dataobj)
    return input_image, nii_image_affine


def read_image_depth_first_with_meta_info(path_image: Path, file_id: str, file_ending: str = ".nii.gz",
                                          file_id_suffix: str = "_0000") -> tuple[np.ndarray, dict]:
    nii_image, nii_image_affine = read_nii_file(path_image, file_id, file_ending, file_id_suffix)
    input_image = np.asarray(nii_image.dataobj)  # .transpose(2, 0, 1)
    input_image = np.einsum("hwd->dhw", input_image)
    perm = [2, 0, 1]  # From HWD to DHW → (Z, Y, X)
    ori_spacing = nii_image.header.get_zooms()[:3]
    new_spacing = [ori_spacing[i] for i in perm]
    new_affine = np.diag(new_spacing + [1])
    meta_info = {}
    meta_info["original_subject_affine"] = new_affine
    meta_info["original_subject_spacing"] = new_spacing
    meta_info["original_subject_spatial_shape"] = input_image.shape
    return input_image, meta_info


def write_output_masks_to_nii(output_msk: Union[list[np.ndarray], np.array], labels_lookup: dict, output_dir: Path,
                              file_id: str, file_ending: str, affine: np.ndarray, verbose: bool = False):
    for idx, msk in enumerate(output_msk):
        msk_ = np.einsum("dhw -> hwd", msk)
        label_name = list(labels_lookup.keys())[list(labels_lookup.values()).index(str(idx + 1))]
        write_output_mask_to_nii_simple(msk_, label_name, output_dir, file_id, file_ending, affine, verbose)


def write_output_masks_to_nii_depth_last(output_msk: Union[list[np.ndarray], np.array], labels_lookup: dict,
                                         output_dir: Path, file_id: str, file_ending: str, affine: np.ndarray,
                                         verbose: bool = False):
    for idx, msk in enumerate(output_msk):
        label_name = list(labels_lookup.keys())[list(labels_lookup.values()).index(str(idx + 1))]
        write_output_mask_to_nii_simple(msk, label_name, output_dir, file_id, file_ending, affine, verbose)


def write_output_mask_to_nii_simple(output_mask: np.ndarray, label_name: str, output_dir: Path,
                                    file_id: str, file_ending: str, affine: np.ndarray, verbose: bool = False):
    segm_file_name = output_dir / label_name / (file_id + file_ending)
    nii = nib.Nifti1Image(output_mask, affine=affine)
    nib.save(nii, segm_file_name)
    if verbose:
        print(f"write {segm_file_name}")
