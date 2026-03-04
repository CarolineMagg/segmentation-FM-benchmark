#!/bin/bash

set -e

CUDA_VISIBLE_DEVICES=0 python -m src.inference.med_sam.testing_med_sam
CUDA_VISIBLE_DEVICES=0 python -m src.inference.med_sam2.testing_med_sam2
CUDA_VISIBLE_DEVICES=0 python -m src.inference.sam.testing_sam
CUDA_VISIBLE_DEVICES=0 python -m src.inference.sam2.testing_sam2
CUDA_VISIBLE_DEVICES=0 python -m src.inference.sam2.testing_sam2_video
CUDA_VISIBLE_DEVICES=0 python -m src.inference.sam_med2d.testing_sam_med2d
CUDA_VISIBLE_DEVICES=0 python -m src.inference.sam_med3d.testing_sam_med3d
CUDA_VISIBLE_DEVICES=0 python -m src.inference.scribble_prompt.testing_scribble_prompt
CUDA_VISIBLE_DEVICES=0 python -m src.inference.seg_vol.testing_seg_vol
CUDA_VISIBLE_DEVICES=0 python -m src.inference.vista3d.testing_vista3d
