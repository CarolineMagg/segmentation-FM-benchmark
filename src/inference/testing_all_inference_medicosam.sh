#!/bin/bash

set -e

CUDA_VISIBLE_DEVICES=0 python -m src.inference.medico_sam.testing_medico_sam2d
CUDA_VISIBLE_DEVICES=0 python -m src.inference.medico_sam.testing_medico_sam3d
