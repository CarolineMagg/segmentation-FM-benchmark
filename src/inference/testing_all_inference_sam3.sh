#!/bin/bash

set -e

CUDA_VISIBLE_DEVICES=1 python -m src.inference.sam3.testing_sam3
CUDA_VISIBLE_DEVICES=1 python -m src.inference.sam3.testing_sam3_video