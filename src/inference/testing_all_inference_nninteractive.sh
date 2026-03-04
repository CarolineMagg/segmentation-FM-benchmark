#!/bin/bash

set -e

CUDA_VISIBLE_DEVICES=0 python -m src.inference.nninteractive.testing_nninteractive
