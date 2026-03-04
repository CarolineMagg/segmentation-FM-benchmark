from pathlib import Path

from src.inference.sam2.sam2_inference import run_inference_sam2
from src.project_root import PROJECT_ROOT

json_file = Path(PROJECT_ROOT / "assets" / "demo" / "prompts" / "prompts_2d.json")
output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "sam2_new_hiera_small" / "bbox_single")
prompt_type = ["bbox"]
number_prompts = 1
random_number_prompts = 1
model_type = "sam2_new_hiera_small"

run_inference_sam2(json_file=json_file, output_folder=output_folder, prompt_type=prompt_type,
                   number_prompts=number_prompts, number_random_prompts=random_number_prompts,
                   model_type=model_type, debug=True)
