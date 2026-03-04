from pathlib import Path

from src.inference.scribble_prompt.scribble_prompt_inference import run_inference_scribble_prompt
from src.project_root import PROJECT_ROOT

json_file = Path(PROJECT_ROOT / "assets" / "demo" / "prompts" / "prompts_2d.json")
number_prompts = 1
random_number_prompts = 1
model_type = "sam"
model_type2 = "unet"

output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "scribble_prompt_sam" / "center_single")
run_inference_scribble_prompt(json_file=json_file, output_folder=output_folder, prompt_type=["center"],
                              number_prompts=number_prompts, number_random_prompts=random_number_prompts,
                              model_type=model_type, debug=True)

output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "scribble_prompt_sam" / "bbox_multiple")
run_inference_scribble_prompt(json_file=json_file, output_folder=output_folder, prompt_type=["bbox"],
                              number_prompts=5, number_random_prompts=random_number_prompts,
                              model_type=model_type, debug=True)

output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "scribble_prompt_unet" / "bbox_center_multiple")
run_inference_scribble_prompt(json_file=json_file, output_folder=output_folder, prompt_type=["bbox", "center"],
                              number_prompts=5, number_random_prompts=random_number_prompts,
                              model_type=model_type2, debug=True)
