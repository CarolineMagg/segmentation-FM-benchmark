from pathlib import Path

from src.inference.sam.sam_inference import run_inference_sam
from src.project_root import PROJECT_ROOT

json_file = Path(PROJECT_ROOT / "assets" / "demo" / "prompts" / "prompts_2d.json")
number_prompts = 1
random_number_prompts = 1
model_type = "vit_b"

output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "medico_sam3d" / "center_single")
run_inference_sam(json_file=json_file, output_folder=output_folder, prompt_type=["center"],
                  number_prompts=number_prompts, number_random_prompts=random_number_prompts,
                  model_type=model_type, debug=True)

output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "medico_sam3d" / "bbox_single")
run_inference_sam(json_file=json_file, output_folder=output_folder, prompt_type=["bbox"],
                  number_prompts=number_prompts, number_random_prompts=random_number_prompts,
                  model_type=model_type, debug=True)

output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "medico_sam3d" / "bbox_center_single")
run_inference_sam(json_file=json_file, output_folder=output_folder, prompt_type=["bbox", "center"],
                  number_prompts=number_prompts, number_random_prompts=random_number_prompts,
                  model_type=model_type, debug=True)
