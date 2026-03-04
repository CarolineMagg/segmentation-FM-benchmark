from pathlib import Path

from src.inference.sam_med2d.sam_med2d_inference import run_inference_sam_med2d
from src.project_root import PROJECT_ROOT

json_file = Path(PROJECT_ROOT / "assets" / "demo" / "prompts" / "prompts_2d.json")
number_prompts = 1
random_number_prompts = 1

output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "sam_med2d" / "center_single")
run_inference_sam_med2d(json_file=json_file, output_folder=output_folder, prompt_type=["center"],
                        number_prompts=number_prompts, random_number_prompts=random_number_prompts,
                        debug=True)

output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "sam_med2d" / "bbox_single")
run_inference_sam_med2d(json_file=json_file, output_folder=output_folder, prompt_type=["bbox"],
                        number_prompts=number_prompts, random_number_prompts=random_number_prompts,
                        debug=True)

output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "sam_med2d" / "combi_single")
run_inference_sam_med2d(json_file=json_file, output_folder=output_folder, prompt_type=["bbox", "center"],
                        number_prompts=number_prompts, random_number_prompts=random_number_prompts,
                        debug=True)
