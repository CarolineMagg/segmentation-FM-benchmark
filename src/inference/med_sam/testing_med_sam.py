from pathlib import Path

from src.inference.med_sam.med_sam_inference import run_inference_med_sam
from src.project_root import PROJECT_ROOT

json_file = Path(PROJECT_ROOT / "assets" / "demo" / "prompts" / "prompts_2d.json")
output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "med_sam" / "bbox_single")
number_prompts = 1

run_inference_med_sam(json_file=json_file, output_folder=output_folder, number_prompts=number_prompts, debug=True)
