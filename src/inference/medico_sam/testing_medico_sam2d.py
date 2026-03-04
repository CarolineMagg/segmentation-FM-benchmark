from pathlib import Path

from src.inference.medico_sam.medico_sam2d_inference import run_inference_medicosam2d
from src.project_root import PROJECT_ROOT

json_file = Path(PROJECT_ROOT/ "assets" / "demo" / "prompts" / "prompts_2d.json")
number_prompts = 1
random_number_prompts = 1
prompt_mode = "2d_prompts"
initial_frame_selection3d = ["none"]
number_additional_frames = 0
gap_between_frames = 0

output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "medico_sam2d" / "combi")
run_inference_medicosam2d(json_file=json_file, output_folder=output_folder, prompt_type=["bbox", "center"],
                          number_prompts=5, number_random_prompts=random_number_prompts,
                          model_type="vit_b_medical_imaging", debug=True)
