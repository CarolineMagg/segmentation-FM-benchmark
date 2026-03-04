from pathlib import Path

from src.inference.sam_med3d.sam_med3d_inference import run_inference_sam_med3d
from src.project_root import PROJECT_ROOT

json_file = Path(PROJECT_ROOT / "assets" / "demo" / "prompts" / "prompts_3d.json")
prompt_type = ["center"]
number_prompts = 1
prompt_mode = "3d_prompts"
random_number_prompts = 1
initial_frame_selection3d = ["none"]
number_additional_frames = 0
gap_between_frames = 0
model_type = "sam_med3d"

output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "sam_med3d" / "resample_center")
data_processing = "resample"
run_inference_sam_med3d(json_file, output_folder, prompt_type, prompt_mode, number_prompts, random_number_prompts,
                        initial_frame_selection3d, number_additional_frames, gap_between_frames, data_processing,
                        model_type, debug=True)

output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "sam_med3d" / "crop_center")
data_processing = "crop"
run_inference_sam_med3d(json_file, output_folder, prompt_type, prompt_mode, number_prompts, random_number_prompts,
                        initial_frame_selection3d, number_additional_frames, gap_between_frames, data_processing,
                        model_type, debug=True)

model_type2 = "sam_med3d_turbo"
output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "sam_med3d_turbo" / "resample_center")
data_processing = "resample"
run_inference_sam_med3d(json_file, output_folder, prompt_type, prompt_mode, number_prompts, random_number_prompts,
                        initial_frame_selection3d, number_additional_frames, gap_between_frames, data_processing,
                        model_type2, debug=True)
