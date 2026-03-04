from pathlib import Path

from src.inference.nninteractive.nninteractive_inference import run_inference_nninteractive
from src.project_root import PROJECT_ROOT

number_prompts = 1
random_number_prompts = 1
number_additional_frames = 0
gap_between_frames = 0

prompt_mode2 = "2d_prompts"
json_file2 = Path(PROJECT_ROOT / "assets" / "demo" / "prompts" / "prompts_2d.json")
initial_frame_selection2d = ["center", "slice_5"]

prompt_mode3 = "3d_prompts"
json_file3 = Path(PROJECT_ROOT / "assets" / "demo" / "prompts" / "prompts_3d.json")
initial_frame_selection3d = ["none"]

output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "nninteractive" / "bbox_2d_center")
prompt_type = ["bbox"]
run_inference_nninteractive(json_file=json_file2, output_folder=output_folder, prompt_type=prompt_type,
                            prompt_mode=prompt_mode2, number_prompts=number_prompts,
                            number_random_prompts=random_number_prompts,
                            initial_frame_selection=initial_frame_selection2d,
                            number_additional_frames=number_additional_frames, gap_between_frames=gap_between_frames,
                            debug=True)

output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "nninteractive" / "center_3d_center")
prompt_type = ["center"]
run_inference_nninteractive(json_file=json_file2, output_folder=output_folder, prompt_type=prompt_type,
                            prompt_mode=prompt_mode2, number_prompts=number_prompts,
                            number_random_prompts=random_number_prompts,
                            initial_frame_selection=initial_frame_selection2d,
                            number_additional_frames=number_additional_frames, gap_between_frames=gap_between_frames,
                            debug=True)

output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "nninteractive" / "center_3d")
prompt_type = ["center"]
run_inference_nninteractive(json_file=json_file3, output_folder=output_folder, prompt_type=prompt_type,
                            prompt_mode=prompt_mode3, number_prompts=number_prompts,
                            number_random_prompts=random_number_prompts,
                            initial_frame_selection=initial_frame_selection3d,
                            number_additional_frames=number_additional_frames, gap_between_frames=gap_between_frames,
                            debug=True)
