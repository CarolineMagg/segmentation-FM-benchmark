from pathlib import Path

from src.inference.vista3d.vista3d_inference import run_inference_vista3d
from src.project_root import PROJECT_ROOT

json_file = Path(PROJECT_ROOT / "assets" / "demo" / "prompts" / "prompts_3d.json")
json_file_2d = Path(PROJECT_ROOT / "assets" / "demo" / "prompts" / "prompts_2d.json")
number_prompts = 1
random_number_prompts = 1
prompt_mode = "3d_prompts"
initial_frame_selection3d = ["none"]
number_additional_frames = 0
gap_between_frames = 0

output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "vista3d" / "center_3d")
run_inference_vista3d(json_file=json_file, output_folder=output_folder, prompt_type=["center"],
                      prompt_mode=prompt_mode, number_prompts=number_prompts,
                      number_random_prompts=random_number_prompts,
                      initial_frame_selection=initial_frame_selection3d,
                      number_additional_frames=number_additional_frames, gap_between_frames=gap_between_frames,
                      debug=True)

output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "vista3d" / "center_2d_center")
run_inference_vista3d(json_file=json_file_2d, output_folder=output_folder, prompt_type=["center"],
                      prompt_mode="2d_prompts", number_prompts=number_prompts,
                      number_random_prompts=random_number_prompts,
                      initial_frame_selection=["center"],
                      number_additional_frames=number_additional_frames, gap_between_frames=gap_between_frames,
                      debug=True)
