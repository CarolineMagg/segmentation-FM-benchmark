from pathlib import Path

from src.inference.seg_vol.seg_vol_inference import run_inference_segvol
from src.project_root import PROJECT_ROOT

json_file = Path(PROJECT_ROOT / "assets" / "demo" / "prompts" / "prompts_3d.json")
json_file2 = Path(PROJECT_ROOT / "assets" / "demo" / "prompts" / "prompts_2d.json")
number_prompts = 1
random_number_prompts = 1
prompt_mode = "3d_prompts"
prompt_mode_2d = "2d_prompts"
initial_frame_selection3d = ["none"]
initial_frame_selection2d = ["center"]
number_additional_frames = 0
gap_between_frames = 0

output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "seg_vol" / "bbox_3d")
run_inference_segvol(json_file=json_file, output_folder=output_folder, prompt_type=["bbox"],
                     prompt_mode=prompt_mode, number_prompts=number_prompts,
                     number_random_prompts=random_number_prompts,
                     initial_frame_selection=initial_frame_selection3d,
                     number_additional_frames=number_additional_frames, gap_between_frames=gap_between_frames,
                     use_zoom_in=True, debug=True)

output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "seg_vol" / "center_3d")
run_inference_segvol(json_file=json_file, output_folder=output_folder, prompt_type=["center"],
                     prompt_mode=prompt_mode, number_prompts=number_prompts,
                     number_random_prompts=random_number_prompts,
                     initial_frame_selection=initial_frame_selection3d,
                     number_additional_frames=number_additional_frames, gap_between_frames=gap_between_frames,
                     use_zoom_in=True, debug=True)

output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "seg_vol" / "center_3d_not_zoom_in")
run_inference_segvol(json_file=json_file, output_folder=output_folder, prompt_type=["center"],
                     prompt_mode=prompt_mode, number_prompts=number_prompts,
                     number_random_prompts=random_number_prompts,
                     initial_frame_selection=initial_frame_selection3d,
                     number_additional_frames=number_additional_frames, gap_between_frames=gap_between_frames,
                     use_zoom_in=False, debug=True)
