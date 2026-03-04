from pathlib import Path

from src.inference.sam2.sam2_inference_video import run_inference_sam2_video
from src.project_root import PROJECT_ROOT

json_file2d = Path(PROJECT_ROOT / "assets" / "demo" / "prompts" / "prompts_2d.json")
json_file3d = Path(PROJECT_ROOT / "assets" / "demo" / "prompts" / "prompts_3d.json")
random_number_prompts = 1
number_prompts = 5
model_type = "sam2_new_hiera_tiny"
initial_frame_selection2d = ["center"]
initial_frame_selection3d = ["none"]
prompt_mode_3d = "3d_prompts"
prompt_mode_2d = "2d_prompts"
prompt_mode_3d_from_2d = "3d_prompts_from_2d"
number_additional_frames = 0
gap_between_frames = 0
not_use_volume_limits = False
use_volume_limits = True

# output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "sam2_new_hiera_tiny_video" / "bbox_slice_2")
# run_inference_sam2_video(json_file=json_file3d, output_folder=output_folder, prompt_type=["bbox"],
#                          prompt_mode=prompt_mode_3d_from_2d, number_prompts=number_prompts,
#                          number_random_prompts=random_number_prompts,
#                          number_additional_frames=number_additional_frames,
#                          gap_between_frames=gap_between_frames, model_type=model_type,
#                          initial_frame_selection=["sliceR_2"],
#                          use_volume_limits=not_use_volume_limits, debug=True)
#
# output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "sam2_new_hiera_tiny_video" / "center_3d")
# run_inference_sam2_video(json_file=json_file3d, output_folder=output_folder, prompt_type=["center"],
#                          prompt_mode=prompt_mode_3d, number_prompts=number_prompts,
#                          number_random_prompts=random_number_prompts, number_additional_frames=number_additional_frames,
#                          gap_between_frames=gap_between_frames, model_type=model_type,
#                          initial_frame_selection=initial_frame_selection3d,
#                          use_volume_limits=not_use_volume_limits, debug=True)

output_folder = Path(
    PROJECT_ROOT / "assets" / "demo" / "output" / "sam2_new_hiera_tiny_video" / "bbox_center_2d_center")
run_inference_sam2_video(json_file2d, output_folder=output_folder, prompt_type=["bbox", "center"],
                         prompt_mode=prompt_mode_2d, number_prompts=1,
                         number_random_prompts=1,
                         number_additional_frames=number_additional_frames,
                         gap_between_frames=gap_between_frames,
                         model_type=model_type, initial_frame_selection=["center"],
                         use_volume_limits=not_use_volume_limits, debug=True)
