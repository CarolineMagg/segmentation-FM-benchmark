from pathlib import Path

from src.inference.med_sam2.med_sam2_inference import run_inference_med_sam2
from src.project_root import PROJECT_ROOT

json_file2d = Path(PROJECT_ROOT/ "assets" / "demo" / "prompts" / "prompts_2d.json")
json_file3d = Path(PROJECT_ROOT/ "assets" / "demo" / "prompts" / "prompts_3d.json")
random_number_prompts = 1
number_prompts = 1
initial_frame_selection2d = ["center"]
initial_frame_selection3d = ["none"]
prompt_mode_3d = "3d_prompts"
prompt_mode_2d = "2d_prompts"
prompt_mode_3d_from_2d = "3d_prompts_from_2d"
number_random_frames = 0
number_additional_frames = 0
gap_between_frames = 0
equally_distributed_random_frames = False
use_volume_limits = True
not_use_volume_limits = False
model_type = "MedSAM2_latest"
model_type1 = "MedSAM2_2411"

output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "med_sam2_2411" / "bbox_slice_2")
run_inference_med_sam2(json_file=json_file3d, output_folder=output_folder, prompt_type=["bbox"],
                       prompt_mode=prompt_mode_3d_from_2d, number_prompts=number_prompts,
                       number_random_prompts=random_number_prompts,
                       number_additional_frames=number_additional_frames,
                       gap_between_frames=gap_between_frames, model_type=model_type1,
                       initial_frame_selection=["sliceR_2"],
                       use_volume_limits=not_use_volume_limits, debug=True)

output_folder = Path(PROJECT_ROOT/ "assets" / "demo" / "output" / "med_sam2_latest" / "center_3d")
run_inference_med_sam2(json_file=json_file3d, output_folder=output_folder, prompt_type=["center"],
                       prompt_mode=prompt_mode_3d, number_prompts=number_prompts,
                       number_random_prompts=random_number_prompts, number_additional_frames=number_additional_frames,
                       gap_between_frames=gap_between_frames, model_type=model_type1,
                       initial_frame_selection=initial_frame_selection3d,
                       use_volume_limits=not_use_volume_limits, debug=True)

output_folder = Path(PROJECT_ROOT/ "assets" / "demo" / "output" / "med_sam2_latest" / "bbox_slice_2")
run_inference_med_sam2(json_file=json_file3d, output_folder=output_folder, prompt_type=["bbox"],
                       prompt_mode=prompt_mode_3d_from_2d, number_prompts=number_prompts,
                       number_random_prompts=random_number_prompts,
                       number_additional_frames=number_additional_frames,
                       gap_between_frames=gap_between_frames, model_type=model_type,
                       initial_frame_selection=["sliceR_2"],
                       use_volume_limits=not_use_volume_limits, debug=True)

output_folder = Path(PROJECT_ROOT/ "assets" / "demo" / "output" / "med_sam2_latest" / "center_2d_largest")
run_inference_med_sam2(json_file2d, output_folder=output_folder, prompt_type=["center"],
                       prompt_mode=prompt_mode_2d, number_prompts=number_prompts,
                       number_random_prompts=random_number_prompts,
                       number_additional_frames=number_additional_frames,
                       gap_between_frames=gap_between_frames, model_type=model_type,
                       initial_frame_selection=["largest"],
                       use_volume_limits=not_use_volume_limits, debug=True)
