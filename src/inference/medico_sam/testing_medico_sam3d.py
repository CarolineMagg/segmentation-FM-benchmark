from pathlib import Path

from src.inference.medico_sam.medico_sam3d_inference import run_inference_medicosam3d
from src.project_root import PROJECT_ROOT

number_prompts = 1
random_number_prompts = 1
number_additional_frames = 0
gap_between_frames = 0

json_file = Path(PROJECT_ROOT / "assets" / "demo" / "prompts" / "prompts_3d.json")
prompt_mode = "3d_prompts"
initial_frame_selection3d = ["none"]

prompt_mode2 = "2d_prompts"
json_file2 = Path(PROJECT_ROOT / "assets" / "demo" / "prompts" / "prompts_2d.json")
initial_frame_selection2d = ["center"]

output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "medico_sam3d" / "center_3d")
prompt_type = ["center"]
run_inference_medicosam3d(json_file=json_file, output_folder=output_folder, prompt_type=prompt_type,
                          prompt_mode=prompt_mode, number_prompts=number_prompts,
                          number_random_prompts=random_number_prompts,
                          initial_frame_selection=initial_frame_selection3d,
                          number_additional_frames=number_additional_frames, gap_between_frames=gap_between_frames,
                          model_type="vit_b_medical_imaging", debug=True)

output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "medico_sam3d" / "bbox_2d_center_slice")
prompt_type = ["bbox"]
run_inference_medicosam3d(json_file=json_file2, output_folder=output_folder, prompt_type=prompt_type,
                          prompt_mode=prompt_mode2, number_prompts=number_prompts,
                          number_random_prompts=random_number_prompts,
                          initial_frame_selection=initial_frame_selection2d,
                          number_additional_frames=number_additional_frames, gap_between_frames=gap_between_frames,
                          model_type="vit_b_medical_imaging", debug=True)

output_folder = Path(PROJECT_ROOT / "assets" / "demo" / "output" / "medico_sam3d" / "bbox_center_2d_center_slice")
prompt_type = ["bbox", "center"]
run_inference_medicosam3d(json_file=json_file2, output_folder=output_folder, prompt_type=prompt_type,
                          prompt_mode=prompt_mode2, number_prompts=number_prompts,
                          number_random_prompts=random_number_prompts,
                          initial_frame_selection=initial_frame_selection2d,
                          number_additional_frames=number_additional_frames, gap_between_frames=gap_between_frames,
                          model_type="vit_b_medical_imaging", debug=True)
