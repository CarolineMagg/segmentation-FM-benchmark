from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from src.inference.sam3.sam3_inference import run_inference_sam3

json_file = Path(
    "/mnt/caroline_amc_storage/qurai/ProjectData/Muscoloskeletal lab/SAM_reader_study/prompts/516_shoulder_prompts_2d.json")
output_folder = Path(
    "/mnt/caroline_amc_storage/qurai/ProjectData/Muscoloskeletal lab/SAM_reader_study/framework_test/sam3_image/shoulder2/bbox_single/")
prompt_type = ["bbox"]
number_prompts = 1
random_number_prompts = 1
run_inference_sam3(json_file=json_file, output_folder=output_folder, prompt_type=prompt_type,
                   number_prompts=number_prompts, number_random_prompts=random_number_prompts,
                   debug=True)

output_folder = Path(
    "/mnt/caroline_amc_storage/qurai/ProjectData/Muscoloskeletal lab/SAM_reader_study/framework_test/sam3_image/shoulder2/center_single/")
prompt_type = ["center"]
number_prompts = 1
random_number_prompts = 1
run_inference_sam3(json_file=json_file, output_folder=output_folder, prompt_type=prompt_type,
                   number_prompts=number_prompts, number_random_prompts=random_number_prompts,
                   debug=True)

output_folder = Path(
    "/mnt/caroline_amc_storage/qurai/ProjectData/Muscoloskeletal lab/SAM_reader_study/framework_test/sam3_image/shoulder2/combi_single/")
prompt_type = ["bbox", "center"]
number_prompts = 1
random_number_prompts = 1
run_inference_sam3(json_file=json_file, output_folder=output_folder, prompt_type=prompt_type,
                   number_prompts=number_prompts, number_random_prompts=random_number_prompts,
                   debug=True)
