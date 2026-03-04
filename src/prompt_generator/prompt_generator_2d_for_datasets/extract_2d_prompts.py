import json
from pathlib import Path

import numpy as np

from src.project_root import PROJECT_ROOT
from src.prompt_generator.extract_prompts import generate_2d_prompts_for_folder


def main(suffix = "/mnt/caroline_amc_storage"):
    # extract 2d prompts (box, centroid, center, pos + neg slices)
    np.random.seed(42)
    path_labels = Path(PROJECT_ROOT/ "assets" / "demo" / "labels")
    json_file = Path(PROJECT_ROOT/ "assets" / "demo" / "prompts" / "prompts_2d.json")
    all_prompts = generate_2d_prompts_for_folder(path_labels, [1, 2, 3, 4])
    all_prompts[
        "image_path"] = str(Path(PROJECT_ROOT/ "assets" / "demo" / "images"))

    # store file
    with open(json_file, "w") as f:
        json.dump(all_prompts, f)
    print(f"write {json_file}")


if __name__ == "__main__":
    main()
