import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.project_root import PROJECT_ROOT
from src.prompt_generator.extract_prompts import generate_3d_prompts_for_folder


def main(suffix="/mnt/caroline_amc_storage"):
    # extract 3d prompts (box, centroid, center, random slices)
    np.random.seed(42)
    df_path = Path(PROJECT_ROOT/ "assets" / "demo" / "prompts" / "dataset_overview.csv")
    path_2d_prompts = Path(PROJECT_ROOT/ "assets" / "demo" / "prompts" / "prompts_2d.json")
    df = pd.read_csv(df_path)
    with open(path_2d_prompts, "r") as f:
        json_2d_prompts = json.load(f)
    path_labels = Path(PROJECT_ROOT/ "assets" / "demo" / "labels")
    json_file = Path(PROJECT_ROOT/ "assets" / "demo" / "prompts" / "prompts_3d.json")
    all_prompts = generate_3d_prompts_for_folder(path_labels, [1, 2, 3, 4],
                                                 df=df, json_2d_prompts=json_2d_prompts)
    all_prompts[
        "image_path"] = str(Path(PROJECT_ROOT/ "assets" / "demo" / "images"))

    # store file
    with open(json_file, "w") as f:
        json.dump(all_prompts, f)
    print(f"write {json_file}")


if __name__ == "__main__":
    main()
