import os
from pathlib import Path

import ipdb
from tqdm import tqdm


# zenodo10k pptx files have spaces in the filenames, change the whitespace to underbar (_).

if __name__ == "__main__":
    pptx_root_dir = Path("./raw_data/pptx")
    pptx_paths = pptx_root_dir.rglob("*.pptx")

    for path in tqdm(pptx_paths):
        new_name = path.name.replace(" ", "_")
        new_path = path.parent / new_name
        os.rename(str(path), str(new_path))


