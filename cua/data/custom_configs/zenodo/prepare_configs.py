import random
import uuid
from collections import defaultdict
from pathlib import Path

import ipdb
import jsonlines
from datasets import load_dataset
from tqdm import tqdm


if __name__ == "__main__":
    parquet_file_path = Path("/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/cua/prorl-agent-server-v2/cua/data/custom_configs/zenodo/raw_data/data")
    save_filename = Path("./osworld_setup_configs.jsonl")

    # load dataset
    ds = load_dataset(str(parquet_file_path), split="train")

    viable_licenses = {
        "mit-license", "apache2.0", "cc-zero", "cc-pddc", "cc-by-4.0", "cc-by-3.0", "cc-by-3.0-us", "cc-by-3.0-at",
        "cc-by-2.0", "cc-by"
    }
    ds = ds.filter(lambda x: x['license'] in viable_licenses)

    # group pptx files according to their shared urls
    pptx_groups = defaultdict(list)  # "id": list of dicts corresponding to the group
    for line in tqdm(ds):
        if not line['url'].startswith("https://zenodo.org/api/records/"):
            continue
        group_id = line["url"][len("https://zenodo.org/api/records/"):].split("/")[0]

        pptx_file_dir = Path(f"./raw_data/pptx/{line['license']}/{line['created'][:4]}")
        pptx_file_path = pptx_file_dir / f"{line['checksum'][4:]}-{line['filename'].replace(' ', '_')}"

        if not pptx_file_path.exists():
            pptx_file_path = pptx_file_dir / (str(pptx_file_path.name)[:240] + ".pptx")
            if not pptx_file_path.exists():
                ipdb.set_trace()
                pass

        pptx_groups[group_id].append({
            "pptx_file_path": pptx_file_path,
            "license": line["license"],
            "zenodo10k_url": line["url"],
        })

    # prepare configs
    configs = []
    for group_id, group_pptx_list in pptx_groups.items():
        config_list = []
        vm_dest_paths = []
        # upload all files to the VM
        for pptx_dict in group_pptx_list:
            vm_dest_path = f"/home/user/Desktop/{pptx_dict['pptx_file_path'].name}"
            config_list.append({
                "type": "upload_file",
                "parameters": {
                    "files": [
                        {
                            "local_path": str(pptx_dict["pptx_file_path"].absolute()),
                            "path": vm_dest_path,
                        }
                    ]
                }
            })
            vm_dest_paths.append(vm_dest_path)

        # open up to 3 files inside VM
        num_files_to_open = min(len(vm_dest_paths), random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2], k=1)[0])
        files_to_open = random.sample(vm_dest_paths, num_files_to_open)
        for vm_dest_path in files_to_open:
            config_list.append({
                "type": "open",
                "parameters": {
                    "path": vm_dest_path,
                }
            })
            config_list.append({
                "type": "sleep",
                "parameters": {
                    "seconds": 3
                }
            })

        # metadata - add license, zenodo10k_url
        metadata = [{"license": s['license'], "zenodo10k_url": s['zenodo10k_url']} for s in group_pptx_list]

        # finalize config
        config = {
            "id": f'zenodo:{group_id}',
            "snapshot": "libreoffice_impress",
            "instruction": "",
            "source": "zenodo10k",
            "metadata": metadata,
            "config": config_list,
            "related_apps": ["libreoffice_impress"],
            "evaluator": {"func": "infeasible"},
        }
        configs.append(config)

    with jsonlines.open(save_filename, "w") as f:
        f.write_all(configs)

    print(f"Saved {len(configs)} configs to {save_filename}.")
    print(f"Example: {configs[0]}")
