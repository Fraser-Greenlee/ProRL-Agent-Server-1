"""
Generate osworld-compatible setup configs for SpreadsheetBench xlsx files.

Each config uploads an input xlsx to the VM desktop and opens it with LibreOffice Calc.
Works with both NVCF (DesktopEnv) and Singularity (KVM) runtimes.

Usage:
    python prepare_configs.py
Output:
    a jsonlines file of osworld configs (in an identical format as osworld_test_nogdrive.json), including
    {
        'id': 'spreadsheetbench:2_69-17_answer.xlsx',
        'snapshot': 'libreoffice_calc',
        'instruction': '',
        'source': 'spreadsheetbench',
        'config': [
            {
                'type': 'upload_file',
                'parameters': {
                    'files': [{
                        'local_path': '...',
                        'path': '/home/user/Desktop/2_69-17_answer.xlsx'}
                     ]
                 }
            },
            {
                'type': 'open',
                'parameters': {
                    'path': '/home/user/Desktop/2_69-17_answer.xlsx'}
            },
            {
                'type': 'sleep',
                'parameters': {'seconds': 3}
            }
        ],
        'related_apps': ['libreoffice_calc'],
        'evaluator': {'func': 'infeasible'}
    }
"""
from pathlib import Path

import jsonlines


if __name__ == "__main__":
    data_dir = Path("./all_data_912_v0.1")
    save_filename = Path("./osworld_setup_configs.jsonl")

    # find all xlsx files
    xlsx_file_paths = [path.absolute() for path in data_dir.rglob("*.xlsx")]

    # prepare configs
    configs = []
    for xlsx_file_path in xlsx_file_paths:
        if not xlsx_file_path.exists():
            continue
        else:
            vm_dest_path = f"/home/user/Desktop/{xlsx_file_path.name}"

            config = {
                "id": f"spreadsheetbench:{xlsx_file_path.name}",
                "snapshot": "libreoffice_calc",
                "instruction": "",
                "source": "spreadsheetbench",
                "config": [
                    {
                        "type": "upload_file",
                        "parameters": {
                            "files": [
                                {
                                    "local_path": str(xlsx_file_path),
                                    "path": vm_dest_path
                                }
                            ]
                        }
                    },
                    {
                        "type": "open",
                        "parameters": {
                            "path": vm_dest_path
                        }
                    },
                    {
                        "type": "sleep",
                        "parameters": {
                            "seconds": 3
                        }
                    },
                ],
                "related_apps": ["libreoffice_calc"],
                "evaluator": {"func": "infeasible"},
            }
            configs.append(config)

    with jsonlines.open(save_filename, "w") as f:
        f.write_all(configs)

    print(f"Saved {len(configs)} configs to {save_filename}.")
    print(f"Example: {configs[0]}")
