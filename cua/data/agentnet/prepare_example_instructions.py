import jsonlines
from huggingface_hub import snapshot_download


if __name__ == "__main__":
    # snapshot_download("xlangai/AgentNet", allow_patterns=["agentnet_ubuntu_5k.jsonl"],
    #                   repo_type="dataset", local_dir="./raw_data")

    with jsonlines.open("./raw_data/agentnet_ubuntu_5k.jsonl") as f:
        instructions = [s["instruction"] for s in list(f)]

    with open("./processed/instructions.txt", "w") as f:
        f.write("\n".join(instructions))



