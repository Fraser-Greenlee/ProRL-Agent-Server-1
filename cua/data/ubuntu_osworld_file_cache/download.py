from huggingface_hub import snapshot_download


if __name__ == '__main__':
    # cache file used in OS World Setup
    # Original code downloaded cache files from HF everytime we booted up the VMs, but that leads to frequent 429 errors
    # (too many requests).
    # Instead, when the download URL is right, we simply move the matched file from here to the correct VM-mapped dir.
    snapshot_download("xlangai/ubuntu_osworld_file_cache", repo_type="dataset",
                      allow_patterns=["*"], local_dir=".")

