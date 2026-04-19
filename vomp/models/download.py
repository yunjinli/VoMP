import os
from huggingface_hub import snapshot_download

def setup_checkpoints(path: str = "weights") -> str:
    """
    Downloads the VoMP model weights from Hugging Face.
    
    Args:
        target_path: The directory where the VoMP weights should be stored.
                     (e.g., 'weights/vomp' or just 'weights/')
    """
    # 1. Idempotency Check
    # The VoMP repo contains 'geometry_transformer.pt'. We check for its 
    # existence so we don't repeatedly ping Hugging Face if it's already downloaded.
    expected_file = os.path.join(path, "geometry_transformer.pt")
    
    if os.path.exists(expected_file):
        print(f"[VoMP] Checkpoints already exist at '{path}'. Skipping download.")
        return path

    print(f"[VoMP] Downloading weights to '{path}'...")
    os.makedirs(path, exist_ok=True)

    try:
        # 2. Download directly to the target directory
        # snapshot_download is atomic and handles incomplete downloads safely.
        snapshot_download(
            repo_id="nvidia/PhysicalAI-Simulation-VoMP-Model",
            repo_type="model",
            local_dir=path,
            max_workers=4,
            # We explicitly filter out READMEs, .gitattributes, and history,
            # only pulling the heavy PyTorch/Safetensor weights and JSON configs.
            allow_patterns=["*.pt", "*.safetensors", "*.json"] 
        )

    except Exception as e:
        print(f"[VoMP] ERROR downloading weights: {e}")
        raise

    print(f"[VoMP] Successfully set up checkpoints at '{path}'")
    return path
