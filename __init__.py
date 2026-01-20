import importlib.util
import subprocess
import sys

def ensure_package(package_name, install_name=None):
    if install_name is None:
        install_name = package_name
    
    if importlib.util.find_spec(package_name) is None:
        print(f"[HeartMuLa] Package '{package_name}' not found. Installing '{install_name}'...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', install_name])
            print(f"[HeartMuLa] Successfully installed '{install_name}'.")
        except subprocess.CalledProcessError as e:
            print(f"[HeartMuLa] Failed to install '{install_name}'. Please install it manually.")
            raise e

# Check for critical dependencies
ensure_package("torchtune")
ensure_package("vector_quantize_pytorch")
ensure_package("torchao")
ensure_package("huggingface_hub")

from .nodes import (
    HeartMuLaModelLoader,
    HeartCodecLoader,
    HeartMuLaGenerator,
    HeartTranscriptorLoader,
    HeartTranscriptor,
)

NODE_CLASS_MAPPINGS = {
    "HeartMuLaModelLoader": HeartMuLaModelLoader,
    "HeartCodecLoader": HeartCodecLoader,
    "HeartMuLaGenerator": HeartMuLaGenerator,
    "HeartTranscriptorLoader": HeartTranscriptorLoader,
    "HeartTranscriptor": HeartTranscriptor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HeartMuLaModelLoader": "HeartMuLa Model Loader",
    "HeartCodecLoader": "HeartCodec Loader",
    "HeartMuLaGenerator": "HeartMuLa Music Generator",
    "HeartTranscriptorLoader": "HeartTranscriptor Loader",
    "HeartTranscriptor": "HeartTranscriptor",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
