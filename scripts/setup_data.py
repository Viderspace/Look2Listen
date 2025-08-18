#!/usr/bin/env python3
"""
Download and setup required datasets for AV-Speech Enhancement.

This script downloads:
- MUSAN noise dataset
- AVSpeech metadata
"""

import tarfile
from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm

# Dataset URLs and info
DATASETS = {
    "musan": {
        "url": "https://www.openslr.org/resources/17/musan.tar.gz",
        "size": "11GB",
        "extract_to": "data/musan",
        "description": "MUSAN noise dataset for augmentation",
    },
    "avspeech_metadata": {
        "url": "https://looking-to-listen.github.io/avspeech/",  # Replace with actual URL
        "size": "500MB",
        "extract_to": "data/avspeech_metadata.jsonl",
        "description": "AVSpeech face detection metadata",
        "torrent": "magnet:?xt=urn:btih:%EF%BF%BDx%EF%BF%BD\%EF%BF%BDG%EF%BF%BD%EF%BF%BD%EF%BF%BD~%EF%BF%BD*4%EF%BF%BD1%EF%BF%BD%EF%BF%BD]%EF%BF%BDA&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce",
    },
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path):
    """Download file with progress bar."""
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=output_path.name
    ) as t:
        urlretrieve(url, output_path, reporthook=t.update_to)


def setup_musan(data_dir: Path):
    """Download and extract MUSAN dataset."""
    musan_dir = data_dir / "musan"

    if musan_dir.exists() and any(musan_dir.rglob("*.wav")):
        print("✓ MUSAN dataset already exists")
        return

    print("\nMUSAN Noise Dataset")
    print("-" * 40)
    print(f"Size: {DATASETS['musan']['size']}")
    print("Used for: Audio augmentation during training")

    response = input("\nDownload MUSAN dataset? [y/N]: ")
    if response.lower() != "y":
        print("Skipping MUSAN download")
        print("You can manually download from:")
        print(f"  {DATASETS['musan']['url']}")
        return

    # Download
    tar_path = data_dir / "musan.tar.gz"
    print(f"\nDownloading to {tar_path}...")
    download_file(DATASETS["musan"]["url"], tar_path)

    # Extract
    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_dir)

    # Clean up
    tar_path.unlink()
    print("✓ MUSAN setup complete")


def setup_metadata(data_dir: Path):
    """Download AVSpeech metadata."""
    metadata_path = data_dir / "avspeech_metadata.jsonl"

    if metadata_path.exists():
        print("✓ AVSpeech metadata already exists")
        return

    print("\nAVSpeech Metadata")
    print("-" * 40)
    print(f"Size: {DATASETS['avspeech_metadata']['size']}")
    print("Contains: Face detection coordinates for all videos")

    print("\nDownload options:")
    print("1. Direct download (if available)")
    print("2. Torrent (recommended for large file)")
    print("3. Skip (download manually)")

    choice = input("\nChoice [1/2/3]: ")

    if choice == "1":
        # Direct download
        url = DATASETS["avspeech_metadata"].get("url")
        if url and not url.startswith("https://example.com"):
            print(f"Downloading {metadata_path.name}...")
            download_file(url, metadata_path)
        else:
            print("Direct download URL not configured")
            print("Please download manually from the AVSpeech website")

    elif choice == "2":
        # Torrent
        print("\nTorrent download:")
        print(f"  {DATASETS['avspeech_metadata']['torrent']}")
        print(f"\nSave to: {metadata_path}")
        print("Open the magnet link in your torrent client")
        input("Press Enter when download is complete...")

    else:
        print("Skipping metadata download")
        print("Download manually and place at:")
        print(f"  {metadata_path}")


def create_default_config(project_root: Path):
    """Create default configuration file."""
    config_path = project_root / "config.yaml"

    if config_path.exists():
        return

    config_content = """# AV-Speech Enhancement Configuration

# Data paths (automatically configured)
data:
  musan_dir: data/musan
  metadata_path: data/avspeech_metadata.jsonl

# These can be overridden with CLI arguments
defaults:
  sample_rate: 16000
  augmentation: music
"""

    config_path.write_text(config_content)
    print(f"✓ Created default config: {config_path}")


def main():
    print("=" * 50)
    print("AV-Speech Enhancement - Data Setup")
    print("=" * 50)

    # Create data directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    # Setup each dataset
    setup_musan(data_dir)
    setup_metadata(data_dir)

    # Create config
    create_default_config(project_root)

    print("\n" + "=" * 50)
    print("Setup Summary")
    print("=" * 50)

    # Check what's available
    musan_exists = (data_dir / "musan").exists()
    metadata_exists = (data_dir / "avspeech_metadata.jsonl").exists()

    print(f"MUSAN dataset:    {'✓' if musan_exists else '✗'}")
    print(f"AVSpeech metadata: {'✓' if metadata_exists else '✗'}")

    if musan_exists and metadata_exists:
        print("\n✓ All datasets ready!")
        print("You can now run: python scripts/prepare_dataset.py")
    else:
        print("\n⚠ Some datasets are missing")
        print("Download them manually or run this script again")


if __name__ == "__main__":
    main()
