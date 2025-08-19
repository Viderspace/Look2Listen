import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from avspeech.utils.structs import SampleT


@dataclass
class DatasetDownloadDescriptor:
    """
    Descriptor for downloading datasets from GCS buckets.
    Contains the GCS URIs and local paths for each dataset.
    """

    sample_type: SampleT
    gcs_files: List[str]
    is_validation: bool = False

    def get_local_dir(self, notebook_root: Path) -> Path:
        """
        Returns the local path where the dataset will be stored.
        """
        if self.is_validation:
            return notebook_root / "validation" / self.sample_type.value

        return notebook_root / "train" / self.sample_type.value


class PreStagingManager:
    """
    Downloads and extracts datasets from GCS buckets to local storage.
    Copied as-is from your latest version (only moved into this file).
    """

    def __init__(
        self,
        download_descriptors: List[DatasetDownloadDescriptor],
        local_base_dir="/content/datasets",
    ):
        self.download_descriptors = download_descriptors
        self.local_base_dir = Path(local_base_dir)
        self.train_sets: Dict[SampleT, Path] = {}
        self.validation_sets: Dict[SampleT, Path] = {}

    def prepare_all_datasets(self) -> Tuple[Dict[SampleT, Path], Dict[SampleT, Path]]:
        print("============== PreStagingManager - Preparing datasets ==============")
        for descriptor in self.download_descriptors:
            dataset_path = descriptor.get_local_dir(self.local_base_dir)
            # If the dataset already exists, skip downloading
            if not os.path.exists(str(dataset_path)):
                self._download_and_extract(descriptor)

            # Add to the appropriate dictionary based on validation status
            if descriptor.is_validation:
                self.validation_sets[descriptor.sample_type] = dataset_path
            else:
                self.train_sets[descriptor.sample_type] = dataset_path

        print(
            f"============== PreStagingManager - {len(self.train_sets) + len(self.validation_sets)} datasets prepared =============="
        )
        return self.train_sets, self.validation_sets

    #
    # def _download_and_extract(self, descriptor: DatasetDownloadDescriptor) -> None:
    #     """
    #     1. Download the dataset tar.gs files from GCS (Google cloud storage) bucket, and extract them to the local directory.
    #     2. Extracts the tar.gz files into the local directory.
    #     3. Deletes the tar.gz files after extraction.
    #
    #     :param descriptor: DatasetDownloadDescriptor containing GCS URIs and local path.
    #     """
    #     local_dir = descriptor.get_local_dir(self.local_base_dir)
    #     local_dir.mkdir(parents=True, exist_ok=True)
    #     for gcs_uri in descriptor.gcs_files:
    #         tar_path = local_dir / Path(gcs_uri).name
    #         print(f"\nDownloading {gcs_uri} → {tar_path}")
    #         subprocess.run(["gsutil", "cp", gcs_uri, str(tar_path)], check=True)
    #         print(f"Extracting {tar_path}…")
    #         with tarfile.open(tar_path, 'r:gz') as tf:
    #             tf.extractall(local_dir)
    #         tar_path.unlink(missing_ok=True)

    def _download_and_extract(self, descriptor: DatasetDownloadDescriptor) -> None:
        """
        Fast path with flattening:
          - Bulk copy all tars in one command (gcloud storage cp).
          - Extract each tar to a temporary directory
          - Move all sample directories to the final location (flattened)
          - Clean up temporary directories and archives
        """
        local_dir = descriptor.get_local_dir(self.local_base_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        uris = descriptor.gcs_files

        # --- 1) Download ---
        if shutil.which("gcloud"):
            cmd = ["gcloud", "storage", "cp", *uris, f"{str(local_dir)}/"]
            print("\n[Download] gcloud storage cp …")
            subprocess.run(cmd, check=True)
        else:
            cmd = [
                "gsutil",
                "-m",
                "-o",
                "GSUtil:parallel_thread_count=32",
                "-o",
                "GSUtil:sliced_object_download_max_components=32",
                "-o",
                "GSUtil:check_hashes=never",
                "cp",
                *uris,
                f"{str(local_dir)}/",
            ]
            print("\n[Download] gsutil tuned (parallel + sliced, no hashes) …")
            subprocess.run(cmd, check=True)

        # --- 2) Extract and flatten ---
        use_pigz = shutil.which("pigz") is not None

        for tar_path in sorted(local_dir.glob("*.tar.gz")):
            print(f"[Extract & Flatten] {tar_path.name} → {local_dir}")

            # Create temporary extraction directory
            temp_extract_dir = local_dir / f"temp_{tar_path.stem}"
            temp_extract_dir.mkdir(exist_ok=True)

            # Extract to temporary directory
            if use_pigz:
                cmd = [
                    "tar",
                    "-I",
                    "pigz",
                    "-xf",
                    str(tar_path),
                    "-C",
                    str(temp_extract_dir),
                ]
            else:
                cmd = ["tar", "-xf", str(tar_path), "-C", str(temp_extract_dir)]
            subprocess.run(cmd, check=True)

            # Flatten: move all sample directories to the main directory
            self._flatten_extracted_content(temp_extract_dir, local_dir)

            # Clean up
            shutil.rmtree(temp_extract_dir)
            tar_path.unlink(missing_ok=True)

    # ======== Flattening logic for extracted content ================ (Both flat and nested structures)

    def _flatten_extracted_content(self, temp_dir: Path, target_dir: Path) -> None:
        """
        Move sample directories from temp_dir to target_dir, handling both:
        - Flat structure: sample directories at root level
        - Nested structure: sample directories inside master folders
        """
        direct_items = list(temp_dir.iterdir())
        first_dir = next((item for item in direct_items if item.is_dir()), None)

        if not first_dir:
            print("  → No directories found in extracted content")
            return

        # Determine structure and get all sample directories
        if self._is_sample_directory(first_dir):
            print("  → Detected flat structure (no master folder)")
            sample_dirs = [item for item in direct_items if item.is_dir()]
        else:
            print("  → Detected nested structure (with master folder)")
            sample_dirs = self._get_nested_samples(direct_items)

        # Move all samples to target
        moved_count = self._move_samples(sample_dirs, target_dir)
        print(f"  → Moved {moved_count} samples to {target_dir}")

    def _is_sample_directory(self, dir_path: Path) -> bool:
        """Check if a directory is a sample (contains audio/ or face/ subdirs)."""
        return (dir_path / "audio").is_dir() or (dir_path / "face").is_dir()

    def _get_nested_samples(self, master_folders: List[Path]) -> List[Path]:
        """Extract all sample directories from master folders."""
        samples = []
        for master in master_folders:
            if master.is_dir():
                samples.extend([item for item in master.iterdir() if item.is_dir()])
        return samples

    def _move_samples(self, sample_dirs: List[Path], target_dir: Path) -> int:
        """Move sample directories to target, handling name conflicts."""
        moved_count = 0
        for sample in sample_dirs:
            target_path = self._get_safe_target_path(sample.name, target_dir)
            shutil.move(str(sample), str(target_path))
            moved_count += 1
        return moved_count

    def _get_safe_target_path(self, name: str, target_dir: Path) -> Path:
        """Get a safe target path, adding suffix if name already exists."""
        target_path = target_dir / name
        if not target_path.exists():
            return target_path

        counter = 1
        while (target_dir / f"{name}_{counter}").exists():
            counter += 1
        return target_dir / f"{name}_{counter}"

    # =================================================================================

    # TODO - Private asset - Remove this when not needed


def get_preconfigured_datasets() -> List[DatasetDownloadDescriptor]:
    """
    Returns a list of preconfigured DatasetDownloadDescriptors for downloading datasets.
    """

    # Training datasets
    s1_noise_descriptor = DatasetDownloadDescriptor(
        sample_type=SampleT.S1_NOISE,
        gcs_files=[
            "gs://av_speech_60k_dataset/avspeech_1s_noise_subset10k.tar.gz",
            "gs://av_speech_60k_dataset/avspeech_1s_noise_subset5k_additional.tar.gz",
        ],
        is_validation=False,
    )
    s2c_tars = [f"gs://avspeech_s2c/S2C_45K_chunk_{i:03d}.tar.gz" for i in range(1, 33)]

    s2_clean_descriptor = DatasetDownloadDescriptor(
        sample_type=SampleT.S2_CLEAN, gcs_files=s2c_tars, is_validation=False
    )

    s2n_tars = [f"gs://avspeech_s2n/S2N_45K_chunk_{i:03d}.tar.gz" for i in range(1, 33)]
    s2_noise_descriptor = DatasetDownloadDescriptor(
        sample_type=SampleT.S2_NOISE,
        gcs_files=s2n_tars,
        is_validation=False,
    )

    # Validation datasets
    s1_noise_validation_descriptor = DatasetDownloadDescriptor(
        sample_type=SampleT.S1_NOISE,
        gcs_files=["gs://av_speech_validation/1s_noise.tar.gz"],
        is_validation=True,
    )
    s2_clean_validation_descriptor = DatasetDownloadDescriptor(
        sample_type=SampleT.S2_CLEAN,
        gcs_files=["gs://av_speech_validation/2s_clean.tar.gz"],
        is_validation=True,
    )
    s2_noise_validation_descriptor = DatasetDownloadDescriptor(
        sample_type=SampleT.S2_NOISE,
        gcs_files=["gs://av_speech_validation/2s_noise.tar.gz"],
        is_validation=True,
    )

    return [
        s1_noise_descriptor,
        s2_clean_descriptor,
        s2_noise_descriptor,
        s1_noise_validation_descriptor,
        s2_clean_validation_descriptor,
        s2_noise_validation_descriptor,
    ]
