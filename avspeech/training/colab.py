import os
import tarfile
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from avspeech.utils.structs import SampleT
from dataclasses import dataclass
import torch.nn as nn
from avspeech.training.training_phase import TrainingPhase


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

    def __init__(self, download_descriptors: List[DatasetDownloadDescriptor], local_base_dir="/content/datasets"):
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

        print(f"============== PreStagingManager - {len(self.train_sets) + len(self.validation_sets)} datasets prepared ==============")
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
                    "gsutil", "-m",
                    "-o", "GSUtil:parallel_thread_count=32",
                    "-o", "GSUtil:sliced_object_download_max_components=32",
                    "-o", "GSUtil:check_hashes=never",
                    "cp", *uris, f"{str(local_dir)}/"
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
                cmd = ["tar", "-I", "pigz", "-xf", str(tar_path), "-C", str(temp_extract_dir)]
            else:
                cmd = ["tar", "-xf", str(tar_path), "-C", str(temp_extract_dir)]
            subprocess.run(cmd, check=True)

            # Flatten: move all sample directories to the main directory
            self._flatten_extracted_content(temp_extract_dir, local_dir)

            # Clean up
            shutil.rmtree(temp_extract_dir)
            tar_path.unlink(missing_ok=True)


    def _flatten_extracted_content(self, temp_dir: Path, target_dir: Path) -> None:
        """
        Simple flattening assuming structure: extracted_folder/sample_directories/
        """
        sample_count = 0

        # Look for directories that contain sample directories
        for extracted_folder in temp_dir.iterdir():
            if extracted_folder.is_dir():
                # Move all subdirectories (samples) to target
                for sample_dir in extracted_folder.iterdir():
                    if sample_dir.is_dir():
                        target_path = target_dir / sample_dir.name
                        if target_path.exists():
                            # Handle name conflicts (should not happen)
                            counter = 1
                            while (target_dir / f"{sample_dir.name}_{counter}").exists():
                                counter += 1
                            target_path = target_dir / f"{sample_dir.name}_{counter}"

                        shutil.move(str(sample_dir), str(target_path))
                        sample_count += 1

        print(f"  → Moved {sample_count} samples to {target_dir}")

    # TODO - Private asset - Remove this when not needed
def get_preconfigured_datasets() -> List[DatasetDownloadDescriptor]:
    """
    Returns a list of preconfigured DatasetDownloadDescriptors for downloading datasets.
    """

    # Training datasets
    s1_noise_descriptor = DatasetDownloadDescriptor(
            sample_type=SampleT.S1_NOISE,
            gcs_files=['gs://av_speech_60k_dataset/avspeech_1s_noise_subset10k.tar.gz',
                       'gs://av_speech_60k_dataset/avspeech_1s_noise_subset5k_additional.tar.gz'],
            is_validation=False
    )
    s2_clean_descriptor = DatasetDownloadDescriptor(
            sample_type=SampleT.S2_CLEAN,
            gcs_files=[
                    'gs://av_speech_2s_clean_14k/2s_clean.tar.gz',
                    'gs://av_speech_2s_clean_14k/2s_clean_al_an.tar.gz',
            ],
            is_validation=False
    )
    s2_noise_descriptor = DatasetDownloadDescriptor(
            sample_type=SampleT.S2_NOISE,
            gcs_files=[
                    'gs://av_speech_2s_clean_14k/2s_noise.tar.gz',
                    'gs://av_speech_2s_clean_14k/2s_noise_ao_aq.tar.gz',
            ],
            is_validation=False
    )

    # Validation datasets
    s1_noise_validation_descriptor = DatasetDownloadDescriptor(
            sample_type=SampleT.S1_NOISE,
            gcs_files=['gs://av_speech_validation/1s_noise.tar.gz'],
            is_validation=True
    )
    s2_clean_validation_descriptor = DatasetDownloadDescriptor(
            sample_type=SampleT.S2_CLEAN,
            gcs_files=['gs://av_speech_validation/2s_clean.tar.gz'],
            is_validation=True
    )
    s2_noise_validation_descriptor = DatasetDownloadDescriptor(
            sample_type=SampleT.S2_NOISE,
            gcs_files=['gs://av_speech_validation/2s_noise.tar.gz'],
            is_validation=True
    )

    return [s1_noise_descriptor,
            s2_clean_descriptor,
            s2_noise_descriptor,
            s1_noise_validation_descriptor,
            s2_clean_validation_descriptor,
            s2_noise_validation_descriptor]

