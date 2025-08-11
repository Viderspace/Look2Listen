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
        Fast path:
          - Bulk copy all tars in one command (gcloud storage cp).
          - Fallback: gsutil -m with sliced downloads and no hash checks (no crcmod).
          - Extract with system `tar` (optionally pigz), then delete archives.
        """
        local_dir = descriptor.get_local_dir(self.local_base_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        uris = descriptor.gcs_files
        # --- 1) Download (prefer gcloud; fallback to tuned gsutil) ---
        if shutil.which("gcloud"):
            # Modern, parallel by default; this matched your 121 MB/s test
            cmd = ["gcloud", "storage", "cp", *uris, f"{str(local_dir)}/"]
            print("\n[Download] gcloud storage cp …")
            subprocess.run(cmd, check=True)
        else:
            # Fallback: high‑throughput gsutil without crcmod
            # Note: -o flags must be on the same line as gsutil
            cmd = [
                    "gsutil", "-m",
                    "-o", "GSUtil:parallel_thread_count=32",
                    "-o", "GSUtil:sliced_object_download_max_components=32",
                    "-o", "GSUtil:check_hashes=never",
                    "cp", *uris, f"{str(local_dir)}/"
            ]
            print("\n[Download] gsutil tuned (parallel + sliced, no hashes) …")
            subprocess.run(cmd, check=True)

        # --- 2) Extract (system tar is faster than Python tarfile) ---
        # If pigz is present, tar will use it for parallel gunzip.
        use_pigz = shutil.which("pigz") is not None
        tar_decomp_flag = ["-I", "pigz"] if use_pigz else ["-z"]  # -z == gzip

        for tar_path in sorted(local_dir.glob("*.tar.gz")):
            print(f"[Extract] {tar_path.name} → {local_dir}")
            # tar -x (extract), -f file, -C target dir
            # If pigz is installed, let tar call pigz to parallelize decompression.
            if use_pigz:
                cmd = ["tar", *tar_decomp_flag, "-xvf", str(tar_path), "-C", str(local_dir)]
            else:
                cmd = ["tar", "-xvf", str(tar_path), "-C", str(local_dir)]
            subprocess.run(cmd, check=True)
            tar_path.unlink(missing_ok=True)


    # TODO - Private asset - Remove this when not needed
def get_preconfigured_datasets() -> List[DatasetDownloadDescriptor]:
    """
    Returns a list of preconfigured DatasetDownloadDescriptors for downloading datasets.
    """

    # Training datasets
    s1_noise_descriptor = DatasetDownloadDescriptor(
            sample_type=SampleT.S1_NOISE,
            gcs_files=['gs://av_speech_60k_dataset/avspeech_1s_noise_subset10k.tar.gz'],
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

