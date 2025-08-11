#!/usr/bin/env python3
"""
previously - data_loader.py  TODO : remove cache logic and resume functionality
==============

DataLoader class for managing AVSpeech dataset loading and iteration.
Minimal implementation that replicates existing functionality exactly.
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Iterator, Optional
from dataclasses import dataclass
import logging


@dataclass
class ClipData:
    """Container for a single clip's data and metadata."""
    unique_clip_id: str
    video_path: str  # Store as string for JSON serialization
    clip_metadata: dict

    @property
    def video_path_obj(self) -> Path:
        """Get video_path as Path object."""
        return Path(self.video_path)

    @property
    def face_hint_pos(self) -> Tuple[float, float]:
        """Get face hint position from metadata."""
        return self.clip_metadata["x"],  self.clip_metadata["y"]


def load_metadata(metadata_path: Path) -> Dict[str, List[Dict]]:
    """Load metadata.jsonl and return mapping {video_id: [frames]}."""
    index = {}
    try:
        with open(metadata_path, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    video_id = entry.get("video_id") or entry["metadata"]["video_id"]
                    index[video_id] = entry["frames"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logging.error(f"❌ Error loading metadata from {metadata_path}: {e}")
        raise
    return index


class DataLoader:
    """
    Handles loading and iteration over AVSpeech dataset clips.

    This minimal implementation replicates the exact logic from the original
    main file to ensure identical behavior.
    """

    def __init__(self, clips_dir: Path, metadata_path: Path, max_clips: int = 0):
        """
        Initialize the DataLoader with dataset paths.

        Args:
            clips_dir: Path to AVSpeech clips directory
            metadata_path: Path to metadata.jsonl file
            max_clips: Maximum number of clips to process (0 = all)
        """
        self.clips_dir = clips_dir
        self.metadata_path = metadata_path
        self.max_clips = max_clips
        self._starting_index = 0


        # State for iteration
        self._paired_clips: List[ClipData] = []

        # Statistics for summary
        self._stats = {
                'total_videos_in_metadata': 0,
                'total_video_files_found' : 0,
                'valid_pairs_created'     : 0,
                'videos_without_metadata' : 0,
                'metadata_without_videos' : 0
        }

        # Load data using existing logic
        self._load_and_pair_data()



        # Print processing segment info
        self._print_processing_segment_info()

    def _load_and_pair_data(self) -> None:

        self._build_clip_data_fresh()

        # Save to cache
        # if self.use_cache:
        #     self._save_to_cache()




    def _build_clip_data_fresh(self) -> None:
        """Build clip data from scratch (original logic)."""
        # Load metadata using existing function
        metadata_index = load_metadata(self.metadata_path)
        self._stats['total_videos_in_metadata'] = len(metadata_index)
        print(f"✓ Loaded metadata for {len(metadata_index)} videos")

        # Find video files (replicates find_video_files() logic)
        video_files = self._find_video_files()
        self._stats['total_video_files_found'] = len(video_files)
        print(f"✓ Found {len(video_files)} video files")

        # Pair files with metadata (replicates pair_files_with_metadata() logic)
        self._paired_clips = self._pair_files_with_metadata(video_files, metadata_index)
        self._stats['valid_pairs_created'] = len(self._paired_clips)
        print(f"🔗 Pairing video files with metadata... Created {len(self._paired_clips)} file-metadata pairs")

        # Print summary stats
        self._print_loading_summary()

    def _find_video_files(self) -> List[Tuple[Path, str]]:
        """
        Find video files (exact replica of original find_video_files function).

        Returns:
            List of (video_path, video_id) tuples
        """
        video_files = []
        for mp4_path in self.clips_dir.rglob("*.mp4"):
            video_id = mp4_path.parent.name
            if "_" in mp4_path.stem:
                video_files.append((mp4_path, video_id))
        return sorted(video_files)

    def _pair_files_with_metadata(
            self,
            video_files: List[Tuple[Path, str]],
            metadata_index: dict
    ) -> List[ClipData]:
        """
        Pair video files with metadata (exact replica of original function).

        Args:
            video_files: List of (video_path, video_id) tuples
            metadata_index: Dictionary mapping video_id to metadata list

        Returns:
            List of ClipData objects
        """
        paired_data = []

        # Group video files by video_id
        video_id_to_paths = {}
        for path, vid in video_files:
            video_id_to_paths.setdefault(vid, []).append(path)

        # Sort paths within each video_id group
        for vid in video_id_to_paths:
            video_id_to_paths[vid].sort()

        # Track videos without metadata
        videos_without_metadata = 0

        # Pair each video with its metadata
        for video_id, paths in video_id_to_paths.items():
            if video_id not in metadata_index:
                print(f"⚠️  Warning: No metadata found for video {video_id}, skipping")
                videos_without_metadata += 1
                continue

            clips_metadata = metadata_index[video_id]

            for clip_idx, clip_meta in enumerate(clips_metadata):
                if clip_idx < len(paths):
                    video_path = paths[clip_idx]
                else:
                    print(f"⚠️  Skipping: No matching video file for clip {clip_idx} in video {video_id}")
                    continue

                unique_clip_id = f"{video_id}_{clip_idx}"
                paired_data.append(ClipData(
                        unique_clip_id=unique_clip_id,
                        video_path=str(video_path),  # Convert Path to string for JSON
                        clip_metadata=clip_meta
                ))

        self._stats['videos_without_metadata'] = videos_without_metadata

        # Count metadata entries without corresponding video files
        metadata_without_videos = len(metadata_index) - len(video_id_to_paths)
        self._stats['metadata_without_videos'] = max(0, metadata_without_videos)

        return paired_data

    def _print_loading_summary(self) -> None:
        """Print a single line summary of loading statistics."""
        stats = self._stats
        print(f"📊 Loading Summary: {stats['valid_pairs_created']} valid pairs created, "
              f"{stats['videos_without_metadata']} videos skipped (no metadata), "
              f"{stats['metadata_without_videos']} metadata entries unused")

    def get_next_clip(self) -> Optional[ClipData]:
        """
        Get the next clip for processing.

        Returns:
            ClipData object for the next clip, or None if no more clips
        """
        if self._current_index >= len(self._paired_clips):
            return None

        # Check if we've reached the max_clips limit for this run
        if 0 < self.max_clips <= (self._current_index - self._starting_index):
            return None

        clip_data = self._paired_clips[self._current_index]
        self._current_index += 1
        return clip_data

    def get_current_index(self) -> int:
        """Get the current processing index."""
        return self._current_index

    def get_total_clips(self) -> int:
        """Get the total number of clips available for processing in this run."""
        remaining_clips = len(self._paired_clips) - self._starting_index
        if self.max_clips > 0:
            return min(self.max_clips, remaining_clips)
        return remaining_clips

    def get_total_available_clips(self) -> int:
        """Get the total number of clips available (ignoring max_clips limit)."""
        return len(self._paired_clips)

    def is_processing_complete(self) -> bool:
        """
        Check if ALL available clips have been processed.

        Returns:
            True if we've processed all clips in the dataset (not just max_clips limit)
        """
        return self._current_index >= len(self._paired_clips)

    def reset(self) -> None:
        """Reset the iterator to the current starting position."""
        self._current_index = self._starting_index

    def __iter__(self) -> Iterator[ClipData]:
        """Make DataLoader iterable."""
        self.reset()
        return self

    def __next__(self) -> ClipData:
        """Iterator protocol implementation."""
        clip_data = self.get_next_clip()
        if clip_data is None:
            raise StopIteration
        return clip_data

    def __len__(self) -> int:
        """Return the effective length (considering max_clips limit)."""
        return self.get_total_clips()

    def _print_processing_segment_info(self) -> None:
        """Print detailed info about what segment will be processed in this run."""
        start_index = self._starting_index
        clips_to_process = self.get_total_clips()
        end_index = start_index + clips_to_process - 1
        total_available = len(self._paired_clips)

        if clips_to_process == 0:
            print("⚠️  No clips to process in this run")
            return

        # Build the message components
        range_str = f"[{start_index}-{end_index}]" if clips_to_process > 1 else f"[{start_index}]"

        # Fresh start
        if self.max_clips > 0:
            percent_of_total = (clips_to_process * 100) // total_available
            print(f"🚀 Processing {clips_to_process} new clips {range_str}, "
                  f"(--max-clips = {self.max_clips}, {percent_of_total}% of dataset)")
        else:
            print(f"🚀 Processing all {clips_to_process} clips {range_str}, "
                  f"(processing entire dataset)")

