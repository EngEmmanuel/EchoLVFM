"""
Video and image loading utilities for medical imaging datasets.
"""

import cv2
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import List, Optional, Union
import warnings

def _is_video_file(p: str):
    ext = Path(p).suffix.lower()
    return ext in ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv')

# Use a reasonable cache size for video loading
@lru_cache(maxsize=256)
def load_video_frames(video_path: Path, channels: int = 1) -> np.ndarray:
    """
    Load video frames from a video file.

    Args:
        video_path: Path to the video file
        channels: Number of channels (1 for grayscale, 3 for RGB)

    Returns:
        numpy array of shape (T, H, W) for grayscale or (T, H, W, C) for RGB
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if channels == 1:
                # Convert to grayscale
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)
            else:
                # Keep RGB (convert BGR to RGB)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames found in video: {video_path}")

    return np.array(frames)


# Use a reasonable cache size for video loading
@lru_cache(maxsize=256)
def load_image_sequence(
    image_dir: Path,
    extensions: Optional[List[str]] = None,
    grayscale: bool = True
) -> np.ndarray:
    """
    Load a sequence of images from a directory.

    Args:
        image_dir: Path to directory containing image files
        extensions: List of file extensions to search for
        grayscale: If True, load images as grayscale

    Returns:
        numpy array of shape (T, H, W) for grayscale or (T, H, W, C) for color
    """
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']

    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(list(image_dir.glob(f'*{ext}')))
        image_files.extend(list(image_dir.glob(f'*{ext.upper()}')))

    if not image_files:
        raise ValueError(f"No image files found in directory: {image_dir}")

    # Sort files to ensure consistent ordering
    image_files.sort(key=lambda x: x.name)

    images = []
    for image_file in image_files:
        if grayscale:
            image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image is None:
            warnings.warn(f"Could not load image file: {image_file}")
            continue
        images.append(image)

    if not images:
        raise ValueError(f"No valid image files found in directory: {image_dir}")

    return np.array(images)


def resample_sequence(frames: np.ndarray, target_length: int = 32) -> np.ndarray:
    """
    Resample a sequence to a target length.

    Args:
        frames: Input frames of shape (T, H, W) or (T, H, W, C)
        target_length: Target number of frames

    Returns:
        Resampled frames of shape (target_length, H, W) or (target_length, H, W, C)

    Notes:
        - If T < target_length, append zero-padded frames
        - If T >= target_length, evenly sample frames including endpoints
    """
    T = frames.shape[0]
    if T < target_length:
        # Pad with zeros
        pad_shape = (target_length - T,) + frames.shape[1:]
        pad_block = np.zeros(pad_shape, dtype=frames.dtype)
        return np.concatenate([frames, pad_block], axis=0)
    else:
        # Evenly sample frames including endpoints
        indices = np.linspace(0, T - 1, target_length)
        indices = np.round(indices).astype(int)
        return frames[indices]


# Convenience functions for specific use cases
def load_video(video_path: Union[str, Path], channels: int = 1) -> np.ndarray:
    """Load video with automatic path conversion."""
    return load_video_frames(Path(video_path), channels=channels)


def load_mask_sequence(mask_dir: Union[str, Path]) -> np.ndarray:
    """Load mask sequence (always grayscale)."""
    return load_image_sequence(Path(mask_dir), grayscale=True)


def load_image_sequence_color(image_dir: Union[str, Path], channels: int = 3) -> np.ndarray:
    """Load color image sequence."""
    return load_image_sequence(Path(image_dir), grayscale=(channels == 1))
