"""Utilities for downloading and caching files."""

import time
from pathlib import Path
from typing import Optional, Union

import requests
from tqdm import tqdm


def download_file(
    url: str,
    output_path: Union[str, Path],
    timeout: int = 30,
    chunk_size: int = 8192,
    *,
    overwrite: bool = False,
    desc: Optional[str] = None,
) -> Path:
    """Download a file from a URL with progress bar.

    Args:
        url: URL to download
        output_path: Path where the file will be saved
        timeout: Connection timeout in seconds
        chunk_size: Size of chunks to download
        overwrite: Whether to overwrite existing files
        desc: Description for the progress bar

    Returns:
        Path: Path to the downloaded file
    """
    output_path = Path(output_path)

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # If file exists and we're not overwriting, return the path
    if output_path.exists() and not overwrite:
        return output_path

    # Set description for progress bar
    if desc is None:
        desc = f"Downloading {Path(url).name}"

    # Download the file
    start_time = time.time()
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        # Get file size if available
        total_size = int(response.headers.get("content-length", 0))

        # Download with progress bar
        with open(output_path, "wb") as f:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=desc,
                disable=total_size == 0,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        download_time = time.time() - start_time
        print(f"Downloaded {output_path.name} in {download_time:.2f}s")

        return output_path

    except (OSError, requests.RequestException) as err:
        # Remove partial download if it exists
        if output_path.exists():
            output_path.unlink()
        msg = f"Failed to download {url}: {err!s}"
        raise RuntimeError(msg) from err


def get_cache_dir(project_name: str = "anise") -> Path:
    """Get the cache directory for the project.

    Args:
        project_name: Name of the project

    Returns:
        Path: Path to the cache directory
    """
    # Use repository root for cache directory
    repo_root = Path(__file__).parent.parent.parent.parent
    cache_dir = repo_root / ".cache" / project_name

    # Create directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir


def cached_download(
    url: str,
    cache_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None,
    timeout: int = 30,
    *,
    overwrite: bool = False,
    project_name: str = "anise",
) -> Path:
    """Download a file and cache it.

    Args:
        url: URL to download
        cache_dir: Directory to cache the file in (default: project cache dir)
        filename: Name to save the file as (default: derived from URL)
        timeout: Connection timeout in seconds
        overwrite: Whether to overwrite existing files
        project_name: Name of the project for default cache directory

    Returns:
        Path: Path to the cached file
    """
    # Get cache directory
    if cache_dir is None:
        cache_dir = get_cache_dir(project_name)
    else:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Get filename from URL if not provided
    if filename is None:
        filename = Path(url).name

    # Full path to cached file
    cached_file = cache_dir / filename

    # Download if needed
    return download_file(
        url=url,
        output_path=cached_file,
        timeout=timeout,
        overwrite=overwrite,
    )
