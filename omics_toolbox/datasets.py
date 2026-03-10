"""Dataset download utilities for omics_toolbox tutorials."""

import os
import shutil
import zipfile

import requests

_MENDELEY_DATASET_ID = "v6n743h5ng"
_MENDELEY_VERSION = "1"
# File ID for scRNAseq.zip from the Mendeley public files API.
_MENDELEY_SCRNA_FILE_ID = "b1865840-e8df-4381-8866-b04d57309e1d"

# Primary: Mendeley public files API (direct zip download, no auth required).
# The legacy S3 cache URL now returns 403, and the datasets-v2 API endpoint
# returns JSON (a redirect descriptor) rather than the zip itself.
_DOWNLOAD_URLS = [
    f"https://data.mendeley.com/public-files/datasets/{_MENDELEY_DATASET_ID}/files/{_MENDELEY_SCRNA_FILE_ID}/file_downloaded",
]

_HEADERS = {
    "User-Agent": "omics_toolbox/1.0 (https://github.com/yourrepo/omics_toolbox)",
    "Accept": "application/octet-stream,*/*",
}

_MANUAL_DOWNLOAD_URL = (
    f"https://data.mendeley.com/datasets/{_MENDELEY_DATASET_ID}/{_MENDELEY_VERSION}"
)


def download_embryoid_body(download_path="~/scRNAseq"):
    """Download and extract the Embryoid Body scRNA-seq dataset from Mendeley Datasets.

    The dataset is a 31,000 cell, 27-day time course of embryoid body differentiation
    (Moon et al. 2019). The raw zip file is ~746 MB.

    Parameters
    ----------
    download_path : str
        Directory where the data will be saved. Defaults to ``~/scRNAseq``.

    Returns
    -------
    str
        Absolute path to the directory containing the extracted data.

    Directory structure after extraction::

        <download_path>/
        ├── T0_1A/  (barcodes.tsv, genes.tsv, matrix.mtx)
        ├── T2_3B/
        ├── T4_5C/
        ├── T6_7D/
        └── T8_9E/

    Notes
    -----
    If automatic download fails (Mendeley occasionally blocks direct access),
    download ``scRNAseq.zip`` manually from:

        https://data.mendeley.com/datasets/v6n743h5ng/1

    then call this function with ``skip_download=True`` and place the zip in
    ``download_path``, or extract the folders there directly.
    """
    download_path = os.path.expanduser(download_path)
    dest_dir = os.path.join(download_path, "scRNAseq")

    # --- Skip if already downloaded ----------------------------------------
    expected_subdirs = {"T0_1A", "T2_3B", "T4_5C", "T6_7D", "T8_9E"}
    if os.path.isdir(dest_dir) and expected_subdirs.issubset(os.listdir(dest_dir)):
        print(f"Dataset already exists at: {dest_dir} — skipping download.")
        return dest_dir

    os.makedirs(download_path, exist_ok=True)

    zip_file = os.path.join(download_path, "scRNAseq.zip")

    # --- Download -----------------------------------------------------------
    downloaded = False
    for url in _DOWNLOAD_URLS:
        print(f"Trying {url} ...")
        try:
            with requests.get(url, headers=_HEADERS, stream=True, timeout=60) as r:
                r.raise_for_status()
                content_type = r.headers.get("content-type", "")
                if "json" in content_type:
                    raise RuntimeError(
                        f"URL returned JSON instead of a zip file (content-type: {content_type}). "
                        "The API endpoint may have changed."
                    )
                total = int(r.headers.get("content-length", 0))
                _stream_to_file(r, zip_file, total)
            downloaded = True
            print(f"Downloaded to {zip_file}")
            break
        except requests.HTTPError as exc:
            print(f"  Failed ({exc.response.status_code}), trying next URL...")
        except Exception as exc:
            print(f"  Failed ({exc}), trying next URL...")

    if not downloaded:
        raise RuntimeError(
            "All download URLs failed.\n"
            "Please download the dataset manually from:\n"
            f"  {_MANUAL_DOWNLOAD_URL}\n"
            "Save 'scRNAseq.zip' to:\n"
            f"  {download_path}\n"
            "then re-run this function — it will skip the download and extract directly."
        )

    # --- Extract -----------------------------------------------------------
    temp_extract = os.path.join(download_path, "_temp_extract")
    os.makedirs(temp_extract, exist_ok=True)

    print("Extracting scRNAseq.zip...")
    with zipfile.ZipFile(zip_file, "r") as zf:
        zf.extractall(temp_extract)

    scrna_folder = os.path.join(temp_extract, "scRNAseq")
    source_dir = scrna_folder if os.path.isdir(scrna_folder) else temp_extract

    print("Moving contents to destination directory...")
    os.makedirs(dest_dir, exist_ok=True)
    for item in os.listdir(source_dir):
        src = os.path.join(source_dir, item)
        dst = os.path.join(dest_dir, item)
        shutil.move(src, dst)

    shutil.rmtree(temp_extract)
    os.remove(zip_file)
    print("Cleaned up temporary files.")

    print(f"\nDone! scRNAseq data is in: {dest_dir}")
    return dest_dir


_CANCER_PLASTICITY_FOLDER_ID = "1maxTkGZxo0cZ7zEW5i3pz2cbFSxdBP-v"
_CANCER_PLASTICITY_SAMPLES = [
    "HCC38_2D_culture",
    "TSA-48HR_3",
    "TSA-D12",
    "TSA-D18",
    "TSA-D30",
]


def download_cancer_plasticity(download_path="~/cancer_plasticity"):
    """Download the cancer plasticity scRNA-seq dataset from Google Drive.

    The dataset contains HCC38 breast cancer cells undergoing epigenetic
    reprogramming (TSA treatment) across five conditions/timepoints.

    Parameters
    ----------
    download_path : str
        Directory where the data will be saved. Defaults to ``~/cancer_plasticity``.

    Returns
    -------
    str
        Absolute path to the directory containing the downloaded data.

    Directory structure after download::

        <download_path>/
        ├── HCC38_2D_culture/  (barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz)
        ├── TSA-48HR_3/
        ├── TSA-D12/
        ├── TSA-D18/
        └── TSA-D30/

    Notes
    -----
    Requires ``gdown``: ``pip install gdown``.
    """
    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "gdown is required to download the cancer plasticity dataset.\n"
            "Install it with: pip install gdown"
        ) from exc

    download_path = os.path.expanduser(download_path)
    os.makedirs(download_path, exist_ok=True)

    # Check if already downloaded
    if all(
        os.path.isdir(os.path.join(download_path, s))
        for s in _CANCER_PLASTICITY_SAMPLES
    ):
        print(f"Data already exists at: {download_path}")
        return download_path

    folder_url = f"https://drive.google.com/drive/folders/{_CANCER_PLASTICITY_FOLDER_ID}"
    print(f"Downloading cancer plasticity data from Google Drive...")
    gdown.download_folder(folder_url, output=download_path, quiet=False, use_cookies=False)

    print(f"\nDone! Cancer plasticity data is in: {download_path}")
    return download_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stream_to_file(response, path, total_bytes):
    """Write a streaming requests response to *path* with a tqdm progress bar."""
    try:
        from tqdm.auto import tqdm
        progress = tqdm(total=total_bytes or None, unit="B", unit_scale=True,
                        desc="Downloading")
    except ImportError:
        progress = None

    chunk_size = 1 << 20  # 1 MB
    with open(path, "wb") as fh:
        for chunk in response.iter_content(chunk_size=chunk_size):
            fh.write(chunk)
            if progress is not None:
                progress.update(len(chunk))

    if progress is not None:
        progress.close()
