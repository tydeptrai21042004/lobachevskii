from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

USER_AGENT = "Mozilla/5.0 (compatible; manuscript-repro/2.0)"

ETEX_RELEASE1_BASE = (
    "https://remon.jrc.ec.europa.eu/past_activities/etex/site/database/ETEX_release1/"
)
ETEX_FILES = (
    "stationlist.950130",
    "pmch.dat",
    "pmch.cod",
    "pmch.readme",
    "release1.txt",
)

BOARDING_SCHOOL_URLS = (
    "https://raw.githubusercontent.com/CDCgov/Rt-without-renewal/refs/heads/main/"
    "EpiAware/docs/src/showcase/replications/chatzilena-2019/"
    "influenza_england_1978_school.csv2",
    "https://raw.githubusercontent.com/reconverse/outbreaks/main/data-raw/"
    "influenza_england_1978_school.csv",
)

RWTH_RECORD_ID = 10134011
RWTH_ARCHIVE_NAME = "Data_v1.0.0.zip"
RWTH_ARCHIVE_MD5 = "7d2afccaad303ae647416ded6c65dc3e"
RWTH_WHITE_NOISE_FILE = "LP02_Whitenoise_001.csv"
RWTH_ARCHIVE_URLS = (
    f"https://zenodo.org/records/{RWTH_RECORD_ID}/files/{RWTH_ARCHIVE_NAME}?download=1",
    f"https://zenodo.org/api/records/{RWTH_RECORD_ID}/files/{RWTH_ARCHIVE_NAME}/content",
)


def default_cache_dir(project_root: str | Path) -> Path:
    return Path(project_root) / ".cache" / "datasets"


def _request(url: str, timeout: int = 90):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)


def download(url: str, destination: str | Path, *, refresh: bool = False) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0 and not refresh:
        return destination
    tmp = destination.with_suffix(destination.suffix + ".part")
    tmp.unlink(missing_ok=True)
    try:
        with _request(url) as response, tmp.open("wb") as fh:
            shutil.copyfileobj(response, fh)
        if tmp.stat().st_size == 0:
            raise RuntimeError(f"Downloaded an empty file from {url}")
        tmp.replace(destination)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return destination


def _md5(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # nosec B324 - used only for published dataset integrity checking
    with path.open("rb") as fh:
        while True:
            block = fh.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def ensure_rwth_white_noise(
    cache_dir: str | Path,
    *,
    refresh: bool = False,
    archive_url: str | None = None,
) -> Path:
    """Return the manuscript RWTH white-noise CSV, downloading the Zenodo archive if needed.

    Only the required ``LP02_Whitenoise_001.csv`` member is extracted. No synthetic
    fallback is ever used by this function.
    """
    root = Path(cache_dir) / "rwth_steel_frame"
    extracted = root / RWTH_WHITE_NOISE_FILE
    if extracted.exists() and extracted.stat().st_size > 0 and not refresh:
        return extracted

    archive = root / RWTH_ARCHIVE_NAME
    root.mkdir(parents=True, exist_ok=True)
    urls = (archive_url,) if archive_url else RWTH_ARCHIVE_URLS
    errors: list[str] = []
    for url in urls:
        try:
            download(url, archive, refresh=refresh)
            actual_md5 = _md5(archive)
            if actual_md5.lower() != RWTH_ARCHIVE_MD5.lower():
                archive.unlink(missing_ok=True)
                raise RuntimeError(
                    f"RWTH archive checksum mismatch: expected {RWTH_ARCHIVE_MD5}, got {actual_md5}."
                )
            with zipfile.ZipFile(archive) as zf:
                matches = [name for name in zf.namelist() if Path(name).name.lower() == RWTH_WHITE_NOISE_FILE.lower()]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Expected exactly one {RWTH_WHITE_NOISE_FILE} in {RWTH_ARCHIVE_NAME}; found {len(matches)}."
                    )
                member = matches[0]
                tmp = extracted.with_suffix(extracted.suffix + ".part")
                tmp.unlink(missing_ok=True)
                with zf.open(member) as src, tmp.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                if tmp.stat().st_size == 0:
                    raise RuntimeError(f"Extracted {RWTH_WHITE_NOISE_FILE} is empty.")
                tmp.replace(extracted)
            return extracted
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            if archive_url:
                break
    raise RuntimeError(
        "Could not obtain the RWTH white-noise record. You may pass --dataset-url to "
        "scripts/run_mechanical_rwth.py or --rwth-file with an existing CSV.\n" + "\n".join(errors)
    )


def ensure_etex_i(cache_dir: str | Path, *, refresh: bool = False) -> Path:
    """Download the official JRC ETEX-I tracer/station files on demand."""
    out = Path(cache_dir) / "etex_i"
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for name in ETEX_FILES:
        try:
            download(ETEX_RELEASE1_BASE + name, out / name, refresh=refresh)
        except Exception as exc:
            if name in {"stationlist.950130", "pmch.dat"}:
                errors.append(f"{name}: {exc}")
    if errors:
        raise RuntimeError("Could not download required ETEX-I files:\n" + "\n".join(errors))
    return out


def ensure_boarding_school(cache_dir: str | Path, *, refresh: bool = False) -> Path:
    """Download the canonical 1978 boarding-school influenza CSV on demand."""
    out = Path(cache_dir) / "influenza_1978" / "influenza_england_1978_school.csv"
    if out.exists() and out.stat().st_size > 0 and not refresh:
        return out
    errors: list[str] = []
    for url in BOARDING_SCHOOL_URLS:
        try:
            return download(url, out, refresh=True)
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Could not download the boarding-school data:\n" + "\n".join(errors))
