from __future__ import annotations

import os
import re
import shutil
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

USER_AGENT = "Mozilla/5.0 (compatible; manuscript-repro/1.0)"

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

# Public mirror of the canonical 14-row boarding-school outbreak table.
BOARDING_SCHOOL_URLS = (
    "https://raw.githubusercontent.com/CDCgov/Rt-without-renewal/refs/heads/main/"
    "EpiAware/docs/src/showcase/replications/chatzilena-2019/"
    "influenza_england_1978_school.csv2",
    "https://raw.githubusercontent.com/reconverse/outbreaks/main/data-raw/"
    "influenza_england_1978_school.csv",
)

LANL_LANDING_PAGES = (
    "https://www.lanl.gov/projects/national-security-education-center/engineering/"
    "software/shm-data-sets-and-software.php",
    "https://www.lanl.gov/projects/national-security-education-center/engineering/"
    "engineering-institute/software-downloads.php",
    "https://www.lanl.gov/projects/national-security-education-center/engineering/"
    "engineering-institute/software-and-data.php",
)
LANL_KEYWORDS = ("three", "3-story", "3story", "story", "storey", "nonlinear", "frame")
LANL_EXTENSIONS = (".zip", ".tar.gz", ".tgz", ".tar", ".mat", ".csv", ".txt", ".dat", ".npy", ".npz")


def default_cache_dir(project_root: str | Path) -> Path:
    return Path(project_root) / ".cache" / "datasets"


def _request(url: str, timeout: int = 60) -> urllib.response.addinfourl:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)


def download(url: str, destination: str | Path, *, refresh: bool = False) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0 and not refresh:
        return destination
    tmp = destination.with_suffix(destination.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    with _request(url) as response, tmp.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded an empty file from {url}")
    tmp.replace(destination)
    return destination


def ensure_etex_i(cache_dir: str | Path, *, refresh: bool = False) -> Path:
    """Download the official JRC ETEX-I tracer/station files on demand."""
    out = Path(cache_dir) / "etex_i"
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for name in ETEX_FILES:
        try:
            download(ETEX_RELEASE1_BASE + name, out / name, refresh=refresh)
        except Exception as exc:
            # Only stationlist and pmch.dat are essential. Metadata files are helpful but optional.
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


def _extract_archive(path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    lower = path.name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(destination)
    elif lower.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(path) as tf:
            tf.extractall(destination)
    else:
        return path
    return destination


def _scrape_lanl_links(page_url: str) -> list[str]:
    with _request(page_url, timeout=30) as response:
        html = response.read().decode("utf-8", errors="ignore")
    hrefs = re.findall(r'''href\s*=\s*["']([^"']+)["']''', html, flags=re.I)
    links = [urllib.parse.urljoin(page_url, href) for href in hrefs]

    def score(url: str) -> tuple[int, int]:
        low = url.lower()
        keyword_score = sum(k in low for k in LANL_KEYWORDS)
        extension_score = sum(low.endswith(ext) for ext in LANL_EXTENSIONS)
        return keyword_score, extension_score

    candidates = [u for u in links if score(u)[1] and score(u)[0] > 0]
    return sorted(dict.fromkeys(candidates), key=score, reverse=True)


def ensure_lanl_three_story(
    cache_dir: str | Path,
    *,
    refresh: bool = False,
    dataset_url: str | None = None,
) -> Path:
    """Download/discover the LANL three-story-frame archive and return a local folder/file.

    The LANL hosting URL has changed historically, so the downloader supports three paths:
      1) --dataset-url supplied by the user,
      2) LANL_THREE_STORY_URL environment variable,
      3) automatic discovery from official LANL Engineering Institute landing pages.

    No synthetic data are silently substituted.
    """
    out = Path(cache_dir) / "lanl_three_story"
    out.mkdir(parents=True, exist_ok=True)
    explicit = dataset_url or os.environ.get("LANL_THREE_STORY_URL")
    candidates: list[str] = [explicit] if explicit else []
    discovery_errors: list[str] = []
    if not candidates:
        for page in LANL_LANDING_PAGES:
            try:
                candidates.extend(_scrape_lanl_links(page))
            except Exception as exc:
                discovery_errors.append(f"{page}: {exc}")
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        details = "\n".join(discovery_errors)
        raise RuntimeError(
            "Could not discover the LANL three-story-frame download automatically. "
            "Pass --dataset-url or set LANL_THREE_STORY_URL to the official/mirror archive URL."
            + ("\nDiscovery errors:\n" + details if details else "")
        )

    errors: list[str] = []
    for idx, url in enumerate(candidates):
        try:
            parsed = urllib.parse.urlparse(url)
            name = Path(parsed.path).name or f"lanl_download_{idx}.zip"
            local = download(url, out / name, refresh=refresh)
            extracted = _extract_archive(local, out / "extracted")
            return extracted
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("All LANL download candidates failed:\n" + "\n".join(errors))
