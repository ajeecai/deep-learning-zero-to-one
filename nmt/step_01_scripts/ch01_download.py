#!/usr/bin/env python3

"""Streaming downloader with MD5 check.
If md5 is given and the destination file exists and matches, the download is skipped.
"""
import hashlib
import os
import sys
from typing import Optional
from urllib.parse import urlparse, unquote
import requests
from pathlib import Path


def url_filename_if_gz(url: str):
    p = urlparse(url)
    # First try the path's basename
    name = unquote(Path(p.path).name)
    # If no valid name found in path, try common query params that contain file paths
    if not name or (
        not name.lower().endswith(".gz")
        and not name.lower().endswith(".tgz")
        and not name.lower().endswith(".zip")
        and not name.lower().endswith(".tar")
        and not name.lower().endswith(".tar.gz")
    ):
        # Check query parameters like ?f=.../en-zh.tmx.gz or ?file=.../name.gz
        qs = p.query
        if qs:
            # naive parse: look for f= or file= and extract last segment
            for key in ("f", "file", "filename", "path"):
                for part in qs.split("&"):
                    if part.startswith(key + "="):
                        val = part.split("=", 1)[1]
                        # val may be a path like UN/v20090831/tmx/en-zh.tmx.gz
                        candidate = unquote(Path(val).name)
                        if candidate and (
                            candidate.lower().endswith(".gz")
                            or candidate.lower().endswith(".tgz")
                        ):
                            return candidate
        return None  # 视为没有文件名或不是 gz 文件
    return name


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_md5(
    url: str,
    dest_path: str,
    expected_md5: Optional[str] = None,
    custom_filename: Optional[str] = None,
    chunk_size: int = 1 << 16,
) -> bool:
    """Download URL to dest_path. If expected_md5 is provided and matches existing file, skip.
    Returns True on success (file exists and matches expected_md5 if given), False on MD5 mismatch or error.
    """
    os.makedirs(dest_path or ".", exist_ok=True)

    if custom_filename:
        file_name = custom_filename
        dest_path_file = os.path.join(dest_path, file_name)
    else:
        # Fallback to old logic if no custom filename is provided
        file_name = url_filename_if_gz(url)
        if file_name is None:
            print(f"Could not determine filename from URL: {url}")
            return False
        # include domain in filename to indicate source and avoid name collisions
        parsed = urlparse(url)
        # sanitize netloc: replace '@' and ':' with '_' to make a safe filename segment
        domain_segment = parsed.netloc.replace("@", "_").replace(":", "_")
        safe_name = f"{domain_segment}-{file_name}"
        dest_path_file = os.path.join(dest_path, safe_name)

    if expected_md5 and os.path.exists(dest_path_file):
        cur = file_md5(dest_path_file)
        # print(f"cur_md5={cur}, expected_md5={expected_md5}")
        if cur == expected_md5:
            print(f"File exists and MD5 matches: {dest_path_file}")
            return True

    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = r.headers.get("Content-Length")
            total = int(total) if total and total.isdigit() else None
            downloaded = 0
            md5 = hashlib.md5()
            with open(dest_path_file, "wb") as wf:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    wf.write(chunk)
                    md5.update(chunk)
                    downloaded += len(chunk)
                    # Progress display: if total known, show percent, else show bytes
                    if total:
                        pct = downloaded * 100 / total
                        print(
                            f"\rDownloading {file_name}: {downloaded:,}/{total:,} bytes ({pct:5.1f}%)",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            f"\rDownloading {file_name}: {downloaded} bytes",
                            end="",
                            flush=True,
                        )
                # finish line after download completes
                print()
            got = md5.hexdigest()
            if expected_md5 and got != expected_md5:
                print(f"MD5 mismatch: expected {expected_md5}, got {got}")
                return False
            print(f"Downloaded {dest_path_file} (md5={got})")
            return True
    except Exception as e:
        print(f"Download of {url} failed: {e}")
        # hint: servers sometimes return HTML pages saying "can't open file downloads.txt"
        # when they block non-browser clients. Using browser-like headers or a real browser
        # (or curl with cookies) often bypasses this simple protection.
        return False


def main(argv=None):
    RAW_DIR = "../data/raw"
    urls_info = [
        (
            # very high quality, line-aligned, massive data, but some lines are very long
            "https://huggingface.co/datasets/wmt/uncorpus/resolve/main-zip/UNv1.0.en-zh.zip",
            "a1f03a09477411dd4a98f8dd589ef256",
            "huggingface.co-UNv1.0.en-zh.zip",
        ),
        # these two are not line aligned, so we won't use them for now
        # (
        #     # from https://opus.nlpl.eu/News-Commentary/en&zh/v16/News-Commentary
        #     "https://object.pouta.csc.fi/OPUS-News-Commentary/v16/mono/zh.txt.gz",
        #     "ddc571f08920b6a7d88df853e8258736",
        # ),
        # (
        #     # from https://opus.nlpl.eu/News-Commentary/en&zh/v16/News-Commentary
        #     "https://object.pouta.csc.fi/OPUS-News-Commentary/v16/mono/en.txt.gz",
        #     "b937799b80170737254f8db39b0a9538",
        # ),
        (
            # moses version has been line-aligned,large and high quality, some lines are long
            # from https://opus.nlpl.eu/News-Commentary/en&zh/v16/News-Commentary
            "https://object.pouta.csc.fi/OPUS-News-Commentary/v16/moses/en-zh.txt.zip",
            "7915a982d2999562155d7ee40288c312",
            "object.pouta.csc.fi-News-Commentary-v16.en-zh.zip",
        ),
        # 406 Client Error: Not Acceptable, manually download if needed
        # # Tranditional Chinese data, and need to split into en-zh file, so skip for now
        # (
        #     # from https://www.manythings.org/anki/
        #     "https://www.manythings.org/anki/cmn-eng.zip",
        #     None,
        #     "manythings.or-cmn-eng.zip",
        # ),
        (
            # large and high quality, some lines are long
            # from https://object.pouta.csc.fi/Tatoeba-Challenge-v2020-07-28/
            "https://object.pouta.csc.fi/Tatoeba-Challenge-v2020-07-28/eng-zho.tar",
            "24eeb995fc46bae3b3817241702d49c4",
            "object.pouta.csc.fi-Tatoeba-Challenge.eng-zho.tar",
        ),
        (
            # (Maybe) some mis-alignments in the middle, but large and high quantity,
            # there is some tranditional Chinese, need to convert. Lines are short.
            # from https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-zh_cn.txt.zip
            "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-zh_cn.txt.zip",
            "420bf9a9beb3eeb9bcb3ef9e7751345b",
            "object.pouta.csc.fi-OpenSubtitles-v2018.en-zh_cn.zip",
        ),
    ]
    for url, md5, filename in urls_info:
        print(f" ==> Processing URL: {url}")
        success = download_with_md5(
            url, RAW_DIR, expected_md5=md5, custom_filename=filename
        )
        if not success:
            sys.exit(2)


if __name__ == "__main__":
    main()
