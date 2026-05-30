#!/usr/bin/python3

import os
import sys
import shutil
import urllib.request
import zipfile

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/48.0.2564.116 Safari/537.36"
)

BISTRO_PACKAGES = [
    ("bistro/Exterior",       "https://casual-effects.com/g3d/data10/research/model/bistro/Exterior.zip"),
    ("bistro/PropTextures",   "https://casual-effects.com/g3d/data10/research/model/bistro/PropTextures"),
    ("bistro/OtherTextures",  "https://casual-effects.com/g3d/data10/research/model/bistro/OtherTextures"),
    ("bistro/BuildingTextures", "https://casual-effects.com/g3d/data10/research/model/bistro/BuildingTextures"),
]

def download(url, dest_path):
    print("Downloading {} ...".format(url))
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as f:
        shutil.copyfileobj(response, f)

def extract(archive_path, target_dir):
    print("Extracting to {} ...".format(target_dir))
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        common = os.path.commonprefix(names)
        # strip trailing slash so we can use it as a strip prefix
        strip = common if common.endswith("/") else ""
        for member in names:
            rel = member[len(strip):] if strip else member
            if not rel:
                continue
            dest = os.path.join(target_dir, rel.replace("/", os.sep))
            if member.endswith("/"):
                os.makedirs(dest, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with zf.open(member) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    content_src = os.path.join(script_dir, "content", "src")
    archives_dir = os.path.join(script_dir, "content", "archives")

    os.makedirs(content_src, exist_ok=True)
    os.makedirs(archives_dir, exist_ok=True)

    for name, url in BISTRO_PACKAGES:
        target_dir = os.path.join(content_src, name.replace("/", os.sep))
        if os.path.isdir(target_dir):
            print("Already present, skipping: {}".format(name))
            continue

        filename = os.path.basename(url) or name.replace("/", "_")
        if not os.path.splitext(filename)[1]:
            filename += ".zip"
        archive_path = os.path.join(archives_dir, filename)

        if not os.path.exists(archive_path):
            download(url, archive_path)
        else:
            print("Archive cached, skipping download: {}".format(filename))

        os.makedirs(target_dir, exist_ok=True)
        extract(archive_path, target_dir)

    print("Done.")

if __name__ == "__main__":
    main()
