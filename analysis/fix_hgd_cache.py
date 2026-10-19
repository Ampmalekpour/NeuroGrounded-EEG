# fix_hgd_cache.py
#
# Repairs the local MOABB cache of the High-Gamma Dataset (Schirrmeister2017).
#
# Background: on some machines the cached "test" recording of subjects 6-14 is a byte-identical copy of
# the "train" recording (and for subject 11 the cached "train" file is the small test file), although the
# server holds distinct files (train ~0.9 GB, test ~0.16 GB). Testing on such a copy means testing on
# training data. This script
#   1. asks the server for the size of every train/test EDF (HEAD request, no download),
#   2. compares it with the cached file (the new MOABB layout AND the legacy <dataset>/<train|test>/N.edf
#      layout, which MOABB reads first),
#   3. deletes the cached files whose size differs and downloads the correct ones to the canonical
#      location, verifying the size.
# Files that already have the right size are left alone. Nothing else is modified.
#
#   python analysis/fix_hgd_cache.py              # check all 14 subjects and fix what is wrong
#   python analysis/fix_hgd_cache.py --check      # only report
#   python analysis/fix_hgd_cache.py 6 7 11       # some subjects
#
# Afterwards run  python analysis/check_hgd_data.py  (it must report no identical recordings).
# Override the server with EEG_HGD_BASE_URL (testing only).

import os
import sys
import argparse
import urllib.request

BASE_URLS = ([os.environ["EEG_HGD_BASE_URL"]] if os.environ.get("EEG_HGD_BASE_URL") else [
    "https://web.gin.g-node.org/robintibor/high-gamma-dataset/raw/master/data",      # the host MOABB uses
    "https://gin.g-node.org/robintibor/high-gamma-dataset/raw/master/data"])         # same repository
SIGN = "SCHIRRMEISTER2017"


def cache_root():
    import mne
    base = (mne.get_config("MNE_DATASETS_%s_PATH" % SIGN) or mne.get_config("MNE_DATA")
            or os.path.join(os.path.expanduser("~"), "mne_data"))
    return os.path.join(os.path.abspath(base), "MNE-%s-data" % SIGN.lower())


def local_paths(root, kind, subject):
    """(canonical path used by current MOABB, legacy path read first by MOABB)."""
    parts = ["robintibor", "high-gamma-dataset", "raw", "master", "data", kind, "%d.edf" % subject]
    return os.path.join(root, *parts), os.path.join(root, kind, "%d.edf" % subject)


def server_size(kind, subject):
    """(url, size in bytes) from the first host that answers."""
    last = None
    for base in BASE_URLS:
        url = "%s/%s/%d.edf" % (base, kind, subject)
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=60) as r:
                return url, int(r.headers["Content-Length"])
        except Exception as exc:                  # try the next host
            last = exc
    raise SystemExit("cannot reach the HGD server (%s: %s)" % (type(last).__name__, last))


def download(url, dest, expected):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    done = 0
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            print("\r    %.0f / %.0f MB" % (done / 1e6, expected / 1e6), end="", flush=True)
    print()
    if done != expected:
        os.remove(tmp)
        raise RuntimeError("download incomplete: %d of %d bytes" % (done, expected))
    os.replace(tmp, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subjects", nargs="*", type=int)
    ap.add_argument("--check", action="store_true", help="only report, change nothing")
    a = ap.parse_args()
    subjects = a.subjects or list(range(1, 15))
    root = cache_root()
    print("cache:", root)
    bad = 0
    for s in subjects:
        for kind in ("train", "test"):
            url, want = server_size(kind, s)
            canonical, legacy = local_paths(root, kind, s)
            have = {p: os.path.getsize(p) for p in (canonical, legacy) if os.path.isfile(p)}
            wrong = [p for p, n in have.items() if n != want]
            status = "ok" if have and not wrong else ("missing" if not have else "WRONG SIZE")
            print("S%02d %-5s server %12s | local %s -> %s" % (
                s, kind, "{:,}".format(want),
                ", ".join("{:,}".format(n) for n in have.values()) or "none", status))
            if status == "ok":
                continue
            bad += 1
            if a.check:
                continue
            for p in wrong:                       # also the legacy copy, which MOABB would read first
                os.remove(p)
                print("    deleted", p)
            download(url, canonical, want)
            print("    downloaded", canonical)
    print("\n%d file(s) %s." % (bad, "need fixing" if a.check else "fixed or already ok after this run"))
    if not a.check:
        print("Now run: python analysis/check_hgd_data.py")


if __name__ == "__main__":
    main()
