"""
data_downloader.py — Autonomous PlantVillage Tomato Dataset Downloader
======================================================================
Downloads the tomato-disease subset of PlantVillage, organises it into
a clean directory tree, and creates a backup ZIP.

Download strategies (tried in order, no API keys needed):
    1. Hugging Face `datasets` library  → most reliable, no auth
    2. gdown  → Google Drive public mirror
    3. Manual → Clear instructions printed to console

Output
------
    PlantVillage/                ← training-ready folder tree
    Tomato_Disease_Dataset.zip   ← portable backup
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import zipfile
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 emoji crashes
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ── Configuration ───────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "PlantVillage"
ZIP_NAME = "Tomato_Disease_Dataset.zip"
ZIP_PATH = BASE_DIR / ZIP_NAME

# The 10 tomato classes we want (PlantVillage label strings)
# The HF dataset uses label integers; we map them via the ClassLabel feature.
TOMATO_PREFIXES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_(Two-spotted_spider_mite)",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Friendly short names used inside PlantVillage/ directory
SHORT_NAMES = {
    "Tomato___Bacterial_spot": "Bacterial_spot",
    "Tomato___Early_blight": "Early_blight",
    "Tomato___Late_blight": "Late_blight",
    "Tomato___Leaf_Mold": "Leaf_Mold",
    "Tomato___Septoria_leaf_spot": "Septoria_leaf_spot",
    "Tomato___Spider_mites_(Two-spotted_spider_mite)": "Spider_mites_(Two-spotted_spider_mite)",
    "Tomato___Target_Spot": "Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus": "Tomato_mosaic_virus",
    "Tomato___healthy": "healthy",
}

# Google Drive file-IDs (community mirrors) — fallback only
GDRIVE_IDS = [
    "1Hf8_6MR70kM32GXab-w5yZl9GtEBnXh9",
    "1nnFbsJ2o1u1yy0i7Xd6_PKDQFhziSfBE",
]


# ── Helpers ─────────────────────────────────────────────────
def _is_image(path: str) -> bool:
    return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _dataset_looks_ready() -> bool:
    """Return True if DATASET_DIR already has ≥ 8 class folders with images."""
    if not DATASET_DIR.is_dir():
        return False
    class_dirs = [d for d in DATASET_DIR.iterdir() if d.is_dir()]
    if len(class_dirs) < 8:
        return False
    total_images = sum(
        1 for d in class_dirs for f in d.iterdir() if _is_image(str(f))
    )
    return total_images > 500  # sanity threshold


# ── Strategy 1: Hugging Face `datasets` (MOST RELIABLE) ────
def _try_huggingface() -> bool:
    """
    Download PlantVillage from Hugging Face Hub via the `datasets` library.
    No API key or login required — the dataset is fully public.
    """
    try:
        from datasets import load_dataset  # noqa: WPS433
    except ImportError:
        print("  ⚠  `datasets` library not installed. Installing now …")
        os.system(f"{sys.executable} -m pip install datasets")
        try:
            from datasets import load_dataset
        except ImportError:
            print("  ✗  Could not install `datasets`. Skipping HF strategy.")
            return False

    print("  → Downloading from Hugging Face (mohanty/PlantVillage) …")
    print("    This may take a few minutes on the first run.\n")

    try:
        ds = load_dataset("mohanty/PlantVillage", split="train")
    except Exception as exc:
        print(f"  ✗  Hugging Face download failed: {exc}")
        return False

    # Get label names from the dataset features
    label_feature = ds.features["label"]
    label_names = label_feature.names  # list of all 38 class names
    print(f"  📋 Total classes in PlantVillage: {len(label_names)}")

    # Identify tomato class indices
    tomato_map: dict[int, str] = {}  # label_idx → short_name
    for idx, name in enumerate(label_names):
        for prefix, short in SHORT_NAMES.items():
            if name == prefix or name.startswith(prefix) or prefix in name:
                tomato_map[idx] = short
                break

    if not tomato_map:
        # Fallback: grab anything with "tomato" in the name
        for idx, name in enumerate(label_names):
            if "tomato" in name.lower():
                # Use the part after the last "___" as the short name
                short = name.split("___")[-1] if "___" in name else name
                tomato_map[idx] = short

    if not tomato_map:
        print("  ✗  No tomato classes found in the dataset!")
        return False

    print(f"  🍅 Found {len(tomato_map)} tomato classes:")
    for idx, short in sorted(tomato_map.items()):
        print(f"       [{idx:>2}] {short}")
    print()

    # Create output directories
    _ensure_dir(DATASET_DIR)
    for short in tomato_map.values():
        _ensure_dir(DATASET_DIR / short)

    # Save images into class folders
    counts: dict[str, int] = {s: 0 for s in tomato_map.values()}

    for i, sample in enumerate(ds):
        label_idx = sample["label"]
        if label_idx not in tomato_map:
            continue  # skip non-tomato classes

        short = tomato_map[label_idx]
        img = sample["image"]  # PIL Image
        dest = DATASET_DIR / short / f"{short}_{counts[short]:05d}.jpg"
        img.save(str(dest), "JPEG", quality=95)
        counts[short] += 1

        # Progress indicator every 500 images
        total_saved = sum(counts.values())
        if total_saved % 500 == 0:
            print(f"    … saved {total_saved} images so far …")

    total_saved = sum(counts.values())
    print(f"\n  📂 Saved {total_saved} tomato images into {DATASET_DIR}/")
    for short, n in sorted(counts.items()):
        print(f"    ✓ {short:.<50s} {n:>5}")

    return total_saved > 0


# ── Strategy 2: gdown (Google Drive) ───────────────────────
def _try_gdrive() -> bool:
    """Attempt to download from Google Drive via gdown."""
    try:
        import gdown  # noqa: WPS433
    except ImportError:
        print("  ⚠  gdown not installed — skipping Google Drive strategy.")
        return False

    tmp_zip = BASE_DIR / "_tmp_plantvillage.zip"
    for fid in GDRIVE_IDS:
        url = f"https://drive.google.com/uc?id={fid}"
        print(f"  → Trying Google Drive file {fid} …")
        try:
            gdown.download(url, str(tmp_zip), quiet=False)
            if tmp_zip.exists() and tmp_zip.stat().st_size > 1_000_000:
                _extract_zip_and_organise(tmp_zip)
                tmp_zip.unlink(missing_ok=True)
                return True
            tmp_zip.unlink(missing_ok=True)
        except Exception as exc:
            print(f"    ✗ Failed: {exc}")
            tmp_zip.unlink(missing_ok=True)
    return False


# ── ZIP extraction & organisation ──────────────────────────
def _extract_zip_and_organise(zip_path: Path) -> None:
    """
    Unzip into a temp dir, find tomato class folders (even if nested),
    and copy them into DATASET_DIR with clean short names.
    """
    tmp_dir = BASE_DIR / "_tmp_extract"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print("  📦 Extracting archive …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)

    _ensure_dir(DATASET_DIR)
    found = 0
    for root, dirs, _files in os.walk(tmp_dir):
        for d in dirs:
            full_path = Path(root) / d
            for keyword, short in SHORT_NAMES.items():
                if d == keyword or d.startswith(keyword) or keyword in d:
                    dest = DATASET_DIR / short
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(full_path, dest)
                    img_count = sum(1 for f in dest.iterdir() if _is_image(str(f)))
                    print(f"    ✓ {short:.<50s} {img_count:>5} images")
                    found += 1
                    break

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if found == 0:
        # Loose match: any folder whose name contains "Tomato"
        print("  ⚠  No exact matches. Trying loose 'Tomato' keyword scan …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)
        for root, dirs, _ in os.walk(tmp_dir):
            for d in dirs:
                if "tomato" in d.lower():
                    full_path = Path(root) / d
                    dest = DATASET_DIR / d
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(full_path, dest)
                    img_count = sum(1 for f in dest.iterdir() if _is_image(str(f)))
                    print(f"    ✓ {d:.<50s} {img_count:>5} images")
                    found += 1
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n  📂 Organised {found} class folder(s) into {DATASET_DIR}")


# ── Create backup ZIP ──────────────────────────────────────
def create_backup_zip() -> None:
    """Zip the entire PlantVillage directory into Tomato_Disease_Dataset.zip."""
    if not DATASET_DIR.is_dir():
        print("  ⚠  Dataset directory not found — cannot create ZIP.")
        return
    print(f"\n🗜️  Creating backup archive → {ZIP_PATH.name}")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in DATASET_DIR.rglob("*"):
            if fpath.is_file():
                zf.write(fpath, fpath.relative_to(BASE_DIR))
    size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"  ✅ {ZIP_PATH.name}  ({size_mb:.1f} MB)")


# ── Manual fallback instructions ───────────────────────────
def _print_manual_instructions() -> None:
    border = "═" * 62
    print(f"\n  {border}")
    print("  ║  AUTOMATED DOWNLOAD FAILED — MANUAL STEPS                ║")
    print(f"  {border}")
    print("""
  1. Go to  https://www.kaggle.com/datasets/arjuntejaswi/plant-village
  2. Download the ZIP and extract it.
  3. Copy ONLY the 10 Tomato_* folders into:
         PlantVillage/
  4. Re-run this script — it will detect the data and create the ZIP.

  Alternative: Install `datasets` from Hugging Face:
      pip install datasets
  Then re-run this script.
  """)


# ── Main ────────────────────────────────────────────────────
def main() -> None:
    print("=" * 62)
    print("  🍅 Tomato Disease Dataset Downloader")
    print("=" * 62)

    if _dataset_looks_ready():
        print(f"\n✅ Dataset already present at {DATASET_DIR}")
    else:
        print("\n🔍 Attempting autonomous download …\n")

        # Strategy 1: Hugging Face (most reliable, no auth)
        success = _try_huggingface()

        # Strategy 2: Google Drive via gdown
        if not success:
            success = _try_gdrive()

        # All strategies failed
        if not success:
            _print_manual_instructions()
            return

    # Always (re)create the backup ZIP
    create_backup_zip()

    # Summary
    class_dirs = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])
    total = 0
    print("\n📊  Dataset Summary")
    print("-" * 50)
    for cd in class_dirs:
        n = sum(1 for f in cd.iterdir() if _is_image(str(f)))
        total += n
        print(f"  {cd.name:.<42s} {n:>5}")
    print("-" * 50)
    print(f"  {'TOTAL':.<42s} {total:>5}")
    print("\n🎉 Done!  Data is ready for training.\n")


if __name__ == "__main__":
    main()
