"""One-time BiRefNet mask precomputation for all PlantVillage images.

Run this script before training to populate ``data/masks/`` with per-pixel foreground
masks.  ``RandomBackground`` will use these masks instead of the heuristic brightness
threshold, producing clean leaf cutouts regardless of background color.

Usage
-----
    # GPU (recommended, ~30-45 min for 54k images with batch_size=8):
    python scripts/precompute_masks.py

    # Resume after interruption (skips already-computed masks):
    python scripts/precompute_masks.py --resume

    # Adjust batch size (lower if OOM, higher if VRAM headroom):
    python scripts/precompute_masks.py --batch-size 4

    # CPU-only (much slower, not recommended):
    python scripts/precompute_masks.py --device cpu --batch-size 1

    # Custom paths:
    python scripts/precompute_masks.py --data-root data/my_dataset --mask-dir data/my_masks

Mask format
-----------
Grayscale PNG, same filename as the source image, stored under ``mask_dir`` in the same
subdirectory structure as the dataset.  White (255) = leaf foreground; black (0) = background.

Model
-----
ZhengPeng7/BiRefNet from HuggingFace.  Downloaded automatically on first run (~500 MB).
Requires ``transformers>=4.35.0``, ``einops``, ``kornia``, ``timm`` (all in requirements.txt).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from greenvision.constants import DATA_ROOT, MASK_DIR

BIREFNET_MODEL_ID = "ZhengPeng7/BiRefNet"
DEFAULT_INPUT_SIZE = 512   # 512×512 is sufficient; original leaves are ≤1024px
DEFAULT_BATCH_SIZE = 8


def _build_transform(size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _load_model(device: str):
    from transformers import AutoModelForImageSegmentation

    print(f"Loading {BIREFNET_MODEL_ID} …")
    model = AutoModelForImageSegmentation.from_pretrained(
        BIREFNET_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.to(device).eval()
    print(f"Model ready on {device}.")
    return model


def _process_batch(
    model,
    batch_imgs: list[Image.Image],
    batch_paths: list[Path],
    batch_mask_paths: list[Path],
    transform: transforms.Compose,
    device: str,
) -> int:
    """Run BiRefNet on a batch of images; save masks; return error count."""
    tensors = []
    orig_sizes = []
    for img in batch_imgs:
        orig_sizes.append(img.size)  # (W, H)
        tensors.append(transform(img.convert("RGB")))

    inp = torch.stack(tensors).to(device)
    try:
        with torch.no_grad():
            # autocast lets the GPU use float16 where safe (Tensor Cores), giving
            # ~2x speedup on Ampere/Ada GPUs with no accuracy loss for inference.
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device == "cuda"
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with autocast_ctx:
                preds = model(inp)
        # BiRefNet returns a list; [-1] is the final output, shape [B, 1, H, W]
        pred_batch = preds[-1].sigmoid().squeeze(1).float().cpu().numpy()  # [B, H, W]
    except Exception as exc:
        for p in batch_paths:
            tqdm.write(f"ERROR batch containing {p.name}: {exc}")
        return len(batch_imgs)

    errors = 0
    for i, (mask_arr, mask_path, orig_size) in enumerate(
        zip(pred_batch, batch_mask_paths, orig_sizes)
    ):
        try:
            mask = Image.fromarray((mask_arr * 255).astype(np.uint8), mode="L")
            if mask.size != orig_size:
                mask = mask.resize(orig_size, Image.BILINEAR)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask.save(mask_path)
        except Exception as exc:
            tqdm.write(f"ERROR saving {mask_path}: {exc}")
            errors += 1
    return errors


def _collect_images(data_root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return [p for p in data_root.rglob("*") if p.suffix in exts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute BiRefNet leaf masks.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Root of the PlantVillage dataset (default: constants.DATA_ROOT).",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=MASK_DIR,
        help="Output directory for mask PNGs (default: constants.MASK_DIR).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images whose mask PNG already exists.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Images per forward pass (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=DEFAULT_INPUT_SIZE,
        help=f"BiRefNet input resolution (default: {DEFAULT_INPUT_SIZE}px square).",
    )
    args = parser.parse_args()

    image_paths = _collect_images(args.data_root)
    if not image_paths:
        print(f"No images found under {args.data_root}. Check --data-root.")
        sys.exit(1)

    print(f"Found {len(image_paths):,} images under {args.data_root}")
    print(f"Masks will be saved to {args.mask_dir}")
    print(f"Batch size: {args.batch_size}  Input size: {args.input_size}px  Device: {args.device}")

    model = _load_model(args.device)
    transform = _build_transform(args.input_size)

    skipped = 0
    errors = 0
    batch_imgs: list[Image.Image] = []
    batch_paths: list[Path] = []
    batch_mask_paths: list[Path] = []

    def flush_batch() -> None:
        nonlocal errors
        if batch_imgs:
            errors += _process_batch(
                model, batch_imgs, batch_paths, batch_mask_paths, transform, args.device
            )
            batch_imgs.clear()
            batch_paths.clear()
            batch_mask_paths.clear()

    with tqdm(image_paths, desc="BiRefNet masks", unit="img") as pbar:
        for img_path in pbar:
            rel = img_path.relative_to(args.data_root)
            mask_path = args.mask_dir / rel.with_suffix(".png")

            if args.resume and mask_path.exists():
                skipped += 1
                continue

            try:
                img = Image.open(img_path)
                batch_imgs.append(img)
                batch_paths.append(img_path)
                batch_mask_paths.append(mask_path)
            except Exception as exc:
                tqdm.write(f"ERROR opening {img_path}: {exc}")
                errors += 1
                continue

            if len(batch_imgs) >= args.batch_size:
                flush_batch()

    flush_batch()  # process any remaining images

    total = len(image_paths)
    done = total - skipped - errors
    print(f"\nDone. Computed: {done:,}  Skipped: {skipped:,}  Errors: {errors:,}")
    print(f"Masks saved to {args.mask_dir}")


if __name__ == "__main__":
    main()
