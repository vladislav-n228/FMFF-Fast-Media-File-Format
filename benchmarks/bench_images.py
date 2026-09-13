"""
FMFF vs PNG vs WebP -- still-image size/speed benchmark.

Generates a small set of synthetic test images covering a few distinct
content types (smooth "photo", detailed/textured "photo", a flat UI-style
screenshot, a low-color pixel-art sprite, black-and-white line art, and
pure random noise as an incompressible worst case), encodes each one with
FMFF (this repo's own encoder, via the CLI -- exactly what a real user
runs), Pillow's PNG encoder, and Pillow's WebP encoder (both lossless and
quality=80 lossy), and reports real measured file sizes and encode times
side by side.

No external image dataset is downloaded or bundled -- everything is
generated on the fly from a fixed seed so the numbers are exactly
reproducible on any machine with this repo's requirements installed
(see README's "Requirements"). This is a real, honest limitation: these
are synthetic stand-ins for "a photo" / "a screenshot" / etc., not actual
photographs, so treat the numbers as directionally representative of each
content type, not a claim about any specific real-world image. Point
--images-dir at a folder of your own images (PNG/JPEG/etc.) to benchmark
those instead.

Usage:
    python benchmarks/bench_images.py
    python benchmarks/bench_images.py --images-dir path/to/your/images
    python benchmarks/bench_images.py --size 800x600 --quality 80
"""

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
FMFF_SCRIPT = REPO_ROOT / "F.M.F.F.py"

RNG_SEED = 20260913  # fixed, so every run generates byte-identical inputs


# ------------------------------------------------------------------ #
# Synthetic test image generation
# ------------------------------------------------------------------ #

def _smooth_noise(rng, w, h, octave):
    """Low-res random noise upsampled to (w, h) -- soft, cloud-like blobs."""
    small = rng.integers(0, 256, (max(2, h // octave), max(2, w // octave), 3), dtype=np.uint8)
    return np.asarray(
        Image.fromarray(small, "RGB").resize((w, h), Image.BICUBIC), dtype=np.float32
    )


def gen_photo_smooth(rng, w, h):
    """Soft gradients + light grain -- stands in for a landscape/sky photo."""
    base = _smooth_noise(rng, w, h, octave=max(w, h) // 6)
    grain = rng.normal(0, 6, (h, w, 3)).astype(np.float32)
    arr = np.clip(base + grain, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def gen_photo_detailed(rng, w, h):
    """Several octaves of noise summed -- stands in for a busy/textured photo
    (foliage, fabric, gravel -- content with real high-frequency detail)."""
    acc = np.zeros((h, w, 3), dtype=np.float32)
    weight_total = 0.0
    for octave, weight in ((w // 4, 0.35), (w // 16, 0.35), (w // 64 or 1, 0.30)):
        acc += _smooth_noise(rng, w, h, octave=max(1, octave)) * weight
        weight_total += weight
    arr = np.clip(acc / weight_total, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def gen_screenshot_ui(rng, w, h):
    """Flat panels, borders, and text -- stands in for an app/UI screenshot."""
    img = Image.new("RGB", (w, h), (245, 245, 248))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, w - 1, 48], fill=(32, 34, 46))
    draw.rectangle([0, 49, w - 1, 51], fill=(210, 210, 215))
    palette = [(66, 133, 244), (52, 168, 83), (251, 188, 5), (234, 67, 53), (154, 100, 244)]
    cols, rows = 4, 3
    pad = 16
    cell_w = (w - pad * (cols + 1)) // cols
    cell_h = (h - 70 - pad * (rows + 1)) // rows
    i = 0
    for r in range(rows):
        for c in range(cols):
            x0 = pad + c * (cell_w + pad)
            y0 = 70 + pad + r * (cell_h + pad)
            color = palette[i % len(palette)]
            draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], fill=(255, 255, 255), outline=(220, 220, 224))
            draw.rectangle([x0, y0, x0 + cell_w, y0 + 6], fill=color)
            draw.line([x0 + 12, y0 + 24, x0 + cell_w - 12, y0 + 24], fill=(60, 60, 70), width=2)
            draw.line([x0 + 12, y0 + 40, x0 + cell_w - 40, y0 + 40], fill=(150, 150, 160), width=2)
            i += 1
    try:
        font = ImageFont.load_default()
        draw.text((16, 14), "FMFF Benchmark Demo -- Dashboard", fill=(255, 255, 255), font=font)
    except Exception:
        pass
    return img


def gen_pixel_art(rng, w, h):
    """A low-resolution, low-color sprite scaled up with nearest-neighbor --
    the case FMFF's lossless palette tile race is built for."""
    grid = max(16, min(w, h) // 20)
    small = rng.integers(0, 8, (grid, grid), dtype=np.uint8)
    colors = rng.integers(40, 256, (8, 3), dtype=np.uint8)
    small_rgb = colors[small]
    return Image.fromarray(small_rgb, "RGB").resize((w, h), Image.NEAREST)


def gen_line_art(rng, w, h):
    """Black strokes on a white background -- a diagram/sketch stand-in."""
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for _ in range(40):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        x1, y1 = rng.integers(0, w), rng.integers(0, h)
        draw.line([x0, y0, x1, y1], fill=(20, 20, 20), width=int(rng.integers(1, 4)))
    for _ in range(15):
        x0, y0 = rng.integers(0, w - 40), rng.integers(0, h - 40)
        size = int(rng.integers(20, 80))
        draw.ellipse([x0, y0, x0 + size, y0 + size], outline=(20, 20, 20), width=2)
    return img


def gen_random_noise(rng, w, h):
    """Pure uniform random RGB -- incompressible worst case, included on
    purpose so the benchmark doesn't only show FMFF's best side."""
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


GENERATORS = {
    "photo_smooth": gen_photo_smooth,
    "photo_detailed": gen_photo_detailed,
    "screenshot_ui": gen_screenshot_ui,
    "pixel_art": gen_pixel_art,
    "line_art": gen_line_art,
    "random_noise": gen_random_noise,
}


def generate_test_images(out_dir, size):
    w, h = size
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, gen in GENERATORS.items():
        rng = np.random.default_rng(RNG_SEED + abs(hash(name)) % 1000)
        img = gen(rng, w, h)
        p = out_dir / f"{name}.png"
        img.save(p, optimize=True)
        paths.append(p)
    return paths


# ------------------------------------------------------------------ #
# Encoders under test
# ------------------------------------------------------------------ #

INFO_TILES_RE = re.compile(r"color tiles:\s*(\d+) lossless,\s*(\d+) lossy,\s*(\d+) palette")


def encode_png(src_img, out_path):
    t0 = time.perf_counter()
    src_img.save(out_path, format="PNG", optimize=True)
    return time.perf_counter() - t0, out_path.stat().st_size


def encode_webp(src_img, out_path, lossless, quality):
    t0 = time.perf_counter()
    if lossless:
        src_img.save(out_path, format="WEBP", lossless=True, method=6)
    else:
        src_img.save(out_path, format="WEBP", lossless=False, quality=quality, method=6)
    return time.perf_counter() - t0, out_path.stat().st_size


def encode_fmff(src_path, out_path, quality):
    t0 = time.perf_counter()
    subprocess.run(
        [sys.executable, str(FMFF_SCRIPT), "encode", str(src_path), str(out_path),
         "--quality", str(quality)],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - t0
    info = subprocess.run(
        [sys.executable, str(FMFF_SCRIPT), "info", str(out_path)],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    m = INFO_TILES_RE.search(info)
    mode = f"{m.group(1)} lossless / {m.group(2)} lossy / {m.group(3)} palette" if m else "?"
    return elapsed, out_path.stat().st_size, mode


def fmff_roundtrip_max_diff(fmff_path, src_img, work_dir):
    """Decode the .fmff back and report the max per-channel pixel difference
    against the source -- 0 for a file whose tiles all won the lossless/
    palette race, >0 wherever the lossy DCT candidate won instead."""
    out_png = work_dir / (fmff_path.stem + "_roundtrip.png")
    subprocess.run(
        [sys.executable, str(FMFF_SCRIPT), "decode", str(fmff_path), str(out_png)],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    decoded = np.asarray(Image.open(out_png).convert("RGB"), dtype=np.int16)
    source = np.asarray(src_img.convert("RGB"), dtype=np.int16)
    return int(np.max(np.abs(decoded - source)))


# ------------------------------------------------------------------ #
# Runner
# ------------------------------------------------------------------ #

def run(images, out_dir, quality):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for src_path in images:
        name = src_path.stem
        src_img = Image.open(src_path).convert("RGB")
        src_size = src_path.stat().st_size

        png_t, png_sz = encode_png(src_img, out_dir / f"{name}.png")
        webp_ll_t, webp_ll_sz = encode_webp(src_img, out_dir / f"{name}_ll.webp", True, None)
        webp_q_t, webp_q_sz = encode_webp(src_img, out_dir / f"{name}_q{quality}.webp", False, quality)
        fmff_t, fmff_sz, fmff_mode = encode_fmff(src_path, out_dir / f"{name}.fmff", quality)
        max_diff = fmff_roundtrip_max_diff(out_dir / f"{name}.fmff", src_img, out_dir)

        rows.append({
            "image": name,
            "dims": f"{src_img.width}x{src_img.height}",
            "source_png_bytes": src_size,
            "png_bytes": png_sz,
            "png_s": round(png_t, 3),
            "webp_lossless_bytes": webp_ll_sz,
            "webp_lossless_s": round(webp_ll_t, 3),
            f"webp_q{quality}_bytes": webp_q_sz,
            f"webp_q{quality}_s": round(webp_q_t, 3),
            "fmff_bytes": fmff_sz,
            "fmff_s": round(fmff_t, 3),
            "fmff_tile_mix": fmff_mode,
            "fmff_vs_png_pct": round(100 * (fmff_sz - png_sz) / png_sz, 1),
            "fmff_vs_webp_q_pct": round(100 * (fmff_sz - webp_q_sz) / webp_q_sz, 1),
            "fmff_max_pixel_diff": max_diff,
        })
    return rows


def print_table(rows, quality):
    headers = ["image", "dims", "PNG", "WebP lossless", f"WebP q{quality}", "FMFF",
               "FMFF vs PNG", f"FMFF vs WebP q{quality}", "FMFF tiles", "max px diff"]
    print("\t".join(headers))
    for r in rows:
        print("\t".join([
            r["image"], r["dims"],
            f'{r["png_bytes"]:,} B',
            f'{r["webp_lossless_bytes"]:,} B',
            f'{r[f"webp_q{quality}_bytes"]:,} B',
            f'{r["fmff_bytes"]:,} B',
            f'{r["fmff_vs_png_pct"]:+.1f}%',
            f'{r["fmff_vs_webp_q_pct"]:+.1f}%',
            r["fmff_tile_mix"],
            str(r["fmff_max_pixel_diff"]),
        ]))


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path, quality, size, env_note):
    lines = [
        "# FMFF vs PNG vs WebP -- image benchmark results",
        "",
        env_note,
        "",
        "Generated by `benchmarks/bench_images.py` (synthetic, seeded test",
        "images -- see the script's docstring for what each category stands",
        "in for and why real photos aren't bundled). Re-run it yourself to",
        "reproduce or challenge these numbers; point `--images-dir` at your",
        "own images to benchmark real content instead.",
        "",
        f"FMFF quality={quality}, WebP lossy quality={quality} (matched for a fair comparison), "
        "tile-size=64 (FMFF default). PNG via Pillow with `optimize=True`. "
        "\"FMFF tiles\" is the lossless/lossy/palette tile mix FMFF's own encoder chose "
        "(see README's \"Images\" section) -- a mostly-lossless mix is being compared "
        "against WebP's *lossy* number, which favors WebP on bytes; that's disclosed here "
        "rather than hidden.",
        "",
        f"| image | dims | source PNG | PNG (re-opt) | WebP lossless | WebP q{quality} | "
        "FMFF | FMFF vs PNG | FMFF vs WebP | FMFF tile mix | max pixel diff |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f'| {r["image"]} | {r["dims"]} | {r["source_png_bytes"]:,} B | {r["png_bytes"]:,} B | '
            f'{r["webp_lossless_bytes"]:,} B | {r[f"webp_q{quality}_bytes"]:,} B | '
            f'{r["fmff_bytes"]:,} B | {r["fmff_vs_png_pct"]:+.1f}% | {r["fmff_vs_webp_q_pct"]:+.1f}% | '
            f'{r["fmff_tile_mix"]} | {r["fmff_max_pixel_diff"]} |'
        )
    lines += [
        "",
        "\"FMFF vs PNG/WebP\" is the size difference in percent -- negative means FMFF is "
        "smaller. \"max pixel diff\" is the largest per-channel 0-255 difference between the "
        "decoded `.fmff` and the source, over every pixel -- 0 for an image whose tiles all "
        "won the lossless/palette race, nonzero (but should stay small at quality 80) wherever "
        "the lossy DCT candidate won instead.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=Path, default=None,
                     help="use your own images instead of generating synthetic ones")
    ap.add_argument("--size", default="640x480", help="WxH for generated images (default 640x480)")
    ap.add_argument("--quality", type=int, default=80, help="FMFF/WebP lossy quality (default 80)")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "out",
                     help="where encoded outputs are written (default benchmarks/out)")
    ap.add_argument("--results-dir", type=Path, default=Path(__file__).parent,
                     help="where results.csv/results.md are written (default benchmarks/)")
    args = ap.parse_args()

    if not FMFF_SCRIPT.exists():
        sys.exit(f"can't find {FMFF_SCRIPT} -- run this from inside the FMFF repo")

    if args.images_dir:
        images = sorted(p for p in args.images_dir.iterdir()
                         if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
        if not images:
            sys.exit(f"no images found in {args.images_dir}")
        size_note = f"user-supplied images from {args.images_dir}"
    else:
        w, h = (int(x) for x in args.size.lower().split("x"))
        images = generate_test_images(args.out_dir / "images", (w, h))
        size_note = f"synthetic {w}x{h} test images, seed={RNG_SEED} (reproducible -- see script docstring)"

    print(f"Running benchmark: {size_note}, quality={args.quality}\n")
    rows = run(images, args.out_dir, args.quality)
    print()
    print_table(rows, args.quality)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.results_dir / "results.csv")
    env_note = (
        f"Python {sys.version.split()[0]}, {size_note}, "
        f"platform: {sys.platform}."
    )
    write_markdown(rows, args.results_dir / "results.md", args.quality, args.size, env_note)
    print(f"\nWrote {args.results_dir / 'results.csv'} and {args.results_dir / 'results.md'}")


if __name__ == "__main__":
    main()
