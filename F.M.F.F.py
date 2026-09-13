"""
FMFF -- Fast Media File Format
Reference encoder / decoder / viewer for the .fmff media container (v4).

Container layout for a still image (all offsets absolute from byte 0):
    [fixed header]  [tile index]  [frame table]  [thumbnail payload]  [tile payloads ...]

The header has a constant size and lives at offset 0, so a reader always
knows exactly where the tile index and frame table are without scanning
the file. Images are FMFF's own hybrid codec: each tile is raced three
ways -- losslessly (a PNG-style filter + zlib), losslessly again as a
palette/indexed-color tile (a small color table + 1 byte/pixel index,
for tiles with <=256 distinct colors -- see _palette_encode), and lossily
(block DCT + quantize, entropy-coded with whichever of zlib/bz2 comes out
smaller for that tile) -- whichever of the three is smallest is kept, so
flat/text/line-art (and any originally-paletted source, GIF included)
stays pixel-exact and small, while photo/gradient regions compress hard.
Every tile has its own (offset, length) in the index, so a client could
fetch one tile via an HTTP Range request without touching the rest of
the file. This part is genuinely FMFF's own codec, not a wrapper around
anything.

A still image's tile index can also hold one or more named *versions*
(original/retouched/... -- see FMFFEncoder.add_version) alongside the
base image: each version (layer LAYER_VERSION instead of LAYER_FULL)
stores only the tiles that changed relative to the version before it,
sharing the base image's tile grid and quality -- the same "unchanged
gets no entry at all" trick encode_image_sequence already uses for an
animation frame, applied to a version chain instead of a time axis.
This needed no header/entry format change at all -- a reader that only
ever looks for LAYER_FULL/LAYER_THUMB entries (any reader before this
existed included) just never sees them, and still decodes the base
image exactly as before. See README's "Versions" section.

Container layout for a video (frame_count set, is_video true):
    [fixed header]  [thumbnail entry]  [segment table]  [thumbnail payload]
    [media blob: init segment][segment 0][segment 1]...

Video does NOT use FMFF's own tile codec: it holds AV1 (video) + Opus
(audio) produced by FFmpeg, plus a small instant thumbnail encoded the
same way stills are. This is a deliberate trade: FMFF's own intra-only
tile codec has no motion compensation, so it can never get within the
same order of magnitude of a real video codec's file size -- there is no
amount of tuning that fixes that, it's an architectural ceiling. AV1
gives real inter-frame compression, real speed (FFmpeg/SVT-AV1
parallelizes internally in C, no Python loop over frames), and audio,
all "for free" from a mature codec -- at the cost of video no longer
holding FMFF's own hybrid lossless tiles or alpha channel the way images
do. Both AV1 (via SVT-AV1) and Opus are royalty-free formats, so this
carries none of the patent baggage patented codecs like H.264/HEVC would.
Needs FFmpeg on PATH built with libsvtav1 + libopus (e.g.
`winget install BtbN.FFmpeg.LGPL.8.1` on Windows) -- an LGPL build is
used deliberately so linking is not a concern (FMFF only ever shells out
to the ffmpeg.exe process, never links against it).

Unlike a plain MP4/MKV/WebM, the stream is NOT one opaque blob: FFmpeg
is asked for fragmented MP4 output (`-movflags frag_keyframe+...`, the
same mechanism browsers use for MSE/DASH streaming), which naturally
splits into a small init chunk (codec setup) followed by consecutive,
keyframe-aligned (moof+mdat) fragments, each a few seconds long. FMFF
stores each fragment's length and CRC32 in its own [segment table]
(see SEGMENT_FMT) instead of re-merging them into one undifferentiated
blob. This is what "error resilience" means for video, not just images:
extract_media() drops any segment that fails its CRC32 (or that a
truncated/still-downloading file simply doesn't have) and reconstructs
everything else into a still-valid, still-playable file -- the video
just skips however many seconds that segment covered, instead of the
whole file refusing to open. A corrupt init segment is the one thing
this can't route around (it carries the codec setup every later segment
depends on), so that specific case still raises a clear error rather
than degrading.

A JPEG source is a third case (content_mode 2, still using the [thumbnail]
[media blob] layout above, no tile entries): re-encoding it through FMFF's
own tile codec means re-quantizing pixels that were already quantized once
by the source JPEG, which routinely comes out *larger* than the JPEG, not
smaller (see encode_image_jpeg_passthrough's docstring). Instead FMFF pulls
the JPEG's own already-quantized DCT coefficients straight out of it (via
`jpeglib`/libjpeg, losslessly -- no IDCT, no requantization, no pixels
touched) and just re-entropy-codes those exact same coefficients with a
stronger general-purpose compressor than JPEG's baseline Huffman tables:
typically 10-40% smaller, and reconstructs pixels that are identical to
what the JPEG itself decodes to (within the same IDCT-rounding tolerance
any two compliant JPEG decoders can differ by). Needs `pip install
jpeglib`; without it, or for a JPEG this can't handle (progressive scan,
not 3-component), encoding falls back to the normal tile codec with an
automatically-lowered quality (see _default_quality_for).

A document (PDF/.txt) is a fourth case (content_mode 4, same layout as
JPEG passthrough): the original file's bytes, not a rendered picture of
its pages -- see encode_document's docstring, and the "document pages"
section further down, for why an earlier version of this rasterized
pages instead and what was wrong with that (no text layer/search/
hyperlinks/forms, and routinely *bigger* than the source despite
throwing all of that away). The bytes are raced against a general-
purpose compressor and stored however comes out smaller, so this can
never end up more than this container's own small fixed overhead
bigger than the source. Decode recovers the exact original file,
byte-for-byte; a small first-page render is stored only as an instant-
preview thumbnail, not as the document's actual content.

RGB / RGBA, 8-bit only for still images. Fields for bit depth / color
space are reserved for a future HDR extension but not implemented yet.

The `view` window is a general media viewer on top of all this: besides
.fmff it can also open common image formats (via Pillow) and video files
(via OpenCV, using GPU-accelerated decode when available) so there is one
place to look at any media file, plus a screen-region screenshot tool.
Video playback (both .fmff and plain video files) opens in the system's
default player -- a real separate window with real hardware decode,
real audio, and a real seek bar, all of which FMFF's own in-window
loop (_load_video: OpenCV, GPU-accelerated decode when available, but
silent and confined to this window's own canvas) can't match -- see
_open_with_default_player for why that's used instead of shelling out
to `ffplay`, falling back to that silent in-window preview only if
launching the default player fails outright. The one exception is a
video with a real alpha channel (see encode_video's alpha handling):
no external player can composite two separate tracks into transparency
on the fly, so that specific case always stays in-window, silent,
decoded and blended frame-by-frame against the viewer's own dark
canvas (see _load_video_with_alpha).
"""

import argparse
import atexit
import base64
import bz2
import io
import json
import lzma
import multiprocessing as mp
import os
import queue
import shutil
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import zlib
from collections import namedtuple
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import tkinter as tk
    from tkinter import filedialog, simpledialog, ttk
    from PIL import ImageGrab, ImageTk
except ImportError:
    tk = None

try:
    import winreg  # Windows-only stdlib module -- see cmd_register_filetype
except ImportError:
    winreg = None

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    TkinterDnD = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import jpeglib
except ImportError:
    jpeglib = None

try:
    import pymupdf
except ImportError:
    pymupdf = None

try:
    import rawpy
except ImportError:
    rawpy = None

MAGIC = b"FMFF"
VERSION = 13

# Target length of one independently CRC32-checked, droppable video
# fragment -- see the module docstring's "Container layout for a video"
# section. A few seconds is a reasonable balance: fine-grained enough that
# losing one fragment to corruption/truncation is a small, localized gap,
# coarse-grained enough that the fragment/keyframe overhead stays small.
VIDEO_SEGMENT_SECONDS = 4

# Every this-manyth version stores every tile in full, not just the ones
# that changed (see FMFFEncoder.add_version) -- the same "periodic
# checkpoint" idea video's own I-frames use, and for the same reason:
# without it, decoding version N always means replaying every version
# 1..N in order (see FMFFDecoder.full()), which grows without bound as
# a version chain gets longer. A periodic full version gives full() (and
# add_version's own change-comparison) a nearby restart point instead,
# bounding replay depth to at most this many versions regardless of how
# long the whole chain gets -- at the cost of one full version's worth
# of extra storage every this-many versions, not a full copy every
# version. 8 mirrors this file's other periodic-checkpoint tuning
# (VIDEO_SEGMENT_SECONDS above): frequent enough that replay depth never
# grows large, coarse enough that the extra storage stays a small
# fraction of the whole chain.
VERSION_SNAPSHOT_INTERVAL = 8

PLANE_COLOR = 0
PLANE_ALPHA = 1
# A single-channel, 8-bit-per-pixel plane attached to one version of a
# still image (see FMFFEncoder.add_mask) -- a selection mask, or any
# other per-pixel annotation a caller wants back out unchanged. Encoded
# exactly like PLANE_ALPHA (same lossless tile codec, never raced against
# the lossy DCT path -- see _decode_entry), but never composited as
# transparency: full()/full_sequence() never look for it, so it has no
# effect on how the image itself decodes; only FMFFDecoder.extract_mask
# reads it. Unlike a color/alpha tile, a mask plane is never inherited
# across the version chain (see add_mask's own docstring for why) -- it
# belongs to exactly the (layer, frame) it was stored under.
PLANE_MASK = 2

MODE_LOSSLESS = 0
MODE_LOSSY = 1
# A third, still-lossless candidate raced alongside the two above (see
# _palette_encode): a small per-tile color table plus a 1-byte-per-pixel
# index, for tiles with <=256 distinct colors. Flat/graphic content --
# icons, line art, and notably any GIF source, which *started* as exactly
# this before being expanded to RGB -- stores far smaller this way than
# 3 raw bytes/pixel ever can, filtered-and-zlib'd or not.
MODE_PALETTE = 2

# content_mode (stored in the header's former "color_space" byte -- that
# field was never used for anything else): which of FMFF's storage
# schemes a file uses. Video, audio, and JPEG-passthrough all skip the
# tile index/frame table entirely and just hold one opaque blob after
# the thumbnail; only CONTENT_IMAGE uses FMFF's own tile codec.
CONTENT_IMAGE = 0
CONTENT_VIDEO = 1
CONTENT_JPEG_PASSTHROUGH = 2
# Audio reuses video's entire segmented-media-blob mechanism (fragmented
# MP4, per-segment CRC32, the works -- see encode_audio) rather than
# being a fourth storage scheme in its own right; it's a new
# content_mode only so the decoder can tell "this segmented blob is a
# picture" from "this segmented blob is sound" apart.
CONTENT_AUDIO = 3
# A document (PDF/.txt) reuses JPEG-passthrough's single-opaque-blob
# layout (see encode_image_jpeg_passthrough) rather than being a fifth
# storage scheme of its own: the original file's bytes (optionally
# recompressed losslessly, whichever is smaller -- see encode_document)
# are the blob, exactly like a passthrough-stored JPEG's coefficients
# are. It's a new content_mode only so the decoder can tell "this blob
# is a repacked JPEG" from "this blob is a whole other file format"
# apart -- see FMFFDecoder.extract_document.
CONTENT_DOCUMENT = 4

LAYER_THUMB = 0
LAYER_FULL = 1
# A named "version" of the same still image (original/retouched/... --
# see FMFFEncoder.add_version), sharing the base image's tile grid and
# quality but storing only the tiles that changed relative to the
# version before it -- not a full second copy (see add_version's own
# docstring for why: the same "unchanged gets no entry" saving
# encode_image_sequence already gives an animation frame, here applied
# to a version chain). Entry.frame doubles as the version index here
# (1, 2, ... -- 0 is always the LAYER_FULL tiles already in every
# still-image .fmff, never stored as a LAYER_VERSION entry itself), the
# same way it's an animation frame index for LAYER_FULL entries -- the
# two meanings never collide since versions are only ever added to a
# non-animated file (see add_version). Existing code that only ever
# looked for LAYER_FULL/LAYER_THUMB (full(), full_sequence(),
# thumbnail(), tile_byte_ranges()'s default) ignores LAYER_VERSION
# entries completely, so a file with extra
# versions still opens and decodes to exactly its base image in any reader
# that predates this.
LAYER_VERSION = 2

# fmt: off
HEADER_FMT = ("<4sHIIBBBHHHBBBIIIHHIIIIBII"
              "IIII"
              "IIIIII"
              "II")
#             magic version width height bit_depth channels color_space
#             tile_size tiles_x tiles_y has_alpha quality
#             is_video frame_count fps_x100 num_index_entries
#             thumb_w thumb_h header_size index_offset frame_table_offset data_offset
#             has_audio media_blob_offset media_blob_length
#             init_segment_length init_segment_crc32 segment_count segment_table_offset
#             alpha_media_blob_offset alpha_media_blob_length
#             alpha_init_segment_length alpha_init_segment_crc32
#             alpha_segment_count alpha_segment_table_offset
#             metadata_offset metadata_length
# fmt: on
# metadata_offset/metadata_length point at an optional trailing JSON blob
# (see _pack_metadata/_unpack_metadata) -- source tags (artist/album/...
# for audio and video, from the container's own tags) and/or a raw EXIF
# blob (for a still image, passed through byte-for-byte rather than
# re-derived, the same passthrough philosophy JPEG's own DCT-coefficient
# path already uses). Both zero when there's nothing to store. Appended
# after every other section so adding it never shifted any existing
# offset math -- see each encode_* method's header-writing code.
# For CONTENT_AUDIO (see encode_audio), several of these fields are
# reused with an audio-specific meaning instead of being left at 0:
# is_video is set to 1 (it really means "has a segmented media blob to
# read instead of a tile index", true for audio too -- see
# FMFFDecoder.__init__), channels is the actual audio channel count
# (1=mono, 2=stereo, ...), frame_count holds duration_ms, and fps_x100
# holds the sample rate (not scaled by 100 -- the field is just a raw
# uint32 slot being reused for a differently-shaped number, the same
# way color_space already doubles as content_mode). width/height/
# tile_size/tiles_x/tiles_y/has_alpha/quality are all meaningless for
# audio and left at 0.
HEADER_SIZE = struct.calcsize(HEADER_FMT)

ENTRY_FMT = "<IBHHBBHHIII"
#            frame layer tile_x tile_y plane mode tile_w tile_h offset length crc32
# tile_x/tile_y are a tile-grid cell index (multiply by the header's
# tile_size for a pixel offset) for a still image's LAYER_FULL entries,
# but literal pixel (x, y) for an animated file's -- an animated file has
# no fixed tile grid at all, each changed frame stores one rectangle
# sized to its own change (see encode_image_sequence / full_sequence).
# A LAYER_VERSION entry always uses tile-grid coordinates, the same as
# LAYER_FULL's still-image case -- see LAYER_VERSION's own comment.
ENTRY_SIZE = struct.calcsize(ENTRY_FMT)

FRAME_FMT = "<III"
#            timestamp_ms entry_start entry_count
FRAME_SIZE = struct.calcsize(FRAME_FMT)

SEGMENT_FMT = "<II"
#              length crc32
SEGMENT_SIZE = struct.calcsize(SEGMENT_FMT)


# ---------------------------------------------------------------- DCT core

def _dct_matrix(n=8):
    k = np.arange(n)
    x = np.arange(n)
    m = np.cos((2 * x[None, :] + 1) * k[:, None] * np.pi / (2 * n))
    m *= np.sqrt(2.0 / n)
    m[0, :] *= 1.0 / np.sqrt(2.0)
    return m


_T = _dct_matrix(8)

_LUMA_Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float32)

_CHROMA_Q = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float32)


def _scale_quant(table, quality):
    quality = max(1, min(100, quality))
    scale = 5000.0 / quality if quality < 50 else 200.0 - 2 * quality
    scaled = np.floor((table * scale + 50) / 100)
    return np.clip(scaled, 1, 255).astype(np.float32)


def _pad_to(arr, ph, pw):
    h, w = arr.shape[:2]
    if h == ph and w == pw:
        return arr
    return np.pad(arr, ((0, ph - h), (0, pw - w), (0, 0)), mode="edge")


def _to_blocks(plane):
    h, w = plane.shape
    return plane.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)


def _from_blocks(blocks, h, w):
    nby, nbx = blocks.shape[:2]
    return blocks.transpose(0, 2, 1, 3).reshape(h, w)[:h, :w]


_RGB2YCC = np.array([
    [0.299, 0.587, 0.114],
    [-0.168736, -0.331264, 0.5],
    [0.5, -0.418688, -0.081312],
], dtype=np.float32)
_YCC2RGB = np.linalg.inv(_RGB2YCC)


def _rgb_to_ycc(rgb):
    ycc = rgb @ _RGB2YCC.T
    ycc[:, :, 1:] += 128.0
    return ycc


def _ycc_to_rgb(ycc):
    ycc = ycc.copy()
    ycc[:, :, 1:] -= 128.0
    return ycc @ _YCC2RGB.T


# ------------------------------------------------------------ lossless codec
# "Up" scanline filter (same idea PNG uses): residual[y] = pixel[y] - pixel[y-1]
# mod 256. Because addition mod 256 is associative/commutative, reconstruction
# is just a running sum: recon = cumsum(residual) mod 256 -- fully vectorized,
# no per-pixel loop needed.

# zlib's own recommended default, not its max (9). Level 9 spends a lot of
# extra search time for very little extra ratio past level 6 on real image/
# animation-frame data -- measured on one large (~880KB) filtered region:
# level 9 took 0.062s for 34,945 bytes, level 6 took 0.009s (~7x faster)
# for 36,350 bytes (~4% bigger). That 4% is a fair trade for 7x, and it
# mattered a lot once a single "tile" could cover a large chunk of a big
# animated frame instead of always being a small fixed-size tile (see
# encode_image_sequence): profiling a slow encode on a 1080x1080, 129-frame
# source showed zlib.compress alone was ~72% of total encode time at
# level 9. Decompression speed is unaffected by which level compressed a
# stream -- this is an encode-time-only tradeoff.
_ZLIB_LEVEL = 6


def lossless_encode(planes):
    # planes: (C, H, W) uint8
    planes = planes.astype(np.int16)
    filtered = np.empty_like(planes)
    filtered[:, 0, :] = planes[:, 0, :]
    filtered[:, 1:, :] = (planes[:, 1:, :] - planes[:, :-1, :]) % 256
    return zlib.compress(filtered.astype(np.uint8).tobytes(), _ZLIB_LEVEL)


def lossless_decode(data, c, h, w):
    buf = np.frombuffer(zlib.decompress(data), dtype=np.uint8).reshape(c, h, w)
    recon = np.cumsum(buf.astype(np.int64), axis=1) % 256
    return recon.astype(np.uint8)


# --------------------------------------------------------------- lossy codec
# Entropy coder for the *quantized DCT coefficients*: bz2 usually wins on
# real photo content (its block-sorting transform suits the zero-heavy,
# semi-periodic pattern quantization leaves behind better than LZ77 does)
# -- but bz2 also carries a large fixed per-stream overhead (~30-40 bytes)
# and, unlike zlib, can come out *larger than the input* on genuinely
# high-entropy data (fine noise/grain textures, which quantization doesn't
# tame nearly as well as it does smooth photo gradients). Since every tile
# is its own independent stream, that overhead/worst-case is paid per
# tile, so both are tried and the smaller one kept, tagged with which --
# this can never do worse than plain zlib alone, whatever the content.
# The lossless PNG-style path stays on zlib only, which already tested
# better there (flat/text residuals suit LZ77's local-repeat matching, and
# there's no comparably bad worst case to guard against).

_TILE_LOSSY_CODECS = {b"Z": (lambda b: zlib.compress(b, _ZLIB_LEVEL), zlib.decompress),
                      b"B": (lambda b: bz2.compress(b, 9), bz2.decompress)}

def lossy_encode(rgb_tile, quality):
    h, w = rgb_tile.shape[:2]
    ph, pw = -(-h // 8) * 8, -(-w // 8) * 8
    ycc = _rgb_to_ycc(_pad_to(rgb_tile, ph, pw).astype(np.float32)) - 128.0

    lq, cq = _scale_quant(_LUMA_Q, quality), _scale_quant(_CHROMA_Q, quality)
    out = []
    for ch, q in ((0, lq), (1, cq), (2, cq)):
        blocks = _to_blocks(ycc[:, :, ch])
        # T @ block @ T.T per 8x8 block, batched over the (nby, nbx) leading
        # dims via @'s stacked-matrix broadcasting -- numerically identical
        # to the equivalent np.einsum("ij,abjk,kl->abil", ...) (previously
        # used here) but BLAS-backed instead of einsum's generic evaluator,
        # measured ~19x faster on a large block batch. This was the actual
        # bottleneck behind a slow encode/decode on a large animated frame
        # once the tile-grid rewrite let a single "tile" cover a big region
        # (see encode_image_sequence) instead of many small 64x64 ones.
        coeffs = _T @ blocks @ _T.T
        out.append(np.round(coeffs / q).astype(np.int16))
    raw = np.concatenate([o.ravel() for o in out]).tobytes()
    tag, payload = min(
        ((t, c(raw)) for t, (c, _) in _TILE_LOSSY_CODECS.items()),
        key=lambda kv: len(kv[1]))
    return tag + payload


def lossy_decode(data, h, w, quality):
    ph, pw = -(-h // 8) * 8, -(-w // 8) * 8
    nby, nbx = ph // 8, pw // 8
    flat = np.frombuffer(_TILE_LOSSY_CODECS[data[:1]][1](data[1:]), dtype=np.int16)
    per_ch = nby * nbx * 64
    lq, cq = _scale_quant(_LUMA_Q, quality), _scale_quant(_CHROMA_Q, quality)

    ycc = np.empty((ph, pw, 3), dtype=np.float32)
    for i, q in enumerate((lq, cq, cq)):
        coeffs = flat[i * per_ch:(i + 1) * per_ch].reshape(nby, nbx, 8, 8).astype(np.float32) * q
        # See lossy_encode's forward transform for why this is @ instead
        # of einsum -- same T.T @ block @ T per block, ~19x faster.
        blocks = _T.T @ coeffs @ _T
        ycc[:, :, i] = _from_blocks(blocks, ph, pw)

    rgb = _ycc_to_rgb(ycc + 128.0)
    return np.clip(rgb[:h, :w, :], 0, 255).astype(np.uint8)


# ------------------------------------------------------------ palette codec
# A third candidate raced alongside lossless/lossy for every tile (see
# _encode_tile_task): pack each pixel's 3 bytes into one uint32 so
# np.unique can run its fast 1-D path instead of the much slower
# axis=0-on-rows path (measured ~12x faster on a noisy 64x64 tile where
# this candidate loses anyway and the check itself must stay cheap), and
# if that tile has <=256 distinct colors, store a small color table plus
# a 1 byte/pixel index instead of 3 raw bytes/pixel. This is exactly what
# GIF/paletted source content already was before being expanded to RGB
# for tiling, so it reliably beats the raw-RGB lossless path on that kind
# of content -- on a synthetic flat-color animated-GIF-style test clip,
# this cut total changed-tile payload by more than half versus lossless-
# only. Purely lossless, never a source of drift or quality loss; it can
# only ever win the race, never make a tile worse, since the raw-RGB
# candidates are still computed and compared against it.

def _palette_encode(rgb_tile):
    h, w = rgb_tile.shape[:2]
    flat = rgb_tile.reshape(-1, 3).astype(np.uint32)
    packed = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    uniq_packed, inv = np.unique(packed, return_inverse=True)
    if len(uniq_packed) > 256:
        return None  # index wouldn't fit in a byte -- let lossless/lossy race it out instead
    palette = np.empty((len(uniq_packed), 3), dtype=np.uint8)
    palette[:, 0] = (uniq_packed >> 16) & 0xFF
    palette[:, 1] = (uniq_packed >> 8) & 0xFF
    palette[:, 2] = uniq_packed & 0xFF
    idx = inv.astype(np.uint8).tobytes()
    tag, compressed = min(
        ((t, c(idx)) for t, (c, _) in _TILE_LOSSY_CODECS.items()),
        key=lambda kv: len(kv[1]))
    return struct.pack("<H", len(uniq_packed)) + palette.tobytes() + tag + compressed


def _palette_decode(data, h, w):
    n_colors = struct.unpack_from("<H", data, 0)[0]
    off = 2
    palette = np.frombuffer(data, dtype=np.uint8, count=n_colors * 3, offset=off).reshape(n_colors, 3)
    off += n_colors * 3
    idx_bytes = _TILE_LOSSY_CODECS[data[off:off + 1]][1](data[off + 1:])
    idx = np.frombuffer(idx_bytes, dtype=np.uint8, count=h * w).reshape(h, w)
    return palette[idx]


# ------------------------------------------------------------- FFmpeg helpers
# Video encoding/decoding/playback all shell out to a real `ffmpeg` (and
# `ffplay`/`ffprobe`) executable -- never linked against, just invoked as a
# subprocess, so licensing is a non-issue regardless of which FFmpeg build
# is installed. See the module docstring for why video uses this instead of
# FMFF's own codec.

def _find_exe(name):
    found = shutil.which(name)
    if found:
        return found
    exe = name + (".exe" if os.name == "nt" else "")
    # a just-installed winget package may not be on PATH yet in this process
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if base.is_dir():
        matches = sorted(base.glob(f"**/{exe}"), reverse=True)
        if matches:
            return str(matches[0])
    return None


def _open_with_default_player(path):
    """Launch path with whatever the OS has associated with its file
    extension -- the same thing double-clicking it in Explorer/Finder
    does. The viewer used to shell out to `ffplay` for real audio/seek
    controls, but direct A/B testing found ffplay's own SDL2 rendering
    produces visible block corruption on at least one real system
    (confirmed: the exact same file, clean in the system's own default
    player, corrupted only in ffplay's window -- true even for a plain,
    never-touched-by-FMFF source video, so it's a bug in ffplay/SDL on
    that system, not anything FMFF encodes, decodes, or can fix with
    flags). The system's default player is what already proved reliable,
    so that's what plays video now, at the cost of not necessarily being
    inside a dedicated single-purpose window."""
    if os.name == "nt":
        os.startfile(path)  # noqa: Windows-only API, guarded above
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])




def _find_ffmpeg():
    return _find_exe("ffmpeg")


def _find_ffprobe():
    return _find_exe("ffprobe")


# Pixel formats ffmpeg can decode that carry an actual alpha plane -- a
# source using one of these is treated as a video-with-transparency
# source (see encode_video's alpha handling / the module docstring).
_ALPHA_PIX_FMTS = {
    "yuva420p", "yuva422p", "yuva444p",
    "yuva420p10le", "yuva422p10le", "yuva444p10le",
    "rgba", "bgra", "argb", "abgr", "ya8",
}

# Text-based subtitle codecs FFmpeg can losslessly (as text, not pixels)
# convert into MP4's own "mov_text" timed-text track -- see encode_video's
# subtitle handling. Deliberately excludes image-based subtitle codecs
# (hdmv_pgs_subtitle/dvd_subtitle/dvb_subtitle -- burned-in bitmaps, common
# on Blu-ray/DVD rips): mov_text has no way to carry a picture, and asking
# FFmpeg to convert one to it fails outright rather than degrading, so
# those are left out of the video track instead of risking the whole
# encode over a subtitle that was never going to survive the conversion.
_TEXT_SUBTITLE_CODECS = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text"}


def _ffprobe_info(path, ffprobe_exe):
    """width/height/fps/frame_count/duration_ms/has_audio/has_alpha/
    sample_rate/channels/bit_rate/has_subtitles/tags/video_stream_index/
    audio_stream_indices/subtitle_stream_indices for a media file.
    sample_rate/channels/bit_rate describe the *first* audio stream
    (meaningful whenever has_audio is true -- video's own audio track
    for encode_video, or the whole file for encode_audio, which has no
    video stream at all); audio_stream_indices is every audio stream's
    own global index in the file (see encode_video's multi-track
    handling), not just the first. has_subtitles is true whenever
    subtitle_stream_indices is non-empty -- collected the same way, but
    only a text-based subtitle stream counts at all (see
    _TEXT_SUBTITLE_CODECS and encode_video's subtitle handling) -- an
    image-based one (PGS/DVD subs) doesn't, since there's nowhere for
    FMFF to put it. tags is whatever container-level tags (artist/album/
    title/date/genre/...) ffprobe reports, verbatim -- see
    _pack_metadata."""
    cmd = [ffprobe_exe, "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    # ffprobe always emits UTF-8 JSON regardless of the OS's own locale/
    # console codepage -- without encoding="utf-8" here, subprocess.run
    # falls back to locale.getpreferredencoding() (cp1252 on this project's
    # target Windows systems), which can't decode non-Latin tag metadata
    # (an artist/title/album tag in Cyrillic, say) embedded in that JSON,
    # crashing the reader thread with UnicodeDecodeError before the JSON
    # is even parsed. errors="replace" is a last-resort safety net, not
    # the primary fix -- it would silently mangle the exact metadata
    # bytes encoding="utf-8" alone already decodes correctly.
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=True).stdout
    data = json.loads(out)

    info = {"width": 0, "height": 0, "fps": 25.0, "duration_ms": 0,
            "frame_count": 0, "has_audio": False, "has_alpha": False,
            "sample_rate": 48000, "channels": 2, "bit_rate": 0,
            "has_subtitles": False, "video_stream_index": 0,
            "audio_stream_indices": [], "subtitle_stream_indices": []}
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and not info["width"]:
            info["width"] = int(s.get("width", 0))
            info["height"] = int(s.get("height", 0))
            info["video_stream_index"] = int(s.get("index", 0))
            fr = s.get("avg_frame_rate") or s.get("r_frame_rate") or "25/1"
            try:
                num, den = fr.split("/")
                info["fps"] = float(num) / float(den) if float(den) else 25.0
            except (ValueError, ZeroDivisionError):
                pass
            nb = s.get("nb_frames")
            if nb and str(nb).isdigit():
                info["frame_count"] = int(nb)
            info["has_alpha"] = s.get("pix_fmt") in _ALPHA_PIX_FMTS
        elif s.get("codec_type") == "audio":
            info["audio_stream_indices"].append(int(s.get("index", 0)))
            if not info["has_audio"]:
                info["has_audio"] = True
                info["sample_rate"] = int(s.get("sample_rate", 48000))
                info["channels"] = int(s.get("channels", 2))
                # Not every container puts bit_rate on the stream itself
                # (some only carry it at the format/container level) --
                # try the stream first since it's the more specific
                # number when present, fall back to the format-level
                # one otherwise.
                br = s.get("bit_rate") or data.get("format", {}).get("bit_rate")
                if br:
                    info["bit_rate"] = int(br)
        elif s.get("codec_type") == "subtitle" and s.get("codec_name") in _TEXT_SUBTITLE_CODECS:
            info["subtitle_stream_indices"].append(int(s.get("index", 0)))
    info["has_subtitles"] = bool(info["subtitle_stream_indices"])

    duration = data.get("format", {}).get("duration")
    if duration:
        info["duration_ms"] = int(float(duration) * 1000)
    if not info["frame_count"] and info["duration_ms"]:
        info["frame_count"] = int(info["duration_ms"] / 1000.0 * info["fps"])
    info["tags"] = {str(k): str(v) for k, v in data.get("format", {}).get("tags", {}).items()}
    return info


def _split_mp4_fragments(data):
    """Split a fragmented-MP4 byte string (ftyp+moov, then any number of
    moof+mdat pairs -- what ffmpeg produces with `-movflags
    frag_keyframe+empty_moov+default_base_moof`) into (init_bytes,
    [segment_bytes, ...]), one segment per keyframe-aligned moof+mdat
    pair. init_bytes plus any prefix of the segments, concatenated in
    order, is itself a valid, playable (if shorter) fragmented MP4 --
    the same "independently appendable after an init segment" property
    MSE/DASH streaming relies on -- which is what lets a corrupt or
    missing segment be dropped at decode time instead of failing
    playback of the whole video (see extract_media). Any trailing box
    after the last mdat (ffmpeg appends an optional `mfra` random-access
    index) is discarded -- it's just a seek hint, not needed to play."""
    n = len(data)
    boxes = []
    pos = 0
    while pos + 8 <= n:
        size = struct.unpack(">I", data[pos:pos + 4])[0]
        typ = data[pos + 4:pos + 8]
        if size == 1:
            if pos + 16 > n:
                break
            size = struct.unpack(">Q", data[pos + 8:pos + 16])[0]
        elif size == 0:
            size = n - pos
        if size < 8 or pos + size > n:
            break
        boxes.append((pos, size, typ))
        pos += size

    i = 0
    init_end = 0
    while i < len(boxes) and boxes[i][2] in (b"ftyp", b"moov"):
        init_end = boxes[i][0] + boxes[i][1]
        i += 1
    init_bytes = data[:init_end]

    segments = []
    while i < len(boxes):
        if boxes[i][2] == b"moof":
            start, length = boxes[i][0], boxes[i][1]
            j = i + 1
            if j < len(boxes) and boxes[j][2] == b"mdat":
                length += boxes[j][1]
                j += 1
            segments.append(data[start:start + length])
            i = j
        else:
            i += 1  # skip anything else (e.g. trailing mfra)
    return init_bytes, segments


def _encode_fragmented_av1(ffmpeg, input_path, blob_path, crf, speed, gop, fps,
                            vf, progress_cb, cancel_event, codec="libsvtav1",
                            video_index=0, audio_indices=(), subtitle_indices=()):
    """Run one FFmpeg pass producing fragmented-MP4 AV1 (+ Opus for every
    index in audio_indices, + a mov_text track for every index in
    subtitle_indices) output at blob_path, then split it into
    (init_bytes, segments, segment_blobs) via _split_mp4_fragments --
    shared by encode_video's color pass and its optional alpha pass
    (vf="alphaextract"), which are otherwise identical except for `codec`
    (see encode_video's alpha handling for why the alpha pass uses
    libaom-av1 instead of the color pass's libsvtav1: SVT-AV1, at least
    in this FFmpeg build, produces a bitstream libdav1d can't decode for
    alphaextract's single-plane grayscale input -- confirmed with plain,
    non-fragmented output too, so it's an SVT-AV1/content-shape issue,
    not anything about fragmentation; libaom-av1 handles the exact same
    input cleanly). audio_indices/subtitle_indices are only ever passed
    for the color pass -- audio/subtitles have nothing to do with the
    alpha pass's separate grayscale-only stream.

    Every index here is the stream's own global index in the source file
    (what ffprobe's own "index" field reports -- see _ffprobe_info),
    mapped explicitly (`-map 0:{index}`) rather than left to FFmpeg's
    default "best stream of each type" auto-selection, so *every* audio
    track and every text-based subtitle track survives, not just one of
    each -- language tags included, since FFmpeg carries a mapped
    stream's own metadata over automatically. video_index must still be
    mapped explicitly too once any other explicit -map is used at all:
    FFmpeg's default stream auto-selection is all-or-nothing, so mapping
    only audio/subtitles explicitly would silently drop the video track
    entirely (confirmed directly: `-map 0:1` alone, no video map, produced
    an audio-only output file).

    progress_cb, if given, is called with just the current frame number
    (the caller already knows the total -- see encode_video's
    make_progress). cancel_event/EncodingCancelled work exactly as
    encode_video documents."""
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-progress", "pipe:1", "-i", str(input_path),
           "-map", f"0:{video_index}"]
    for i in audio_indices:
        cmd += ["-map", f"0:{i}"]
    for i in subtitle_indices:
        cmd += ["-map", f"0:{i}"]
    if fps:
        cmd += ["-r", str(fps)]
    if vf:
        cmd += ["-vf", vf]
    if codec == "libaom-av1":
        cmd += ["-c:v", "libaom-av1", "-crf", str(crf), "-cpu-used", str(min(8, speed)),
                "-pix_fmt", "yuv420p", "-g", str(gop)]
    else:
        cmd += ["-c:v", "libsvtav1", "-crf", str(crf), "-preset", str(speed),
                "-pix_fmt", "yuv420p", "-g", str(gop)]
    # Signal color space explicitly rather than leaving it unset. A source
    # with no color tags of its own (common -- plenty of consumer video
    # simply doesn't carry them) leaves it to whichever decoder plays the
    # *output* to guess, and different decoders' guesses for "unspecified"
    # don't always agree -- confirmed by direct A/B testing: the same
    # untagged AV1 output showed a visible pink/magenta tint in dark
    # scenes on one decode path but not another. Tagging bt709 (the
    # standard for consumer HD video, which covers the overwhelming
    # majority of what gets encoded here) removes the ambiguity outright.
    cmd += ["-color_primaries", "bt709", "-color_trc", "bt709",
            "-colorspace", "bt709", "-color_range", "tv"]
    if audio_indices:
        cmd += ["-c:a", "libopus", "-b:a", "128k"]
    else:
        cmd += ["-an"]
    if subtitle_indices:
        # mov_text is MP4's own timed-text subtitle format -- FFmpeg
        # converts each compatible text-based source subtitle (SRT/ASS/
        # WebVTT/already-mov_text -- see _TEXT_SUBTITLE_CODECS, checked
        # by the caller before any index ever lands in subtitle_indices)
        # into it automatically, applying to every mapped subtitle
        # stream the same way -c:a applies to every mapped audio one.
        cmd += ["-c:s", "mov_text"]
    # Fragmented MP4 -- the same mechanism browsers use for MSE/DASH
    # streaming -- so the output naturally splits into a small init
    # chunk and a run of keyframe-aligned fragments instead of one
    # undifferentiated blob (see _split_mp4_fragments and the module
    # docstring).
    cmd += ["-movflags", "frag_keyframe+empty_moov+default_base_moof",
            "-frag_duration", str(int(VIDEO_SEGMENT_SECONDS * 1_000_000))]
    cmd += [blob_path]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=True, encoding="utf-8", errors="replace", bufsize=1)
    cancelled = False
    for line in proc.stdout:
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
        line = line.strip()
        if line.startswith("frame=") and progress_cb:
            try:
                progress_cb(int(line.split("=", 1)[1]))
            except ValueError:
                pass
    if cancelled:
        raise EncodingCancelled("video encoding was cancelled")
    ret = proc.wait()
    if ret != 0 or not os.path.exists(blob_path):
        raise RuntimeError(f"ffmpeg failed to encode video (exit code {ret})")
    blob = open(blob_path, "rb").read()
    init_bytes, segment_blobs = _split_mp4_fragments(blob)
    segments = [(len(s), zlib.crc32(s) & 0xFFFFFFFF) for s in segment_blobs]
    return init_bytes, segments, segment_blobs


def _encode_fragmented_audio(ffmpeg, input_path, blob_path, bitrate, progress_cb, cancel_event):
    """Audio counterpart to _encode_fragmented_av1: one FFmpeg pass
    producing fragmented-MP4, Opus-only output at blob_path, split into
    (init_bytes, segments, segment_blobs) the same way -- see
    encode_audio for why this reuses video's whole segmented-media-blob
    mechanism instead of being a new one. `-vn` drops any attached
    picture (cover art) some audio containers carry as a "video" stream,
    which would otherwise make FFmpeg try and fail to run it through
    libopus alongside the real audio -- see encode_audio for where that
    picture actually goes instead. progress_cb, if given, is called with
    just the elapsed encode position in ms: unlike video there's no
    frame count to report a total against, and ffmpeg's own frame=
    progress field (what _encode_fragmented_av1 reads) stays at 0 for an
    audio-only encode, so out_time_ms is read instead. cancel_event/
    EncodingCancelled work exactly as encode_video documents."""
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-progress", "pipe:1", "-i", str(input_path),
           "-vn", "-c:a", "libopus", "-b:a", bitrate,
           "-movflags", "frag_keyframe+empty_moov+default_base_moof",
           "-frag_duration", str(int(VIDEO_SEGMENT_SECONDS * 1_000_000)),
           blob_path]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=True, encoding="utf-8", errors="replace", bufsize=1)
    cancelled = False
    for line in proc.stdout:
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
        line = line.strip()
        if line.startswith("out_time_ms=") and progress_cb:
            try:
                progress_cb(int(line.split("=", 1)[1]) // 1000)
            except ValueError:
                pass
    if cancelled:
        raise EncodingCancelled("audio encoding was cancelled")
    ret = proc.wait()
    if ret != 0 or not os.path.exists(blob_path):
        raise RuntimeError(f"ffmpeg failed to encode audio (exit code {ret})")
    blob = open(blob_path, "rb").read()
    init_bytes, segment_blobs = _split_mp4_fragments(blob)
    segments = [(len(s), zlib.crc32(s) & 0xFFFFFFFF) for s in segment_blobs]
    return init_bytes, segments, segment_blobs


def _open_video_capture(path):
    """Open a video for *reading*, preferring Windows Media Foundation with
    hardware-accelerated decode (DXVA/D3D11 -- routed to whatever the GPU's
    fixed-function media engine supports, e.g. Intel Quick Sync on an Iris
    chip) instead of OpenCV's default software decoder. This only speeds up
    reading the source video's standard codec (H.264/HEVC/etc): the FMFF
    codec itself has no hardware equivalent to run on, since it isn't a
    standard format any GPU media engine knows about, so encoding/decoding
    FMFF tiles stays on the CPU regardless. Falls back to the default
    backend if MSMF can't open the file (e.g. an unsupported container, or
    on a non-Windows OS)."""
    params = [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY]
    cap = cv2.VideoCapture(str(path), cv2.CAP_MSMF, params)
    if cap.isOpened():
        return cap
    cap.release()
    return cv2.VideoCapture(str(path))


if cv2 is not None:
    _HW_ACCEL_NAMES = {
        cv2.VIDEO_ACCELERATION_NONE: "none (software)",
        cv2.VIDEO_ACCELERATION_ANY: "generic hardware",
        cv2.VIDEO_ACCELERATION_D3D11: "D3D11 (GPU media engine, e.g. Intel Quick Sync)",
        cv2.VIDEO_ACCELERATION_VAAPI: "VAAPI",
        cv2.VIDEO_ACCELERATION_MFX: "Intel Media SDK (Quick Sync)",
    }
else:
    _HW_ACCEL_NAMES = {}


# ------------------------------------------------------------------- entries

class CorruptTileError(Exception):
    """Raised for one tile whose bytes don't match its stored CRC32 (bit
    rot, or a truncated/still-downloading file) or that fail to decompress
    even so. Callers that want error resilience (full(), thumbnail()) catch
    this per-tile and substitute a placeholder instead of aborting the
    whole image -- a corrupt/missing tile stays local, it doesn't take
    down decoding of everything else in the file."""

    def __init__(self, entry):
        self.entry = entry
        super().__init__(f"corrupt tile: frame={entry.frame} layer={entry.layer} "
                          f"tx={entry.tx} ty={entry.ty} plane={entry.plane}")


def _broken_tile_fill(h, w, channels):
    """A visually obvious placeholder for a tile that failed its CRC check
    -- flat mid-gray, distinct from any real photo content, so corruption
    is visible rather than silently papered over with a guess."""
    return np.full((h, w, channels) if channels > 1 else (h, w), 128, dtype=np.uint8)


class JpegPassthroughUnsupported(Exception):
    """Raised when encode_image_jpeg_passthrough can't handle a particular
    JPEG (progressive scan, not 3-component, or jpeglib missing) -- callers
    catch this and fall back to the normal tile codec instead."""


class EncodingCancelled(Exception):
    """Raised by encode_video when a caller-supplied cancel_event was set
    mid-encode -- a deliberate stop, distinct from a real ffmpeg failure,
    so callers (e.g. BatchConvertWindow) can tell the two apart."""


_JPEG_COEFF_CODECS = {b"Z": (lambda b: zlib.compress(b, _ZLIB_LEVEL), zlib.decompress),
                      b"B": (lambda b: bz2.compress(b, 9), bz2.decompress),
                      b"L": (lambda b: lzma.compress(b, preset=1), lzma.decompress)}


def _dc_delta_encode(comp):
    """DC coefficients (position 0 of every 8x8 block) are strongly
    correlated between spatially adjacent blocks -- neighboring patches of
    a photo usually have similar average brightness -- so replacing each
    block's DC with (DC - previous block's DC), in the same raster
    block-scan order JPEG itself predicts DC from, gives a general-purpose
    compressor a mostly-small/zero-centered stream instead of the raw
    values. Tested on real photos: a free ~3-9% smaller than compressing
    the raw coefficients directly, in every case tried, never worse."""
    nby, nbx = comp.shape[:2]
    flat = comp.reshape(nby * nbx, 64).astype(np.int16).copy()
    dc = flat[:, 0].astype(np.int32)
    delta = np.empty_like(dc)
    delta[0] = dc[0]
    delta[1:] = dc[1:] - dc[:-1]
    flat[:, 0] = delta.astype(np.int16)
    return flat.reshape(nby, nbx, 8, 8)


def _dc_delta_decode(coeffs):
    """Inverse of _dc_delta_encode: cumulative sum undoes the DC delta."""
    nby, nbx = coeffs.shape[:2]
    flat = coeffs.reshape(nby * nbx, 64).astype(np.int32)
    flat[:, 0] = np.cumsum(flat[:, 0])
    return flat.reshape(nby, nbx, 8, 8)


class Entry:
    __slots__ = ("frame", "layer", "tx", "ty", "plane", "mode", "w", "h",
                 "offset", "length", "crc32", "payload")

    def __init__(self, frame, layer, tx, ty, plane, mode, w, h, payload=b""):
        self.frame = frame
        self.layer, self.tx, self.ty = layer, tx, ty
        self.plane, self.mode = plane, mode
        self.w, self.h = w, h
        self.payload = payload
        self.offset = 0
        self.length = len(payload)
        self.crc32 = zlib.crc32(payload) & 0xFFFFFFFF

    def pack(self):
        return struct.pack(ENTRY_FMT, self.frame, self.layer, self.tx, self.ty, self.plane,
                            self.mode, self.w, self.h, self.offset, self.length, self.crc32)

    @classmethod
    def unpack(cls, data):
        frame, layer, tx, ty, plane, mode, w, h, offset, length, crc32 = struct.unpack(ENTRY_FMT, data)
        e = cls(frame, layer, tx, ty, plane, mode, w, h)
        e.offset, e.length, e.crc32 = offset, length, crc32
        return e


FrameEntry = namedtuple("FrameEntry", "timestamp_ms entry_start entry_count")


def _race_color_candidates(rgb_tile, quality):
    """Encode one RGB tile (or thumbnail) three ways -- PNG-style lossless
    filter+zlib, DCT lossy, and palette/indexed-color (see _palette_encode)
    -- and keep whichever comes out smallest. Shared by _encode_tile_task
    and _make_thumbnail_entries so both get the same three-way race."""
    lossless_bytes = lossless_encode(rgb_tile.transpose(2, 0, 1))
    lossy_bytes = lossy_encode(rgb_tile, quality)
    candidates = [(MODE_LOSSLESS, lossless_bytes), (MODE_LOSSY, lossy_bytes)]
    palette_bytes = _palette_encode(rgb_tile)
    if palette_bytes is not None:
        candidates.append((MODE_PALETTE, palette_bytes))
    return min(candidates, key=lambda kv: len(kv[1]))


def _now_iso():
    """UTC timestamp for a version's "added" metadata field (see
    FMFFEncoder.add_version) -- time.gmtime() alone is enough for this,
    no extra dependency (datetime) needed for one timestamp string."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _pack_metadata(tags=None, exif_bytes=None, icc_bytes=None, versions=None):
    """The optional trailing metadata blob every encode_* method can
    attach (see HEADER_FMT's metadata_offset/metadata_length comment):
    a small JSON envelope carrying source tags (artist/album/title/...,
    from a container's own tags -- audio and video), a raw EXIF blob
    (a still image, byte-for-byte, not re-derived -- same passthrough
    philosophy as JPEG's own DCT-coefficient path), a raw ICC color
    profile (also byte-for-byte -- a profile encodes a specific
    color-managed workflow's calibration, not something to regenerate
    from scratch), and/or a still image's version history: a list of
    {"index", "name", "note", "added", "size"} dicts, one per LAYER_FULL/
    LAYER_VERSION version stored in the file (see FMFFEncoder.add_version
    and FMFFDecoder.list_versions) -- everything but the pixels a version
    needs to be found and labeled, since that's already carried by the
    LAYER_VERSION tile entries themselves. Both binary blobs are base64'd
    inside the JSON rather than needing their own offset/length pairs; an
    ICC profile is at most a few KB, same order of size as EXIF, so that
    overhead stays negligible. Returns b"" when there's nothing to store
    at all, which callers write as a zero-length section (metadata_offset
    is still recorded, just with length 0)."""
    meta = {}
    if tags:
        meta["tags"] = tags
    if exif_bytes:
        meta["exif_b64"] = base64.b64encode(exif_bytes).decode("ascii")
    if icc_bytes:
        meta["icc_b64"] = base64.b64encode(icc_bytes).decode("ascii")
    if versions:
        meta["versions"] = versions
    if not meta:
        return b""
    return json.dumps(meta).encode("utf-8")


def _unpack_metadata(blob):
    """Inverse of _pack_metadata -- always returns (tags_dict, exif_bytes,
    icc_bytes, versions_list), ({}, None, None, []) for an empty/absent
    blob rather than raising, so callers never need to special-case "this
    file predates metadata" beyond that (which every file before VERSION
    13 does, every file before whichever version added icc_b64 for the
    ICC profile specifically, and every file before whichever version
    added "versions" for version history specifically)."""
    if not blob:
        return {}, None, None, []
    try:
        meta = json.loads(blob.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}, None, None, []
    tags = meta.get("tags") or {}
    exif_bytes = base64.b64decode(meta["exif_b64"]) if "exif_b64" in meta else None
    icc_bytes = base64.b64decode(meta["icc_b64"]) if "icc_b64" in meta else None
    versions = meta.get("versions") or []
    return tags, exif_bytes, icc_bytes, versions


def _ffmpeg_metadata_args(tags):
    """-metadata key=value FFmpeg args for restoring a decoded video/
    audio file's tags on a decode/Save-as remux -- see _pack_metadata."""
    args = []
    for k, v in (tags or {}).items():
        args += ["-metadata", f"{k}={v}"]
    return args


def _make_thumbnail_entries(img, thumb_max, quality):
    """Build the small instant-preview thumbnail entries (color, plus alpha
    if img has one). Two things that matter for a *small* source image
    specifically: the thumbnail is capped well below thumb_max when the
    source itself is small (otherwise a "quick preview" can rival or beat
    the size of the full tile-coded image, which defeats the point of it
    being a cheap preview), and it gets the same hybrid lossless/lossy
    choice as a regular tile instead of always being lossless (a detailed/
    noisy thumbnail forced lossless is expensive for exactly the content
    that benefits least from a pixel-exact preview)."""
    width, height = img.size
    thumb_cap = max(1, min(thumb_max, max(width, height) // 3))
    thumb = img.copy()
    thumb.thumbnail((thumb_cap, thumb_cap), Image.LANCZOS)
    tw, th = thumb.size
    tarr = np.array(thumb)
    rgb = tarr[:, :, :3] if tarr.ndim == 3 else np.dstack([tarr] * 3)

    mode, payload = _race_color_candidates(rgb, quality)
    entries = [Entry(0, LAYER_THUMB, 0, 0, PLANE_COLOR, mode, tw, th, payload)]

    if tarr.ndim == 3 and tarr.shape[2] == 4:
        entries.append(Entry(0, LAYER_THUMB, 0, 0, PLANE_ALPHA, MODE_LOSSLESS, tw, th,
                              lossless_encode(tarr[:, :, 3][None, :, :])))
    return tw, th, entries


# --------------------------------------------------------- multi-core tiling
# Profiling a real encode (3000x2000, 1504 tiles) showed the DCT math is a
# minority of the cost: zlib.compress alone is ~60% of wall time, bz2
# another ~10% -- both are serial, per-tile, CPU-bound byte-level
# algorithms with no GPU equivalent. What they ARE is embarrassingly
# parallel *across* tiles (each tile's lossless-vs-lossy race is fully
# independent), so multiple CPU cores working on different tiles at once
# is the change that actually attacks where the time goes, rather than
# GPU-accelerating just the ~25% that was DCT.

_MP_TILE_THRESHOLD = 64  # below this, process-pool startup overhead isn't worth it
_TILE_POOL = None


def _encode_tile_task(args):
    """One tile's worth of work: race lossless vs lossy, keep the smaller,
    plus the alpha plane if present. Runs identically whether called
    directly (small images) or inside a worker process via the tile pool
    (large images) -- see encode_image. `frame` is just carried through
    unchanged (0 for a plain still image, the frame index for an
    animated one -- see encode_image_sequence) so results can be matched
    back up to the right frame/tile after coming back from the pool in
    whatever order they finish."""
    frame, tx, ty, tile_rgb, alpha_tile, quality = args
    mode, payload = _race_color_candidates(tile_rgb, quality)
    alpha_payload = (lossless_encode(alpha_tile[None, :, :])
                      if alpha_tile is not None else None)
    return frame, tx, ty, mode, payload, alpha_payload


def _get_tile_pool():
    """A lazily-created, process-lifetime multiprocessing.Pool, reused
    across every large-image encode (not just within one call) -- so
    batch-converting many large images only pays worker-startup cost
    once, not per file. Worker count is capped at 8: most of these tiles'
    work is memory-bandwidth-bound (byte-level compression), so beyond
    somewhere around 8 cores the OS scheduling/IPC overhead tends to eat
    the gains rather than add more."""
    global _TILE_POOL
    if _TILE_POOL is None:
        workers = max(1, min(8, os.cpu_count() or 1))
        _TILE_POOL = mp.Pool(processes=workers)
        atexit.register(_shutdown_tile_pool)
    return _TILE_POOL


def _shutdown_tile_pool():
    global _TILE_POOL
    if _TILE_POOL is not None:
        _TILE_POOL.terminate()
        _TILE_POOL = None


def _decode_tile_bytes(data, entry, quality):
    """The actual dequant/IDCT/lossless dispatch for one tile's already-
    read-and-CRC-checked bytes -- shared by FMFFDecoder._decode_entry
    (single-process) and _decode_tile_task (the pooled-decode worker
    below) so both go through exactly the same logic regardless of which
    process ends up running it."""
    if entry.plane in (PLANE_ALPHA, PLANE_MASK):
        return lossless_decode(data, 1, entry.h, entry.w)[0]
    if entry.mode == MODE_LOSSY:
        return lossy_decode(data, entry.h, entry.w, quality)
    if entry.mode == MODE_PALETTE:
        return _palette_decode(data, entry.h, entry.w)
    return lossless_decode(data, 3, entry.h, entry.w).transpose(1, 2, 0)


def _decode_tile_task(args):
    """One tile's worth of decode work: read its bytes directly from the
    file at its own (offset, length), then decode them. Runs identically
    whether called directly (small images) or inside a worker process
    via the tile pool (large images), mirroring _encode_tile_task's own
    directly-callable-or-pooled design -- decoding was single-process-
    only until profiling a real ~12MP photo showed full() taking 1.6s+
    entirely on one core while encode_image's own tile race already had
    a multi-core pool sitting right there; spreading independent tiles'
    decode across it the same way cut that to well under half a second
    once the pool is warm (see FMFFDecoder.full()).

    Never raises: a corrupt/short read or a decode failure both come
    back as (entry, None) rather than an exception, since a single bad
    tile crossing a multiprocessing.Pool.map() call as an exception
    would abort the *whole* batch instead of degrading just that one
    tile -- the caller (full()) substitutes the usual gray placeholder
    and counts it exactly like CorruptTileError already does for the
    single-process path (FMFFDecoder._decode_entry)."""
    path, entry, quality = args
    try:
        with open(path, "rb") as f:
            f.seek(entry.offset)
            data = f.read(entry.length)
        if len(data) != entry.length or (zlib.crc32(data) & 0xFFFFFFFF) != entry.crc32:
            return entry, None
        return entry, _decode_tile_bytes(data, entry, quality)
    except (zlib.error, OSError, ValueError):
        return entry, None


# ---------------------------------------------------- multi-region diffing
# encode_image_sequence's per-frame diff (see its own docstring) started out
# as a single tight bounding box around every changed pixel. That's optimal
# when the change is one blob, but falls apart on something like a logo
# whose glint sweeps through several corners of an otherwise-static frame:
# each frame changes a handful of small, far-apart spots, and one bbox
# around all of them re-stores most of the untouched background sitting
# between them for nothing -- on a real case shaped like that, the single
# bbox covered most of the frame every frame even though the actual changed
# pixel count was a small fraction of it. _split_changed_regions replaces
# that one bbox with however many tight, non-overlapping rectangles the
# change actually needs -- encode_image_sequence already stores an
# arbitrary number of LAYER_FULL entries per frame (color + optional alpha,
# see Entry), so giving it more than one changed rectangle for the same
# frame needed no format or decoder change at all: full_sequence() already
# blits every entry belonging to a frame onto the running canvas in
# sequence, whether that's one rectangle or several.
_REGION_BLOCK = 16  # coarse grid for finding separate regions; unrelated to any tile grid
_REGION_MERGE_GAP = 32  # coalesce two regions whose boxes are within this many px of each other
_REGION_MAX_COUNT = 12  # more separate regions than this and per-entry overhead stops paying off


def _split_changed_regions(changed):
    """Split one frame's changed-pixel mask (color or alpha differs from
    the previous frame -- see encode_image_sequence) into one or more
    tight (y0, y1, x0, x1) rectangles instead of a single bounding box
    around every changed pixel, so spots that change in different parts
    of the frame don't force each other's surrounding untouched
    background to be re-stored too (see this section's own comment).

    Connected-component search runs on a coarse _REGION_BLOCK-px grid
    (a block counts as "changed" if any pixel inside it is) rather than
    a per-pixel flood fill, so its cost tracks frame area / block^2 --
    trivial even at 4K -- not the number of changed pixels, which can be
    most of the frame on an ordinary scene change. Blocks touching
    (8-connectivity) are already one component from that grid alone;
    _REGION_MERGE_GAP additionally coalesces components whose *pixel*
    gap is still small, since two nearby small rectangles each pay their
    own ~27-byte index entry + CRC32 + per-stream compression overhead
    (see encode_image_sequence's docstring) where one slightly larger
    rectangle covering both would pay it once. If that still leaves
    more than _REGION_MAX_COUNT components -- a change spread broadly
    enough that "a few distinct regions" isn't the right shape for it
    any more -- this falls back to the original single whole-bbox
    behavior rather than paying for a pile of small entries.

    Returns [] if `changed` is all-False (caller already checks this
    before calling, via changed.any(), but this stays correct standalone
    too)."""
    ys, xs = np.nonzero(changed)
    if len(xs) == 0:
        return []
    full_y0, full_y1 = int(ys.min()), int(ys.max()) + 1
    full_x0, full_x1 = int(xs.min()), int(xs.max()) + 1
    whole_bbox = [(full_y0, full_y1, full_x0, full_x1)]

    bs = _REGION_BLOCK
    sub = changed[full_y0:full_y1, full_x0:full_x1]
    sh, sw = sub.shape
    pad_h, pad_w = (-sh) % bs, (-sw) % bs
    padded = np.pad(sub, ((0, pad_h), (0, pad_w))) if (pad_h or pad_w) else sub
    grid_h, grid_w = padded.shape[0] // bs, padded.shape[1] // bs
    if grid_h * grid_w <= 1:
        return whole_bbox
    blocks = padded.reshape(grid_h, bs, grid_w, bs).any(axis=(1, 3))

    # 8-connected labeling over the (small) block grid via BFS -- cheap
    # regardless of image resolution since it's bounded by grid_h*grid_w,
    # not by pixel count.
    labels = np.full((grid_h, grid_w), -1, dtype=np.int32)
    components = []
    for by0 in range(grid_h):
        for bx0 in range(grid_w):
            if not blocks[by0, bx0] or labels[by0, bx0] != -1:
                continue
            label = len(components)
            labels[by0, bx0] = label
            stack = [(by0, bx0)]
            cells = []
            while stack:
                by, bx = stack.pop()
                cells.append((by, bx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = by + dy, bx + dx
                        if (dy or dx) and 0 <= ny < grid_h and 0 <= nx < grid_w \
                                and blocks[ny, nx] and labels[ny, nx] == -1:
                            labels[ny, nx] = label
                            stack.append((ny, nx))
            components.append(cells)

    if len(components) <= 1:
        return whole_bbox

    # Each component's block-grid bbox, translated to absolute pixel
    # coordinates and clipped to the overall changed-pixel bbox (a block
    # can be only partly changed, or padding-only past the real edge, so
    # its block-grid footprint can overshoot the real content), then
    # tightened to the actual changed pixels inside it.
    boxes = []
    for cells in components:
        bys = [c[0] for c in cells]
        bxs = [c[1] for c in cells]
        py0 = full_y0 + min(bys) * bs
        py1 = min(full_y1, full_y0 + (max(bys) + 1) * bs)
        px0 = full_x0 + min(bxs) * bs
        px1 = min(full_x1, full_x0 + (max(bxs) + 1) * bs)
        ryy, rxx = np.nonzero(changed[py0:py1, px0:px1])
        boxes.append((py0 + int(ryy.min()), py0 + int(ryy.max()) + 1,
                      px0 + int(rxx.min()), px0 + int(rxx.max()) + 1))

    # A pathological all-noise diff can produce many isolated one-block
    # components; bail before the O(n^2) merge scan below rather than
    # let it run on hundreds of them for no benefit (this many regions
    # falls back to the whole bbox anyway, see the _REGION_MAX_COUNT
    # check after merging).
    if len(boxes) > 8 * _REGION_MAX_COUNT:
        return whole_bbox

    merged = True
    while merged and len(boxes) > 1:
        merged = False
        for i in range(len(boxes)):
            y0a, y1a, x0a, x1a = boxes[i]
            for j in range(i + 1, len(boxes)):
                y0b, y1b, x0b, x1b = boxes[j]
                gap_y = max(y0a, y0b) - min(y1a, y1b)
                gap_x = max(x0a, x0b) - min(x1a, x1b)
                if gap_y <= _REGION_MERGE_GAP and gap_x <= _REGION_MERGE_GAP:
                    boxes[i] = (min(y0a, y0b), max(y1a, y1b), min(x0a, x0b), max(x1a, x1b))
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break

    if len(boxes) > _REGION_MAX_COUNT:
        return whole_bbox
    return boxes


# ------------------------------------------------------------ document pages
# PDF/.txt aren't media in the sense anything else in this file is -- text
# and layout, not pixels/frames/samples. An earlier version of this
# rasterized every page to a picture and stored the result as an ordinary
# multi-frame CONTENT_IMAGE .fmff, the same container an animated GIF uses.
# That traded away everything that makes a document a document -- no
# selectable/searchable text, no hyperlinks, no forms/formulas -- and, on
# top of throwing all of that away, routinely came out *bigger* than the
# source: a rendered page of mostly-flat content still costs more bits than
# the already-compressed text/vector data it was rendered from, even with
# anti-aliasing off and the same three-way lossless/palette/lossy race
# every image tile gets. A worse result that's also less useful is a bad
# trade with nothing to recommend it, so this now stores the document
# itself (CONTENT_DOCUMENT -- see encode_document), the same passthrough
# philosophy as an already-lossy JPEG (see encode_image_jpeg_passthrough):
# the original bytes, raced against a general-purpose compressor and kept
# only if that's actually smaller, so this can never be bigger than the
# source by more than the container's own small fixed overhead. Decode
# gets back the exact original file, byte-for-byte -- text, links, forms,
# and formulas all keep working in a real PDF/text viewer, because that's
# what's actually stored.
#
# Page rendering doesn't disappear, it just moves to being a *preview*
# instead of the storage format: a single first-page thumbnail is stored
# for an instant preview (see _render_document_thumbnail), and FMFF's own
# viewer re-renders every page on demand straight from the recovered
# original bytes when actually browsing one (see
# MediaViewer._load_fmff_document) -- the exact same rendering path
# (_render_document_pages) a plain, not-yet-converted PDF/.txt already
# uses, so nothing about page rendering is duplicated for the .fmff case.
#
# Rendering (for the thumbnail and for on-demand browsing) is deliberately
# scoped to formats a lightweight, no-external-app dependency can
# rasterize: PyMuPDF for PDF (a real C library, `pip install pymupdf`, no
# separate program to install), Pillow's own text/font rendering for .txt.
# Office formats (.docx/.xlsx/...) would need an external renderer
# (LibreOffice, run headless) -- a much heavier dependency (a whole desktop
# application, not a pip package) deliberately left out of this pass. Note
# that rendering is only ever a preview now, not a requirement for storage
# or extraction: extract_document needs no rendering dependency at all (it
# just returns the stored bytes), and PyMuPDF/a font is only ever needed
# to produce a thumbnail/preview, never to keep the original document's
# data intact.

DOCUMENT_EXTS = {".pdf", ".txt"}
DOCUMENT_RENDER_DPI = 150
_DOCUMENT_PAGE_SIZE = (1240, 1754)  # ~A4 at DOCUMENT_RENDER_DPI, for a rendered .txt page
_DOCUMENT_MARGIN = 70
_DOCUMENT_FONT_SIZE = 22
# Tried in order; the first one this system actually has wins. Monospace
# specifically so a fixed character count reliably fits one line width,
# no per-character-width text-shaping needed to line-wrap .txt content.
_MONOSPACE_FONT_CANDIDATES = [
    "consola.ttf", "Consolas.ttf",  # Windows
    "DejaVuSansMono.ttf",  # common on Linux
    "Menlo.ttc", "/System/Library/Fonts/Menlo.ttc",  # macOS
    "cour.ttf", "Courier New.ttf",
]


def _load_monospace_font(size):
    for name in _MONOSPACE_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    # No TrueType font found anywhere on this system -- Pillow's own
    # built-in fallback. load_default(size=...) is a scalable font added
    # in a fairly recent Pillow; older installs only have the tiny fixed
    # bitmap one load_default() gives with no arguments.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _render_pdf_pages(input_path):
    """One PIL Image per page of a PDF, rasterized via PyMuPDF at
    DOCUMENT_RENDER_DPI. See this section's docstring for what this
    deliberately doesn't preserve (text, links, forms, ...).

    Anti-aliasing is turned off first (`TOOLS.set_aa_level(0)`, a global
    MuPDF setting): a typical text-heavy page anti-aliased the normal way
    blends every glyph edge through dozens of intermediate gray shades
    (215 distinct colors measured on a few words of rendered text alone),
    which is real per-pixel entropy no codec here compresses away for
    free -- measured on one real text-heavy page, that alone was a 3.2x
    size difference (13,290 vs 4,142 bytes) for the exact same page,
    fully anti-aliasing on versus off. Off means every pixel is either
    the glyph color or the background, nothing between -- a real,
    visible trade (jaggier edges, particularly on any actual line art in
    the source, not just text) accepted deliberately here because a
    compact preview is the entire point of rasterizing in the first
    place, and this trade is what takes a real 200%+-bigger-than-source
    inflation this feature originally shipped with down to something
    much more reasonable."""
    if pymupdf is None:
        raise RuntimeError(
            "PDF encoding needs `pip install pymupdf` -- pages are rasterized "
            "to images (see the document-pages section of this file for why), "
            "so that's the only real dependency it needs")
    pymupdf.TOOLS.set_aa_level(0)
    doc = pymupdf.open(str(input_path))
    try:
        pages = []
        for page in doc:
            pix = page.get_pixmap(dpi=DOCUMENT_RENDER_DPI)
            mode = "RGBA" if pix.alpha else "RGB"
            pages.append(Image.frombytes(mode, (pix.width, pix.height), pix.samples))
        if not pages:
            raise ValueError(f"{input_path} has no pages")
        return pages
    finally:
        doc.close()


def _render_text_pages(input_path):
    """One PIL Image per page of a plain-text file, word-wrapped and
    paginated onto flat white pages -- Pillow/a system font only, no
    external dependency at all. Monospace so the character count that
    fits one line width can just be computed from one glyph's width,
    rather than needing real per-character text shaping.

    Rendered onto a "1" (1-bit) mode image, not "RGB" directly: Pillow
    anti-aliases text drawn onto an RGB image, blending every glyph edge
    through dozens of intermediate gray shades (measured 215 distinct
    colors from a few words of rendered text alone) -- real per-pixel
    entropy along every letter, in exchange for smoother-looking edges
    nothing about a monospace document preview needs. A "1" mode image
    only has two possible pixel values at all, so PIL can't anti-alias
    onto it -- every pixel is cleanly text-color or background, which
    the lossless/palette race (see _race_color_candidates) handles far
    better: measured on one full page of realistic text, disabling
    anti-aliasing this way cut the encoded page to less than half."""
    text = Path(input_path).read_text(encoding="utf-8", errors="replace")
    font = _load_monospace_font(_DOCUMENT_FONT_SIZE)
    page_w, page_h = _DOCUMENT_PAGE_SIZE
    usable_w = page_w - 2 * _DOCUMENT_MARGIN
    usable_h = page_h - 2 * _DOCUMENT_MARGIN

    char_w = font.getlength("M") or (_DOCUMENT_FONT_SIZE * 0.6)
    chars_per_line = max(10, int(usable_w / char_w))
    top, bottom = font.getbbox("Mg")[1], font.getbbox("Mg")[3]
    line_h = max(1, int((bottom - top) * 1.4))
    lines_per_page = max(1, int(usable_h / line_h))

    wrapped = []
    for raw_line in text.splitlines() or [""]:
        wrapped.extend(textwrap.wrap(
            raw_line, width=chars_per_line, replace_whitespace=False,
            drop_whitespace=False, break_long_words=True) or [""])

    pages = []
    for i in range(0, len(wrapped), lines_per_page):
        img = Image.new("1", (page_w, page_h), 1)
        draw = ImageDraw.Draw(img)
        y = _DOCUMENT_MARGIN
        for line in wrapped[i:i + lines_per_page]:
            draw.text((_DOCUMENT_MARGIN, y), line, font=font, fill=0)
            y += line_h
        pages.append(img.convert("RGB"))
    return pages or [Image.new("RGB", (page_w, page_h), (255, 255, 255))]


def _render_document_pages(input_path):
    """Dispatch to the right renderer for a document's extension -- shared
    by MediaViewer._load_document_file/_load_fmff_document (previews a
    PDF/.txt, plain or recovered from a .fmff, without writing anything
    beyond what the caller already has) and cmd_decode (.fmff -> .gif/
    .webp/.png of a document that was encoded the old way, before this
    became a preview-only path -- see this section's own docstring)."""
    ext = Path(input_path).suffix.lower()
    if ext == ".pdf":
        return _render_pdf_pages(input_path)
    if ext == ".txt":
        return _render_text_pages(input_path)
    raise ValueError(f"not a supported document type: {ext}")


def _render_document_thumbnail(input_path):
    """A single rendered page (the first) for a document's instant-
    preview thumbnail (see encode_document) -- cheap even for a huge
    PDF, unlike _render_document_pages (every page), which nothing
    calls at encode time any more now that a document's pages are a
    preview rather than the storage format themselves (see this
    section's own docstring)."""
    ext = Path(input_path).suffix.lower()
    if ext == ".pdf":
        if pymupdf is None:
            raise RuntimeError(
                "PDF encoding needs `pip install pymupdf` -- a first-page preview is "
                "rendered for the thumbnail, so that's the only real dependency this "
                "needs (the original PDF bytes are stored as-is either way)")
        pymupdf.TOOLS.set_aa_level(0)
        doc = pymupdf.open(str(input_path))
        try:
            if doc.page_count == 0:
                raise ValueError(f"{input_path} has no pages")
            pix = doc[0].get_pixmap(dpi=DOCUMENT_RENDER_DPI)
            mode = "RGBA" if pix.alpha else "RGB"
            return Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        finally:
            doc.close()
    if ext == ".txt":
        return _render_text_pages(input_path)[0]
    raise ValueError(f"not a supported document type: {ext}")


# --------------------------------------------------------------- RAW photos
# A camera RAW file (.cr2/.nef/.arw/.dng/...) holds the sensor's own
# unprocessed data, not pixels -- demosaicing (turning a Bayer sensor
# pattern into RGB) has to happen before there's an image FMFF's own tile
# codec, or anything else here, can work with at all. `rawpy` (a Python
# wrapper around LibRaw, the same library most photo tools use for this)
# does that demosaicing; the result then just becomes an ordinary still
# image, going through encode_image exactly like any decoded JPEG/PNG
# does -- no new content_mode, no new decoder logic, same as documents
# above reuse encode_image_sequence instead of inventing something new.

RAW_EXTS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw"}


def _load_raw_image(input_path):
    """Demosaic a RAW file into a PIL RGB Image via rawpy/LibRaw.
    use_camera_wb=True matches what the camera's own LCD preview and most
    RAW converters default to (the alternative, daylight white balance
    with no camera data at all, tends to look visibly wrong); everything
    else is left at rawpy's own defaults, including output orientation,
    which rawpy already derives from the RAW file's own orientation tag
    when present -- no separate handling needed for that here.

    EXIF (camera/GPS/orientation -- see _pack_metadata) is best-effort:
    most RAW formats are themselves TIFF-based, so Pillow's own TIFF
    reader can often pull the EXIF block out even though it can't
    demosaic the actual sensor data as pixels -- if that fails for a
    particular file/format, encoding still proceeds, just without EXIF,
    rather than failing the whole conversion over a metadata bonus."""
    if rawpy is None:
        raise RuntimeError(
            "RAW encoding needs `pip install rawpy` (a Python wrapper around "
            "LibRaw, the library that actually demosaics a camera sensor's "
            "raw data into pixels -- there's no way to read one of these "
            "files as an image without it)")
    with rawpy.imread(str(input_path)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
    img = Image.fromarray(rgb, "RGB")
    try:
        exif_bytes = Image.open(input_path).info.get("exif")
    except Exception:
        exif_bytes = None
    if exif_bytes:
        img.info["exif"] = exif_bytes
    return img


# --------------------------------------------------------------------- encoder

STILL_TILE_SIZE = 64
DEFAULT_AUDIO_BITRATE_KBPS = 128
# Above this, a source's own reported bitrate reads as lossless/
# uncompressed (WAV/FLAC commonly report four-figure kbps for CD-quality
# stereo PCM) rather than "already lossy at some particular target" --
# see _default_audio_bitrate.
_LOSSY_SOURCE_BITRATE_CEILING_KBPS = 500


def _default_audio_bitrate(source_bit_rate):
    """Auto-pick an Opus target bitrate for encode_audio when the caller
    leaves bitrate unset (None) -- an explicit bitrate is always used as
    given.

    Matching a lossy source's own bitrate exactly tends to come out
    *bigger*, not smaller: Opus is more efficient than most older lossy
    codecs at a given bitrate, so a comparable (or better) result usually
    needs a *lower* number, and blindly reusing the source's own bitrate
    just adds FMFF's own segment-table/index overhead on top of an
    already source-sized stream. Measured on a real 128kbps CBR MP3:
    encoding to a matched 128k Opus target came out 6% *bigger* than the
    source (a source at *exactly* the plain default is exactly the case
    an off-by-one "strictly less than the default" check would miss --
    worth calling out since it's the case that actually got measured);
    aiming for 75% of the source's own bitrate instead (96k here) came
    out 20% smaller, comfortably clearing the container overhead with
    real margin to spare. Capped at the plain default either way, so an
    already-high-bitrate lossy source (a 320k MP3, say) doesn't get
    pushed *above* the normal default just because it was well encoded.

    A high source bit_rate -- a lossless WAV/FLAC source, which ffprobe
    reports as its raw PCM bitrate, typically four-figure kbps -- isn't
    "already lossy at some low target" the same way a compressed source
    is, so that case just gets the plain default instead of some
    arbitrary fraction of a number that says nothing about perceptual
    quality (see _LOSSY_SOURCE_BITRATE_CEILING_KBPS).

    Floored at 32k, not the higher floor an earlier version of this used
    (64k): that higher floor defeated the whole point for an already
    very-low-bitrate source specifically -- a real ~73kbps VBR MP3 (short
    sound-effect content) still came out 11% *bigger* even after this
    function kicked in, because 75% of 73k rounds to 54k, well under the
    64k floor that was silently overriding it back up. Dropping the
    floor to 32k let that same file's real 54k target through, which
    measured 2% smaller -- barely, but no longer a regression."""
    source_kbps = source_bit_rate / 1000 if source_bit_rate else 0
    if source_kbps and source_kbps < _LOSSY_SOURCE_BITRATE_CEILING_KBPS:
        return f"{max(32, min(DEFAULT_AUDIO_BITRATE_KBPS, int(source_kbps * 0.75)))}k"
    return f"{DEFAULT_AUDIO_BITRATE_KBPS}k"


class FMFFEncoder:
    def __init__(self, tile_size=None, quality=80, thumb_max=64):
        # None means "use STILL_TILE_SIZE" (see encode_image). Only
        # relevant to still images -- an animated source has no fixed
        # tile grid at all (see encode_image_sequence), so tile_size is
        # ignored there regardless of what's passed here.
        self.tile_size = tile_size
        self.quality = quality
        self.thumb_max = thumb_max
        self.last_video_backend = None

    def encode(self, input_path, output_path, version_name=None, version_note=None):
        # version_name/version_note only ever apply to the encode_image
        # (plain still) path below -- silently unused for every other
        # branch here, the same way e.g. --tile-size is a no-op for a
        # video/audio source: an animated/JPEG-passthrough/document file
        # doesn't have FMFF's own per-tile addressing for a second version
        # to attach to in the first place (see add_version).
        ext = Path(input_path).suffix.lower()
        if ext in (".jpg", ".jpeg"):
            try:
                return self.encode_image_jpeg_passthrough(input_path, output_path)
            except JpegPassthroughUnsupported:
                pass  # progressive/CMYK JPEG, or jpeglib missing -- fall through below
        if ext in DOCUMENT_EXTS:
            return self.encode_document(input_path, output_path)
        img = Image.open(input_path)
        n_frames = getattr(img, "n_frames", 1)
        if n_frames > 1:
            # convert() right after seek(), not seek-then-copy(): PIL only
            # composites GIF disposal methods into the full frame during
            # the mode conversion, and each call must produce a genuinely
            # independent Image object (seek() mutates the *same* Image
            # in place, so appending `img` itself N times would just
            # append N references to whatever frame was sought last).
            frames, durations = [], []
            for i in range(n_frames):
                img.seek(i)
                has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
                frames.append(img.convert("RGBA" if has_alpha else "RGB"))
                durations.append(max(20, img.info.get("duration", 100)))
            return self.encode_image_sequence(frames, durations, output_path)
        return self.encode_image(img, output_path, version_name=version_name, version_note=version_note)

    def encode_document(self, input_path, output_path):
        """Store a PDF/.txt as itself -- the original bytes, not a picture
        of it. See this file's "document pages" section for the full
        reasoning (an earlier version of this rasterized every page
        instead, which was both less useful -- no text layer, search,
        hyperlinks, forms, or formulas -- and, despite throwing all of
        that away, routinely *bigger* than the source).

        Mirrors encode_image_jpeg_passthrough's approach exactly: race
        the original bytes against a general-purpose compressor (the
        same _JPEG_COEFF_CODECS registry JPEG passthrough uses --
        genuinely general-purpose despite the name) and keep whichever
        is smaller, tagged the same way (a single leading byte: b"R" for
        the original bytes verbatim, or one of _JPEG_COEFF_CODECS' own
        tags for which compressor won). A PDF's internal streams are
        usually already Flate-compressed, so recompression here often
        can't improve much further and the original bytes win -- but a
        .txt source (not compressed at all to start with) typically
        shrinks a lot. Either way this can never end up bigger than the
        source by more than this container's own small fixed overhead
        (header + index + one small preview thumbnail) -- unlike the old
        rasterize-everything approach, which routinely lost by several
        times over.

        The original document is fully intact inside the file: decode
        (or extract_document directly) gets back the exact source bytes,
        byte-for-byte. What FMFF's own viewer shows without a real PDF/
        text reader is just a cheap first-page preview thumbnail (see
        _render_document_thumbnail) -- full multi-page browsing
        re-renders straight from the recovered original bytes (see
        MediaViewer._load_fmff_document), not from anything stored
        per-page here."""
        original_bytes = Path(input_path).read_bytes()
        tag, compressed = min(
            ((tag, compress(original_bytes)) for tag, (compress, _) in _JPEG_COEFF_CODECS.items()),
            key=lambda kv: len(kv[1]))
        if len(compressed) < len(original_bytes):
            blob, compressed_won = tag + compressed, True
        else:
            blob, compressed_won = b"R" + original_bytes, False

        ext = Path(input_path).suffix.lower()
        thumb_img = _render_document_thumbnail(input_path).convert("RGB")
        width, height = thumb_img.size
        # Same reasoning as encode_image_jpeg_passthrough's identical
        # smaller-thumbnail-when-already-unbeatable trade: when the
        # original bytes won (couldn't be shrunk further), the container's
        # own fixed costs are the only thing separating this file from
        # matching the source exactly, so keep that overhead as small as
        # reasonably possible.
        thumb_cap = self.thumb_max if compressed_won else max(16, self.thumb_max // 2)
        tw, th, entries = _make_thumbnail_entries(thumb_img, thumb_cap, self.quality)

        index_offset = HEADER_SIZE
        frame_table_offset = index_offset + len(entries) * ENTRY_SIZE
        data_offset = frame_table_offset
        running = data_offset
        for e in entries:
            e.offset = running
            running += e.length
        media_blob_offset = running
        media_blob_length = len(blob)
        running += media_blob_length

        # doc_ext travels in the generic tags mechanism (same JSON blob
        # audio/video source tags use) rather than a new header field --
        # extract_document needs it to know whether the recovered bytes
        # are a PDF or a .txt when writing them back out to a real file.
        metadata_blob = _pack_metadata({"doc_ext": ext}, None, None)
        metadata_offset = running
        running += len(metadata_blob)

        header = struct.pack(
            HEADER_FMT, MAGIC, VERSION, width, height, 8,
            3, CONTENT_DOCUMENT, 0, 0, 0, 0, 0,
            0, 0, 0, len(entries),
            tw, th, HEADER_SIZE, index_offset, frame_table_offset, data_offset,
            0, media_blob_offset, media_blob_length,
            0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            metadata_offset, len(metadata_blob),
        )
        with open(output_path, "wb") as f:
            f.write(header)
            for e in entries:
                f.write(e.pack())
            for e in entries:
                f.write(e.payload)
            f.write(blob)
            f.write(metadata_blob)

        return {"width": width, "height": height, "size": running,
                "mode": "document-passthrough", "compressed": compressed_won,
                "source_ext": ext}

    def encode_image_jpeg_passthrough(self, input_path, output_path):
        """Losslessly repack an existing JPEG. First choice: pull its own
        already-quantized DCT coefficients straight out of it (via
        jpeglib/libjpeg -- no IDCT, no requantization, no pixel ever
        touched) and re-entropy-code those exact coefficients with
        whichever of a few general-purpose compressors is smallest. This
        routinely beats JPEG's baseline Huffman tables by 10-40% -- but
        not always: a JPEG already saved with *optimized* (per-image)
        Huffman tables (common from phone cameras and apps that
        recompress for sharing) can already be close to what a
        general-purpose byte compressor can do, and our simple int16
        coefficient stream doesn't model JPEG's own magnitude-category
        scheme the way its own entropy coder does -- so this can lose to
        the original. Rather than risk growing the file on exactly the
        files this method exists to shrink, the two are compared and the
        original JPEG bytes are stored verbatim, unchanged, whenever
        recompression doesn't actually win -- so this path never produces
        a file bigger than the source, only sometimes a merely-equal one.
        Either way, reconstructed pixels are identical to what the JPEG
        itself decodes to (within the same IDCT-rounding tolerance any
        two compliant decoders can differ by).

        This is deliberately NOT the same path as encode_image() on the
        JPEG's decoded pixels: that would redo FMFF's own lossy DCT and
        quantization on top of the JPEG's own -- a second lossy pass on
        data whose easy redundancy is already gone, which routinely comes
        out *larger* than the source, not smaller (see
        _default_quality_for for the fallback when this path can't run).

        Only handles baseline (non-progressive), 3-component (Y/Cb/Cr)
        JPEG. Raises JpegPassthroughUnsupported for anything else (or if
        `jpeglib` isn't installed) so the caller can fall back to
        encode_image()."""
        if jpeglib is None:
            raise JpegPassthroughUnsupported("jpeglib is not installed")
        im = jpeglib.read_dct(str(input_path))
        if im.progressive_mode or im.num_components != 3:
            raise JpegPassthroughUnsupported("progressive or non-YCbCr JPEG")

        comps = [im.Y, im.Cb, im.Cr]
        raw = b"".join(_dc_delta_encode(c).tobytes() for c in comps)
        tag, coeff_payload = min(
            ((tag, compress(raw)) for tag, (compress, _) in _JPEG_COEFF_CODECS.items()),
            key=lambda kv: len(kv[1]))

        meta = {
            "samp_factor": np.asarray(im.samp_factor).tolist(),
            "quant_tbl_no": np.asarray(im.quant_tbl_no).tolist(),
            "qt": np.asarray(im.qt).tolist(),
            "shapes": [list(c.shape[:2]) for c in comps],
        }
        meta_json = json.dumps(meta).encode("utf-8")
        recompressed = b"C" + struct.pack("<I", len(meta_json)) + meta_json + tag + coeff_payload

        original_bytes = Path(input_path).read_bytes()
        if len(recompressed) < len(original_bytes):
            blob, recompressed_won = recompressed, True
        else:
            blob, recompressed_won = b"R" + original_bytes, False

        source_pil = Image.open(input_path)
        # Captured before convert() below (which doesn't carry .info over)
        # -- byte-for-byte EXIF passthrough (see _pack_metadata), needed
        # here specifically because the recompressed-coefficients path
        # keeps only the DCT data and quant tables, not the JPEG's other
        # segments -- EXIF would otherwise quietly vanish exactly when
        # recompression (the whole point of this method) wins. The
        # verbatim-original-bytes fallback already carries its EXIF
        # segment inside `blob` regardless, but capturing it here too is
        # harmless and keeps this method's metadata handling uniform.
        exif_bytes = source_pil.info.get("exif")
        icc_bytes = source_pil.info.get("icc_profile")
        thumb_img = source_pil.convert("RGB")
        # When recompression didn't win, the container's own fixed costs
        # (header + index + this thumbnail) are the *only* things standing
        # between this file and matching the source exactly -- storing the
        # original bytes verbatim is itself unbeatable (can't losslessly
        # store N bytes in fewer than N), so that overhead can only be
        # minimized, never fully eliminated. Use a smaller thumbnail here
        # specifically to keep it as small as reasonably possible.
        thumb_cap = self.thumb_max if recompressed_won else max(16, self.thumb_max // 2)
        tw, th, entries = _make_thumbnail_entries(thumb_img, thumb_cap, self.quality)

        index_offset = HEADER_SIZE
        frame_table_offset = index_offset + len(entries) * ENTRY_SIZE
        data_offset = frame_table_offset
        running = data_offset
        for e in entries:
            e.offset = running
            running += e.length
        media_blob_offset = running
        media_blob_length = len(blob)
        running += media_blob_length

        metadata_blob = _pack_metadata(None, exif_bytes, icc_bytes)
        metadata_offset = running
        running += len(metadata_blob)

        header = struct.pack(
            HEADER_FMT, MAGIC, VERSION, im.width, im.height, 8,
            3, CONTENT_JPEG_PASSTHROUGH, 0, 0, 0, 0, 0,
            0, 0, 0, len(entries),
            tw, th, HEADER_SIZE, index_offset, frame_table_offset, data_offset,
            0, media_blob_offset, media_blob_length,
            0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            metadata_offset, len(metadata_blob),
        )
        with open(output_path, "wb") as f:
            f.write(header)
            for e in entries:
                f.write(e.pack())
            for e in entries:
                f.write(e.payload)
            f.write(blob)
            f.write(metadata_blob)

        return {"width": im.width, "height": im.height, "has_alpha": False,
                "tiles": 0, "size": running, "mode": "jpeg-passthrough",
                "recompressed": recompressed_won}

    def encode_image(self, img, output_path, version_name=None, version_note=None):
        # Grabbed before convert() below: Pillow's convert() doesn't carry
        # the source's .info dict over to the new Image object, so this is
        # the only chance to reach the source's raw EXIF bytes (byte-for-
        # byte passthrough -- GPS/camera/orientation tags included -- not
        # re-derived; see _pack_metadata) and its ICC color profile, same
        # deal.
        exif_bytes = img.info.get("exif")
        icc_bytes = img.info.get("icc_profile")
        has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
        img = img.convert("RGBA" if has_alpha else "RGB")
        width, height = img.size
        arr = np.array(img)
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3] if has_alpha else None

        tw, th, entries = _make_thumbnail_entries(img, self.thumb_max, self.quality)

        ts = self.tile_size if self.tile_size is not None else STILL_TILE_SIZE
        tiles_x, tiles_y = -(-width // ts), -(-height // ts)
        frame_entry_start = len(entries)

        tile_dims = {}  # (tx, ty) -> (w, h), needed once results come back
        jobs = []
        for ty in range(tiles_y):
            y0, y1 = ty * ts, min((ty + 1) * ts, height)
            for tx in range(tiles_x):
                x0, x1 = tx * ts, min((tx + 1) * ts, width)
                tile_dims[tx, ty] = (x1 - x0, y1 - y0)
                a_tile = alpha[y0:y1, x0:x1] if has_alpha else None
                jobs.append((0, tx, ty, rgb[y0:y1, x0:x1, :], a_tile, self.quality))

        # See the "multi-core tiling" section above _get_tile_pool for why
        # this is multiprocessing rather than GPU: profiling showed the
        # per-tile cost is dominated by zlib/bz2 (serial, CPU-only), not
        # the DCT math, so spreading independent tiles across CPU cores is
        # what actually attacks the bottleneck. Small images skip the
        # pool -- process-startup cost isn't worth it below the threshold.
        if len(jobs) >= _MP_TILE_THRESHOLD:
            results = _get_tile_pool().map(_encode_tile_task, jobs)
        else:
            results = [_encode_tile_task(job) for job in jobs]

        for _frame, tx, ty, mode, payload, alpha_payload in results:
            w, h = tile_dims[tx, ty]
            entries.append(Entry(0, LAYER_FULL, tx, ty, PLANE_COLOR, mode, w, h, payload))
            if alpha_payload is not None:
                entries.append(Entry(0, LAYER_FULL, tx, ty, PLANE_ALPHA, MODE_LOSSLESS, w, h,
                                      alpha_payload))
        frame_table = [(0, frame_entry_start, len(entries) - frame_entry_start)]

        index_offset = HEADER_SIZE
        frame_table_offset = index_offset + len(entries) * ENTRY_SIZE
        data_offset = frame_table_offset + len(frame_table) * FRAME_SIZE
        running = data_offset
        for e in entries:
            e.offset = running
            running += e.length

        # Only recorded at all when the caller actually named/noted this
        # version -- an encode with neither writes exactly the same
        # metadata blob (and file bytes) as before this existed, so a
        # plain `encode` with no --version-name/--version-note is a
        # complete no-op for this feature (see _pack_metadata/
        # FMFFDecoder.list_versions, which fall back to "original" for an
        # unnamed version 0).
        versions_meta = None
        if version_name or version_note:
            versions_meta = [{
                "index": 0,
                "name": version_name or "original",
                "note": version_note or "",
                "added": _now_iso(),
                "size": sum(e.length for e in entries if e.layer == LAYER_FULL),
            }]
        metadata_blob = _pack_metadata(None, exif_bytes, icc_bytes, versions_meta)
        metadata_offset = running
        running += len(metadata_blob)

        header = struct.pack(
            HEADER_FMT, MAGIC, VERSION, width, height, 8,
            4 if has_alpha else 3, CONTENT_IMAGE, ts, tiles_x, tiles_y,
            1 if has_alpha else 0, self.quality,
            0, len(frame_table), 0, len(entries),
            tw, th, HEADER_SIZE, index_offset, frame_table_offset, data_offset,
            0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            metadata_offset, len(metadata_blob),
        )

        with open(output_path, "wb") as f:
            f.write(header)
            for e in entries:
                f.write(e.pack())
            for t_ms, start, count in frame_table:
                f.write(struct.pack(FRAME_FMT, t_ms, start, count))
            for e in entries:
                f.write(e.payload)
            f.write(metadata_blob)

        return {
            "width": width, "height": height, "has_alpha": has_alpha,
            "tiles": tiles_x * tiles_y, "size": running, "mode": "tile-codec",
        }

    def add_version(self, fmff_path, input_path, name=None, note=None):
        """Append a new named version to an existing still-image .fmff, in
        place -- e.g. an original photo's file gaining a "retouched"
        version alongside it, both in the same container.

        Stores only the tiles whose freshly-encoded bytes actually differ
        from whatever's already the effective encoding at that position
        (version 0's, or the latest earlier version that touched it) --
        not a full second copy of the image. This is the same saving
        encode_image_sequence already gets from an unchanged animation
        frame ("a frame identical to the previous one gets no entry at
        all"), applied to a version chain instead of a time axis -- but
        compared at the *encoded-bytes* level, not raw pixels: a lossy
        tile's decode is never bit-exact, so comparing this version's
        source pixels against the previous version's *decoded* (lossy,
        already-quantized) pixels would flag nearly every tile as
        "changed" from quantization noise alone, even ones nobody
        touched. Comparing what this tile would actually encode to
        against what's already stored has no such false positives (and
        no false negatives either -- any real difference, however small,
        changes the encoded bytes and gets stored) at the cost of every
        tile still being fully encoded to find out either way; the
        saving is in what gets *written*, not in encode time. A version's
        tile grid is guaranteed identical to every other version's in the
        same file (see the width/height check below), so there's no need
        for _split_changed_regions' region-finding -- comparing the fixed
        grid's tiles one by one is enough. A version that only touches a
        corner of the image (crop a watermark, fix a blemish) costs
        roughly that corner, not the whole picture; a version that
        encodes identically to its predecessor (e.g. adding a note
        without changing the picture) costs no tile data at all, just
        its metadata entry. A whole-image edit (a global color grade)
        still touches every tile and costs close to a full copy -- this
        doesn't create savings that aren't there, it just stops charging
        for the ones that are.

        The trade-off for this: decoding version N means replaying every
        version between it and the last full-coverage version at or
        before it (see full()). Left unbounded, that would mean a long,
        heavily-edited chain costs more and more to decode as it grows --
        so every VERSION_SNAPSHOT_INTERVAL-th version is stored in full
        regardless of what actually changed (see is_snapshot below),
        giving both full() and this method's own change-comparison a
        nearby restart point instead of always going back to version 0.
        Replay depth (and the cost of preparing this method's own
        comparison, see _nearest_full_version) is bounded by that
        interval, not by how long the chain has grown to -- at the cost
        of one full version's worth of extra storage every that many
        versions, not a full copy every version.

        Scoped to CONTENT_IMAGE, non-animated files on purpose (see
        LAYER_VERSION's own comment and this file's container-layout
        docstring): video/audio/JPEG-passthrough/a document are already
        one opaque blob with no per-tile addressing to append a version
        into cheaply, and an animated image's entries carry literal
        per-frame pixel rectangles instead of a fixed tile grid, so
        there's no shared per-version geometry the way a plain still's
        tile grid gives for free.

        The new version is encoded at the file's own already-stored
        quality (dec.quality), never self.quality -- a lossy tile's
        dequantization at decode time uses the single, file-wide
        `quality` header field (see FMFFDecoder._decode_entry), so a
        version encoded at a different quality would decode wrong; every
        version in a file necessarily shares one quality setting.

        Alpha is likewise a whole-file decision, not a per-version one:
        the header's `has_alpha` flag (set once, from version 0) applies
        to every version -- a version added to an alpha-less file has
        its own alpha silently dropped, and one added to an alpha file
        gets a fully-opaque alpha plane if its own source has none. A
        per-tile mix of "some versions have alpha data here, some don't"
        would make "did this tile actually change" ambiguous across a
        version boundary where alpha appears/disappears, for no real
        benefit -- a still image's transparency is normally a property
        of the picture itself, not something one retouch pass alone
        introduces.

        The new version must match the file's own width/height exactly:
        versions are meant to be the same picture, retouched/annotated/
        masked/etc., not an unrelated image that happens to share a
        file -- and a fixed tile grid has nowhere to put a differently-
        sized version even if that weren't the intent."""
        dec = FMFFDecoder(fmff_path)
        if dec.content_mode != CONTENT_IMAGE:
            raise ValueError("add_version only supports a still-image .fmff "
                              "(not video/audio/JPEG-passthrough/a document)")
        if dec.is_animated:
            raise ValueError("add_version doesn't support an already-animated .fmff -- "
                              "its entries have no fixed tile grid for a version to share")
        if not dec.has_fixed_tile_grid:
            raise ValueError("this file has no fixed tile grid for a version to share")

        img = Image.open(input_path).convert("RGBA" if dec.has_alpha else "RGB")
        if img.size != (dec.width, dec.height):
            raise ValueError(f"version must match the file's own size {dec.width}x{dec.height} "
                              f"(got {img.size[0]}x{img.size[1]})")

        arr = np.array(img)
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3] if dec.has_alpha else None

        existing_indices = ({0} | {e.frame for e in dec.entries if e.layer == LAYER_VERSION}
                             | {v["index"] for v in dec.versions_meta if "index" in v})
        version_index = max(existing_indices) + 1

        # Every existing entry's payload, read once -- reused both to
        # work out each tile position's currently-effective encoding
        # (compared against below) and, unchanged, to copy every earlier
        # version's bytes through into the rewritten file without
        # re-encoding them.
        with open(dec.path, "rb") as f:
            old_payloads = []
            for e in dec.entries:
                f.seek(e.offset)
                old_payloads.append(f.read(e.length))

        # The (mode, payload) currently in effect at every (tile, plane)
        # position as of version_index - 1: the nearest full-coverage
        # version at or before it (see _nearest_full_version -- version
        # 0's tiles always qualify, so this is always safe) covers every
        # position unconditionally, then each version after it, in order,
        # overwrites whatever positions it actually touched -- the
        # entry-level equivalent of what full(version=version_index - 1)
        # reconstructs pixel-by-pixel, built here from bytes already in
        # hand instead of decoding an image (see this method's own
        # docstring for why pixel comparison against a decoded lossy
        # version isn't safe). Starting from the nearest full version
        # instead of always version 0 keeps this bounded the same way
        # full()'s own replay is (see VERSION_SNAPSHOT_INTERVAL) -- a
        # long version chain doesn't make every later add_version call
        # slower to prepare.
        start = dec._nearest_full_version(version_index - 1)
        base_layer = LAYER_FULL if start == 0 else LAYER_VERSION
        effective = {}
        for e, payload in zip(dec.entries, old_payloads):
            if e.layer == base_layer and e.frame == start:
                effective[e.tx, e.ty, e.plane] = (e.mode, payload)
        for v in range(start + 1, version_index):
            for e, payload in zip(dec.entries, old_payloads):
                if e.layer == LAYER_VERSION and e.frame == v:
                    effective[e.tx, e.ty, e.plane] = (e.mode, payload)

        ts, tiles_x, tiles_y = dec.tile_size, dec.tiles_x, dec.tiles_y
        tile_dims = {}
        jobs = []
        for ty in range(tiles_y):
            y0, y1 = ty * ts, min((ty + 1) * ts, dec.height)
            for tx in range(tiles_x):
                x0, x1 = tx * ts, min((tx + 1) * ts, dec.width)
                tile_dims[tx, ty] = (x1 - x0, y1 - y0)
                a_tile = alpha[y0:y1, x0:x1] if dec.has_alpha else None
                jobs.append((version_index, tx, ty, rgb[y0:y1, x0:x1, :], a_tile, dec.quality))

        if len(jobs) >= _MP_TILE_THRESHOLD:
            results = _get_tile_pool().map(_encode_tile_task, jobs)
        else:
            results = [_encode_tile_task(job) for job in jobs]

        # Every tile got fully (re-)encoded above regardless -- there's
        # no way to know whether it changed enough to matter without
        # actually racing it through the same lossless/lossy/palette
        # codec every earlier version already went through. What's
        # actually saved is what gets *stored*: a tile whose freshly
        # encoded bytes match the position's current effective encoding
        # exactly isn't stored again -- decode falls through to the
        # existing entry the same way an untouched tile position already
        # does for any other version (see full()).
        #
        # Every VERSION_SNAPSHOT_INTERVAL-th version is the one exception:
        # every one of its tiles is stored regardless of whether it
        # changed, making it a full-coverage restart point full() and a
        # later add_version can jump to instead of always replaying back
        # to version 0 (see _nearest_full_version). This is the only
        # place that decision gets made -- everything else about a
        # snapshot version (its metadata, its place in the version list)
        # is identical to an ordinary version.
        is_snapshot = version_index % VERSION_SNAPSHOT_INTERVAL == 0
        new_entries = []
        for v_idx, tx, ty, mode, payload, alpha_payload in results:
            w, h = tile_dims[tx, ty]
            if is_snapshot or effective.get((tx, ty, PLANE_COLOR)) != (mode, payload):
                new_entries.append(Entry(v_idx, LAYER_VERSION, tx, ty, PLANE_COLOR, mode, w, h, payload))
            if alpha_payload is not None:
                if is_snapshot or effective.get((tx, ty, PLANE_ALPHA)) != (MODE_LOSSLESS, alpha_payload):
                    new_entries.append(Entry(v_idx, LAYER_VERSION, tx, ty, PLANE_ALPHA, MODE_LOSSLESS,
                                              w, h, alpha_payload))

        all_entries = list(dec.entries) + new_entries
        all_payloads = old_payloads + [e.payload for e in new_entries]

        versions_meta = [dict(v) for v in dec.versions_meta if "index" in v]
        if not any(v["index"] == 0 for v in versions_meta):
            versions_meta.insert(0, {
                "index": 0, "name": "original", "note": "", "added": "",
                "size": sum(e.length for e in dec.entries
                            if e.layer == LAYER_FULL and e.plane != PLANE_MASK),
            })
        versions_meta.append({
            "index": version_index,
            "name": name or f"version {version_index}",
            "note": note or "",
            "added": _now_iso(),
            "size": sum(e.length for e in new_entries),
        })

        index_offset = HEADER_SIZE
        frame_table_offset = index_offset + len(all_entries) * ENTRY_SIZE
        data_offset = frame_table_offset + len(dec.frame_table) * FRAME_SIZE
        running = data_offset
        for e, payload in zip(all_entries, all_payloads):
            e.offset = running
            running += len(payload)

        metadata_blob = _pack_metadata(dec.tags or None, dec.exif_bytes, dec.icc_bytes, versions_meta)
        metadata_offset = running
        running += len(metadata_blob)

        header = struct.pack(
            HEADER_FMT, MAGIC, VERSION, dec.width, dec.height, 8,
            4 if dec.has_alpha else 3, CONTENT_IMAGE, dec.tile_size, dec.tiles_x, dec.tiles_y,
            1 if dec.has_alpha else 0, dec.quality,
            0, len(dec.frame_table), 0, len(all_entries),
            dec.thumb_w, dec.thumb_h, HEADER_SIZE, index_offset, frame_table_offset, data_offset,
            0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            metadata_offset, len(metadata_blob),
        )

        # Written to a temp file and swapped in atomically -- add_version
        # rewrites the whole file (offsets shift once new entries are
        # inserted into the index), so a crash/interruption mid-write
        # must not leave fmff_path itself half-overwritten.
        tmp_path = Path(str(fmff_path) + ".tmp")
        with open(tmp_path, "wb") as f:
            f.write(header)
            for e in all_entries:
                f.write(e.pack())
            for t_ms, start, count in dec.frame_table:
                f.write(struct.pack(FRAME_FMT, t_ms, start, count))
            for payload in all_payloads:
                f.write(payload)
            f.write(metadata_blob)
        tmp_path.replace(fmff_path)

        return {
            "version_index": version_index,
            "name": versions_meta[-1]["name"],
            "size": versions_meta[-1]["size"],
            "total_size": running,
        }

    def add_mask(self, fmff_path, mask_path, version=None):
        """Attach a selection mask -- or any other single-channel,
        per-pixel annotation a caller wants back out unchanged -- to one
        version of an existing still-image .fmff, in place. Stored as a
        new PLANE_MASK entry under that version's own (layer, frame),
        sharing the tile grid every color/alpha tile already uses.

        Encoded losslessly (see PLANE_MASK's own comment) -- always
        lossless_encode, never raced against the lossy DCT path the way
        a color tile is: a mask's exact edges are the whole point, not
        something to approximate for a smaller file. This mirrors
        PLANE_ALPHA's own encoding exactly.

        version=None (the default) targets the file's most recently
        added version (see list_versions), or version 0 if the file has
        no add_version'd versions at all -- the version most likely to
        be "the one this mask goes with" right after add-version and
        add-mask are run back to back. Calling this again for a version
        that already has a mask replaces it; every other version's own
        mask (or lack of one) is untouched.

        Deliberately NOT part of the version chain's delta/overlay
        mechanism full()'s version path uses for color/alpha (see
        FMFFDecoder.extract_mask) -- a mask belongs to exactly the
        version it was attached to, with no inheritance from an earlier
        version the way an unedited color/alpha tile is inherited. A
        selection mask drawn for one retouch pass isn't implicitly
        "still correct" for a later, different retouch, and there'd be
        no reliable way to tell a deliberately-reused mask from one that
        was simply never revisited.

        The mask image must match the file's own width/height exactly,
        for the same reason a new version must (see add_version) -- a
        fixed tile grid has nowhere to put a differently-sized plane."""
        dec = FMFFDecoder(fmff_path)
        if dec.content_mode != CONTENT_IMAGE:
            raise ValueError("add_mask only supports a still-image .fmff "
                              "(not video/audio/JPEG-passthrough/a document)")
        if dec.is_animated:
            raise ValueError("add_mask doesn't support an already-animated .fmff -- "
                              "its entries have no fixed tile grid to attach a mask to")
        if not dec.has_fixed_tile_grid:
            raise ValueError("this file has no fixed tile grid to attach a mask to")

        versions = dec.list_versions()
        if version is None:
            version = versions[-1]["index"] if versions else 0
        valid = {v["index"] for v in versions}
        if version not in valid:
            raise ValueError(f"version {version} not found in {fmff_path} "
                              f"(available: {sorted(valid)})")

        img = Image.open(mask_path).convert("L")
        if img.size != (dec.width, dec.height):
            raise ValueError(f"mask must match the file's own size {dec.width}x{dec.height} "
                              f"(got {img.size[0]}x{img.size[1]})")
        mask_arr = np.array(img)

        layer = LAYER_FULL if version == 0 else LAYER_VERSION

        with open(dec.path, "rb") as f:
            old_payloads = []
            for e in dec.entries:
                f.seek(e.offset)
                old_payloads.append(f.read(e.length))

        # Drop this version's existing mask, if any -- add_mask replaces
        # rather than stacking. Every other entry (every color/alpha
        # tile, every other version's own mask) is kept exactly as-is.
        kept_entries, kept_payloads = [], []
        for e, payload in zip(dec.entries, old_payloads):
            if e.layer == layer and e.frame == version and e.plane == PLANE_MASK:
                continue
            kept_entries.append(e)
            kept_payloads.append(payload)

        ts, tiles_x, tiles_y = dec.tile_size, dec.tiles_x, dec.tiles_y
        new_entries = []
        for ty in range(tiles_y):
            y0, y1 = ty * ts, min((ty + 1) * ts, dec.height)
            for tx in range(tiles_x):
                x0, x1 = tx * ts, min((tx + 1) * ts, dec.width)
                tile = mask_arr[y0:y1, x0:x1]
                payload = lossless_encode(tile[None, :, :])
                new_entries.append(Entry(version, layer, tx, ty, PLANE_MASK, MODE_LOSSLESS,
                                          x1 - x0, y1 - y0, payload))

        all_entries = kept_entries + new_entries
        all_payloads = kept_payloads + [e.payload for e in new_entries]

        # Nothing in the metadata blob needs to change -- has_mask/
        # mask_size (see list_versions) are derived from the entries
        # themselves, not recorded redundantly in "versions" metadata.
        metadata_blob = _pack_metadata(dec.tags or None, dec.exif_bytes, dec.icc_bytes,
                                        dec.versions_meta or None)

        index_offset = HEADER_SIZE
        frame_table_offset = index_offset + len(all_entries) * ENTRY_SIZE
        data_offset = frame_table_offset + len(dec.frame_table) * FRAME_SIZE
        running = data_offset
        for e, payload in zip(all_entries, all_payloads):
            e.offset = running
            running += len(payload)
        metadata_offset = running
        running += len(metadata_blob)

        header = struct.pack(
            HEADER_FMT, MAGIC, VERSION, dec.width, dec.height, 8,
            4 if dec.has_alpha else 3, CONTENT_IMAGE, dec.tile_size, dec.tiles_x, dec.tiles_y,
            1 if dec.has_alpha else 0, dec.quality,
            0, len(dec.frame_table), 0, len(all_entries),
            dec.thumb_w, dec.thumb_h, HEADER_SIZE, index_offset, frame_table_offset, data_offset,
            0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            metadata_offset, len(metadata_blob),
        )

        # Same atomic swap as add_version -- add_mask also rewrites the
        # whole file (the index grows/shrinks by this version's mask
        # entries), so a crash mid-write must not leave fmff_path itself
        # half-overwritten.
        tmp_path = Path(str(fmff_path) + ".tmp")
        with open(tmp_path, "wb") as f:
            f.write(header)
            for e in all_entries:
                f.write(e.pack())
            for t_ms, start, count in dec.frame_table:
                f.write(struct.pack(FRAME_FMT, t_ms, start, count))
            for payload in all_payloads:
                f.write(payload)
            f.write(metadata_blob)
        tmp_path.replace(fmff_path)

        return {"version_index": version, "size": sum(e.length for e in new_entries),
                "total_size": running}

    def encode_image_sequence(self, frames, durations_ms, output_path):
        """Encode a multi-frame source (animated GIF/WebP/APNG) as a single
        .fmff file with more than one frame, instead of collapsing it to a
        still. This reuses infrastructure the container already had but a
        single still image never exercises: Entry.frame and the frame
        table (HEADER_FMT's frame_count/frame_table_offset, FRAME_FMT --
        sized originally for a pre-FFmpeg version of FMFF's own video
        codec, unused since video moved to AV1/FFmpeg; see the module
        docstring). `durations_ms` is this frame's display duration;
        frame_table stores cumulative start times, the same convention
        `frame_table` already used.

        Earlier versions of this reused the still-image tile grid for
        animation: chop every frame into fixed tiles, skip a tile that's
        pixel-identical to the same grid cell in the previous frame, and
        race lossless/palette/lossy for whichever tiles are left (see
        _race_color_candidates). Tuning the tile size for that (many
        small tiles for parallelism/fine-grained resilience versus one
        tile covering the whole frame for minimal fixed overhead) closed
        most, but not all, of the gap to well-optimized source GIFs.

        What was left was a real, content-shape-dependent limitation of a
        *fixed* grid: a real-world GIF with a small, localized moving
        region (a spinner, a bit of animated text) on a large mostly-
        static canvas changes only a couple thousand pixels a frame --
        but if that region straddles several grid tiles, or is smaller
        than one big single-tile-per-frame region, the grid still
        re-stores every whole tile it touches, most of whose area didn't
        actually change. Measured on a real 550x400, 30-frame GIF like
        this: no fixed tile size (from 32px up to one tile per frame)
        got the .fmff within 50% of the 40KB source, because the
        changed region was consistently much smaller than whatever
        tile(s) it landed in.

        So this doesn't tile the frame into a grid at all. For frame 0
        (no previous frame to compare against) the "changed region" is
        the whole frame, same as before. For every later frame, the
        tight bounding box of every pixel that differs from the previous
        frame (color or alpha) is computed directly -- one np.nonzero
        call, not a per-tile loop -- and, if anything changed, exactly
        that rectangle is raced through _race_color_candidates and
        stored as a single entry positioned at that rectangle's own
        (x, y) instead of a tile-grid cell (LAYER_FULL entries' tx/ty
        fields are literal pixel coordinates for an animated file, not
        grid indices -- see full()/full_sequence()). A frame identical
        to the previous one still gets no entry at all. This adapts
        automatically to content shape with a single mechanism: a small
        localized change stores a small rectangle (exactly what a fixed
        small tile size was trying, and failing, to do without also
        paying per-tile overhead for every tile the change happened to
        touch); a change spread across most of the frame stores close to
        the whole frame (exactly what one-tile-per-frame was trying to
        do); either way it's still one entry per changed frame, so the
        fixed per-entry cost (27-byte index entry, CRC32, per-stream
        compression overhead) is paid at most once per frame regardless
        of how the change is shaped. On that same 550x400 test GIF, this
        took the .fmff from 56% bigger than the source (the best any
        fixed tile size managed) to smaller than the source.

        One bounding box still falls short on a different content shape:
        several small changes scattered around the frame instead of one
        localized blob (e.g. a logo whose glint sweeps through more than
        one corner of an otherwise-static frame) -- a single bbox around
        all of them re-stores whatever untouched background sits between
        them too. _split_changed_regions handles that by splitting the
        changed pixels into however many tight, non-overlapping
        rectangles they actually need instead of always one; see its own
        docstring for how. Frame 0 still gets the whole frame as one
        region, same as before."""
        # Grabbed before convert() below drops it -- rare for an animated
        # source in practice, but harmless to check (see encode_image's
        # identical comment for the still-image case this mirrors).
        exif_bytes = frames[0].info.get("exif")
        icc_bytes = frames[0].info.get("icc_profile")
        has_alpha = any(f.mode in ("RGBA", "LA") or (f.mode == "P" and "transparency" in f.info)
                         for f in frames)
        pil_mode = "RGBA" if has_alpha else "RGB"
        frames = [f.convert(pil_mode) for f in frames]
        width, height = frames[0].size

        tw, th, entries = _make_thumbnail_entries(frames[0], self.thumb_max, self.quality)

        jobs = []  # (frame, x0, y0, color_rect, alpha_rect, quality)
        prev_rgb = prev_alpha = None
        for fi, img in enumerate(frames):
            arr = np.array(img)
            rgb = arr[:, :, :3]
            alpha = arr[:, :, 3] if has_alpha else None
            if fi == 0:
                regions = [(0, height, 0, width)]
            else:
                changed = np.any(rgb != prev_rgb, axis=2)
                if has_alpha:
                    changed |= (alpha != prev_alpha)
                if not changed.any():
                    prev_rgb, prev_alpha = rgb, alpha
                    continue  # identical to the previous frame -- no entry needed
                regions = _split_changed_regions(changed)
            for y0, y1, x0, x1 in regions:
                color_rect = rgb[y0:y1, x0:x1, :]
                alpha_rect = alpha[y0:y1, x0:x1] if has_alpha else None
                jobs.append((fi, x0, y0, color_rect, alpha_rect, self.quality))
            prev_rgb, prev_alpha = rgb, alpha

        # Same multi-core tiling as encode_image (see _get_tile_pool) --
        # every changed-region job is independent of every other one, so
        # the whole animation's jobs are spread across the pool at once
        # rather than frame-by-frame. Most frames still produce at most
        # one color + one alpha job (one region each), but a frame whose
        # change is scattered can now contribute several (see
        # _split_changed_regions) -- this pool activates less often for
        # short animations than it would for a still image's tile grid
        # either way, acceptable since it's encode time, not the file
        # size or decode time this whole rewrite targets.
        if len(jobs) >= _MP_TILE_THRESHOLD:
            results = _get_tile_pool().map(_encode_tile_task, jobs)
        else:
            results = [_encode_tile_task(job) for job in jobs]

        # Pool.map preserves input order even though workers finish out of
        # order, so results lines up element-for-element with jobs -- no
        # need for a dims lookup table, just zip them back together.
        frame_table = []
        cumulative_ms = 0
        current_frame = 0
        frame_start = len(entries)
        for (frame, x0, y0, color_rect, _alpha_rect, _q), (rframe, rx0, ry0, mode, payload, alpha_payload) \
                in zip(jobs, results):
            h, w = color_rect.shape[:2]
            while frame != current_frame:
                frame_table.append((cumulative_ms, frame_start, len(entries) - frame_start))
                cumulative_ms += durations_ms[current_frame]
                current_frame += 1
                frame_start = len(entries)
            entries.append(Entry(frame, LAYER_FULL, x0, y0, PLANE_COLOR, mode, w, h, payload))
            if alpha_payload is not None:
                entries.append(Entry(frame, LAYER_FULL, x0, y0, PLANE_ALPHA, MODE_LOSSLESS, w, h,
                                      alpha_payload))
        while current_frame < len(frames):
            frame_table.append((cumulative_ms, frame_start, len(entries) - frame_start))
            cumulative_ms += durations_ms[current_frame]
            current_frame += 1
            frame_start = len(entries)

        index_offset = HEADER_SIZE
        frame_table_offset = index_offset + len(entries) * ENTRY_SIZE
        data_offset = frame_table_offset + len(frame_table) * FRAME_SIZE
        running = data_offset
        for e in entries:
            e.offset = running
            running += e.length

        metadata_blob = _pack_metadata(None, exif_bytes, icc_bytes)
        metadata_offset = running
        running += len(metadata_blob)

        # tile_size/tiles_x/tiles_y are meaningless here -- there's no
        # fixed grid for an animated file any more (see this method's
        # docstring) -- so they're left at 0; full()/full_sequence() key
        # off is_animated to know LAYER_FULL entries carry literal pixel
        # (x, y) instead of grid indices, not off these fields.
        header = struct.pack(
            HEADER_FMT, MAGIC, VERSION, width, height, 8,
            4 if has_alpha else 3, CONTENT_IMAGE, 0, 0, 0,
            1 if has_alpha else 0, self.quality,
            0, len(frame_table), 0, len(entries),
            tw, th, HEADER_SIZE, index_offset, frame_table_offset, data_offset,
            0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            metadata_offset, len(metadata_blob),
        )

        with open(output_path, "wb") as f:
            f.write(header)
            for e in entries:
                f.write(e.pack())
            for t_ms, start, count in frame_table:
                f.write(struct.pack(FRAME_FMT, t_ms, start, count))
            for e in entries:
                f.write(e.payload)
            f.write(metadata_blob)

        return {
            "width": width, "height": height, "has_alpha": has_alpha,
            "frames": len(frames), "size": running,
            "mode": "tile-codec-animated",
        }

    def encode_video(self, input_path, output_path, crf=30, speed=8, fps=None,
                      progress_cb=None, cancel_event=None):
        """Encode video via FFmpeg into AV1 (libsvtav1) + Opus -- see the
        module docstring's "Container layout for a video" section for why
        video doesn't use FMFF's own tile codec, and how the result ends
        up as a small init chunk plus a run of independently CRC32-checked
        segments instead of one opaque blob. crf: AV1 quality (0-63, lower
        = better/bigger; 28-34 is a reasonable visually-good range).
        speed: SVT-AV1 encoder preset (0-13, higher = faster/lower quality
        at the same crf). cancel_event: an optional threading.Event --
        checked between FFmpeg progress lines (so about as fine-grained as
        the video is long, not just between whole files), and if set the
        ffmpeg subprocess is killed and EncodingCancelled is raised
        instead of writing output_path.

        A text-based subtitle track (SRT/ASS/WebVTT/already-mov_text --
        see _TEXT_SUBTITLE_CODECS) rides along as a third track in this
        same fragmented MP4, converted to MP4's own mov_text format --
        no new header fields or segment table needed for it at all, since
        it's just more bytes inside the one blob extract_media() already
        hands back whole. An image-based subtitle track (PGS/DVD subs)
        is left out instead of attempted and failed -- see
        _TEXT_SUBTITLE_CODECS's docstring."""
        ffmpeg = _find_ffmpeg()
        if ffmpeg is None:
            raise RuntimeError(
                "video encoding needs FFmpeg (with libsvtav1 + libopus) on PATH -- "
                "install it, e.g. `winget install BtbN.FFmpeg.LGPL.8.1` on Windows")
        ffprobe = _find_ffprobe()
        if ffprobe is None:
            raise RuntimeError("video encoding needs ffprobe alongside ffmpeg to read "
                                "source video metadata (it ships in the same FFmpeg build)")

        input_path = str(input_path)
        info = _ffprobe_info(input_path, ffprobe)
        if not info["width"]:
            raise ValueError(f"no video stream found in {input_path}")
        out_fps = fps or info["fps"] or 25.0
        has_video_alpha = info["has_alpha"]

        with tempfile.TemporaryDirectory(prefix="fmff_enc_") as tmp:
            gop = max(1, int(round(out_fps * VIDEO_SEGMENT_SECONDS)))
            total = info["frame_count"] or None
            n_passes = 2 if has_video_alpha else 1

            def make_progress(pass_index):
                if not progress_cb:
                    return None
                def cb(i):
                    if total:
                        progress_cb(pass_index * total + i, total * n_passes)
                    else:
                        progress_cb(i, None)
                return cb

            color_path = os.path.join(tmp, "color.mp4")
            init_bytes, segments, segment_blobs = _encode_fragmented_av1(
                ffmpeg, input_path, color_path, crf, speed, gop, fps, vf=None,
                progress_cb=make_progress(0), cancel_event=cancel_event,
                video_index=info["video_stream_index"],
                audio_indices=info["audio_stream_indices"],
                subtitle_indices=info["subtitle_stream_indices"])

            alpha_init = alpha_segments = alpha_blobs = None
            if has_video_alpha:
                # Neither AV1 nor MP4 has a standard alpha-channel
                # convention (the same gap this whole feature exists to
                # close -- see the module docstring). So the alpha plane
                # is pulled out with FFmpeg's `alphaextract` filter and
                # encoded as its own grayscale AV1 track, segmented and
                # CRC32-checked exactly like the color track, and
                # recombined at decode time (see _decode_video_alpha /
                # the viewer's alpha-compositing playback path). No audio
                # or subtitles here -- just the one grayscale video stream.
                alpha_path = os.path.join(tmp, "alpha.mp4")
                alpha_init, alpha_segments, alpha_blobs = _encode_fragmented_av1(
                    ffmpeg, input_path, alpha_path, crf, speed, gop, fps,
                    vf="alphaextract", codec="libaom-av1",
                    progress_cb=make_progress(1), cancel_event=cancel_event,
                    video_index=info["video_stream_index"])

            thumb_path = os.path.join(tmp, "thumb.png")
            thumb_cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                         "-i", input_path, "-vframes", "1"]
            if has_video_alpha:
                thumb_cmd += ["-pix_fmt", "rgba"]
            subprocess.run(thumb_cmd + [thumb_path], check=False)
            thumb_mode = "RGBA" if has_video_alpha else "RGB"
            if os.path.exists(thumb_path):
                thumb_img = Image.open(thumb_path).convert(thumb_mode)
            else:
                fill = (40, 40, 40, 255) if has_video_alpha else (40, 40, 40)
                thumb_img = Image.new(thumb_mode, (info["width"], info["height"]), fill)
            tw, th, entries = _make_thumbnail_entries(thumb_img, self.thumb_max, self.quality)

        width, height = info["width"], info["height"]
        has_audio = info["has_audio"]

        index_offset = HEADER_SIZE
        frame_table_offset = index_offset + len(entries) * ENTRY_SIZE
        segment_table_offset = frame_table_offset  # no frame-table rows for video
        alpha_segment_table_offset = segment_table_offset + len(segments) * SEGMENT_SIZE
        n_alpha_segments = len(alpha_segments) if has_video_alpha else 0
        data_offset = alpha_segment_table_offset + n_alpha_segments * SEGMENT_SIZE
        running = data_offset
        for e in entries:
            e.offset = running
            running += e.length
        media_blob_offset = running
        media_blob_length = len(init_bytes) + sum(length for length, _ in segments)
        running += media_blob_length

        if has_video_alpha:
            alpha_media_blob_offset = running
            alpha_media_blob_length = len(alpha_init) + sum(length for length, _ in alpha_segments)
            running += alpha_media_blob_length
        else:
            alpha_media_blob_offset = 0
            alpha_media_blob_length = 0

        metadata_blob = _pack_metadata(info["tags"], None)
        metadata_offset = running
        running += len(metadata_blob)

        header = struct.pack(
            HEADER_FMT, MAGIC, VERSION, width, height, 8,
            3, CONTENT_VIDEO, 0, 0, 0, 1 if has_video_alpha else 0, 0,
            1, info["frame_count"], int(round(out_fps * 100)), len(entries),
            tw, th, HEADER_SIZE, index_offset, frame_table_offset, data_offset,
            1 if has_audio else 0, media_blob_offset, media_blob_length,
            len(init_bytes), zlib.crc32(init_bytes) & 0xFFFFFFFF,
            len(segments), segment_table_offset,
            alpha_media_blob_offset, alpha_media_blob_length,
            len(alpha_init) if has_video_alpha else 0,
            (zlib.crc32(alpha_init) & 0xFFFFFFFF) if has_video_alpha else 0,
            n_alpha_segments, alpha_segment_table_offset,
            metadata_offset, len(metadata_blob),
        )

        with open(output_path, "wb") as f:
            f.write(header)
            for e in entries:
                f.write(e.pack())
            for length, crc in segments:
                f.write(struct.pack(SEGMENT_FMT, length, crc))
            if has_video_alpha:
                for length, crc in alpha_segments:
                    f.write(struct.pack(SEGMENT_FMT, length, crc))
            # no frame-table rows to write
            for e in entries:
                f.write(e.payload)
            f.write(init_bytes)
            for s in segment_blobs:
                f.write(s)
            if has_video_alpha:
                f.write(alpha_init)
                for s in alpha_blobs:
                    f.write(s)
            f.write(metadata_blob)

        self.last_video_backend = "ffmpeg/libsvtav1+libopus" if has_audio else "ffmpeg/libsvtav1"
        return {"width": width, "height": height, "frame_count": info["frame_count"],
                "fps": out_fps, "has_audio": has_audio, "has_alpha": has_video_alpha,
                "has_subtitles": info["has_subtitles"], "size": running,
                "segments": len(segments),
                "audio_tracks": len(info["audio_stream_indices"]),
                "subtitle_tracks": len(info["subtitle_stream_indices"])}

    def encode_audio(self, input_path, output_path, bitrate=None,
                      progress_cb=None, cancel_event=None):
        """Encode audio via FFmpeg into Opus (CONTENT_AUDIO) -- deliberately
        not a new storage scheme of its own: it reuses video's entire
        segmented-media-blob mechanism (fragmented MP4, per-segment
        CRC32, a corrupt/missing segment dropped rather than failing the
        whole file -- see the module docstring's "Container layout for a
        video" section and extract_media) wholesale, just without a
        picture track. Opus is royalty-free and already a hard FFmpeg
        dependency here regardless, since video's own audio track uses
        it too. bitrate: Opus bitrate string FFmpeg understands (e.g.
        "128k"), or None to auto-pick one from the source's own bitrate
        (see _default_audio_bitrate) -- Opus is very efficient at spoken-
        word/podcast content well below whatever gets picked; music
        generally wants it. cancel_event: same contract as encode_video's.

        If the source carries embedded cover art (ID3 APIC, a FLAC
        picture block, ...), FFmpeg exposes it as an attached "video"
        stream, so it's pulled out the same way a video's poster frame
        is grabbed, and stored as this file's thumbnail through the
        exact same _make_thumbnail_entries path images and video use --
        a `view` of an audio .fmff shows the album art instead of a
        generic placeholder for free. Falls back to a plain placeholder,
        same as encode_video does, when there isn't one."""
        ffmpeg = _find_ffmpeg()
        if ffmpeg is None:
            raise RuntimeError(
                "audio encoding needs FFmpeg (with libopus) on PATH -- "
                "install it, e.g. `winget install BtbN.FFmpeg.LGPL.8.1` on Windows")
        ffprobe = _find_ffprobe()
        if ffprobe is None:
            raise RuntimeError("audio encoding needs ffprobe alongside ffmpeg to read "
                                "source audio metadata (it ships in the same FFmpeg build)")

        input_path = str(input_path)
        info = _ffprobe_info(input_path, ffprobe)
        if not info["has_audio"]:
            raise ValueError(f"no audio stream found in {input_path}")
        if bitrate is None:
            bitrate = _default_audio_bitrate(info["bit_rate"])

        with tempfile.TemporaryDirectory(prefix="fmff_enc_") as tmp:
            blob_path = os.path.join(tmp, "audio.mp4")
            init_bytes, segments, segment_blobs = _encode_fragmented_audio(
                ffmpeg, input_path, blob_path, bitrate, progress_cb, cancel_event)

            # Most audio sources have no embedded cover art at all, so
            # ffmpeg reporting "no stream"/"Invalid argument" here is the
            # expected, already-handled-below common case, not a real
            # problem -- stderr is discarded so that expected failure
            # doesn't print scary-looking noise for every single ordinary
            # track encoded (it was, before this: check=False already
            # meant the *return code* was never treated as an error, but
            # nothing stopped ffmpeg's own stderr from reaching the
            # console regardless of whether the failure was expected).
            cover_path = os.path.join(tmp, "cover.jpg")
            subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                             "-i", input_path, "-an", "-c:v", "copy", "-frames:v", "1",
                             cover_path], check=False, stderr=subprocess.DEVNULL)
            if os.path.exists(cover_path) and os.path.getsize(cover_path) > 0:
                thumb_img = Image.open(cover_path).convert("RGB")
            else:
                thumb_img = Image.new("RGB", (256, 256), (40, 40, 40))
            tw, th, entries = _make_thumbnail_entries(thumb_img, self.thumb_max, self.quality)

        index_offset = HEADER_SIZE
        frame_table_offset = index_offset + len(entries) * ENTRY_SIZE
        segment_table_offset = frame_table_offset  # no frame-table rows for audio
        data_offset = segment_table_offset + len(segments) * SEGMENT_SIZE
        running = data_offset
        for e in entries:
            e.offset = running
            running += e.length
        media_blob_offset = running
        media_blob_length = len(init_bytes) + sum(length for length, _ in segments)
        running += media_blob_length

        metadata_blob = _pack_metadata(info["tags"], None)
        metadata_offset = running
        running += len(metadata_blob)

        # See HEADER_FMT's comment for what these repurposed fields mean
        # for CONTENT_AUDIO: is_video=1 just means "read a segment table,
        # not a tile index" (true for audio too), frame_count holds
        # duration_ms, fps_x100 holds the sample rate.
        header = struct.pack(
            HEADER_FMT, MAGIC, VERSION, 0, 0, 0,
            info["channels"], CONTENT_AUDIO, 0, 0, 0, 0, 0,
            1, info["duration_ms"], info["sample_rate"], len(entries),
            tw, th, HEADER_SIZE, index_offset, frame_table_offset, data_offset,
            1, media_blob_offset, media_blob_length,
            len(init_bytes), zlib.crc32(init_bytes) & 0xFFFFFFFF,
            len(segments), segment_table_offset,
            0, 0, 0, 0, 0, 0,
            metadata_offset, len(metadata_blob),
        )

        with open(output_path, "wb") as f:
            f.write(header)
            for e in entries:
                f.write(e.pack())
            for length, crc in segments:
                f.write(struct.pack(SEGMENT_FMT, length, crc))
            # no frame-table rows to write
            for e in entries:
                f.write(e.payload)
            f.write(init_bytes)
            for s in segment_blobs:
                f.write(s)
            f.write(metadata_blob)

        return {"duration_ms": info["duration_ms"], "sample_rate": info["sample_rate"],
                "channels": info["channels"], "size": running, "segments": len(segments),
                "bitrate": bitrate, "source_bit_rate": info["bit_rate"], "tags": info["tags"]}


# --------------------------------------------------------------------- decoder

class FMFFDecoder:
    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            raw = f.read(HEADER_SIZE)
            if len(raw) < HEADER_SIZE:
                raise ValueError("file is truncated -- shorter than the fixed header alone")
            (magic, version, width, height, bit_depth, channels, color_space,
             tile_size, tiles_x, tiles_y, has_alpha, quality,
             is_video, frame_count, fps_x100, num_entries,
             thumb_w, thumb_h, header_size, index_offset,
             frame_table_offset, data_offset,
             has_audio, media_blob_offset, media_blob_length,
             init_segment_length, init_segment_crc32,
             segment_count, segment_table_offset,
             alpha_media_blob_offset, alpha_media_blob_length,
             alpha_init_segment_length, alpha_init_segment_crc32,
             alpha_segment_count, alpha_segment_table_offset,
             metadata_offset, metadata_length) = struct.unpack(HEADER_FMT, raw)
            if magic != MAGIC:
                raise ValueError("not an FMFF file")
            if version != VERSION:
                raise ValueError(
                    f"unsupported FMFF version {version} -- this build reads version "
                    f"{VERSION} only (the file needs a matching reader, or re-encoding)")
            f.seek(index_offset)
            idx_raw = f.read(num_entries * ENTRY_SIZE)
            f.seek(frame_table_offset)
            ft_raw = f.read(frame_count * FRAME_SIZE) if not is_video else b""
            f.seek(segment_table_offset)
            seg_raw = f.read(segment_count * SEGMENT_SIZE) if is_video else b""
            f.seek(alpha_segment_table_offset)
            alpha_seg_raw = (f.read(alpha_segment_count * SEGMENT_SIZE)
                              if is_video and has_alpha else b"")
            f.seek(metadata_offset)
            metadata_raw = f.read(metadata_length)

        self.width, self.height = width, height
        self.has_alpha = bool(has_alpha)
        self.tile_size = tile_size
        self.tiles_x, self.tiles_y = tiles_x, tiles_y
        self.quality = quality
        self.thumb_w, self.thumb_h = thumb_w, thumb_h
        self.version = version
        self.content_mode = color_space
        # encode_image always writes a real (>0) tile_size; encode_image_sequence
        # always writes 0 regardless of frame count, since it never uses a
        # fixed grid at all (see its docstring) -- this is the actual,
        # robust signal for which addressing mode a CONTENT_IMAGE file's
        # LAYER_FULL entries use (see full()/full_sequence()).
        self.has_fixed_tile_grid = color_space == CONTENT_IMAGE and tile_size > 0
        # The wire `is_video` flag really means "read a segment table
        # instead of a tile index" -- true for CONTENT_AUDIO too (see
        # encode_audio and HEADER_FMT's comment), so is_video/is_audio
        # here narrow it back down to what each name actually implies.
        self.is_video = bool(is_video) and color_space == CONTENT_VIDEO
        self.is_audio = bool(is_video) and color_space == CONTENT_AUDIO
        self.is_jpeg_passthrough = (color_space == CONTENT_JPEG_PASSTHROUGH)
        self.is_document = (color_space == CONTENT_DOCUMENT)
        if self.is_audio:
            # frame_count/fps_x100 hold duration_ms/sample_rate instead
            # for audio (see HEADER_FMT's comment) -- frame_count/fps
            # themselves stay meaningless zeros rather than repurposed,
            # since nothing should be reading "frames" or "fps" from an
            # audio file to begin with.
            self.frame_count = 0
            self.fps = 0.0
            self.duration_ms = frame_count
            self.sample_rate = fps_x100
            self.channels = channels
        else:
            self.frame_count = frame_count
            self.fps = fps_x100 / 100.0
            self.duration_ms = 0
            self.sample_rate = 0
            self.channels = channels if color_space == CONTENT_IMAGE else 0
        self.has_audio = bool(has_audio)
        self.last_corrupt_tiles = 0
        self.last_corrupt_segments = 0
        self.last_corrupt_alpha_segments = 0
        self.media_blob_offset = media_blob_offset
        self.media_blob_length = media_blob_length
        self.init_segment_length = init_segment_length
        self.init_segment_crc32 = init_segment_crc32
        self.segment_count = segment_count
        self.segment_table_offset = segment_table_offset
        # For CONTENT_IMAGE, has_alpha means "per-pixel RGBA tiles" (used
        # by full()); for CONTENT_VIDEO it means "there's a second,
        # grayscale AV1 track carrying the alpha plane" -- same header
        # byte, different meaning depending on content_mode, exactly like
        # content_mode itself reuses the old color_space byte. This flag
        # spells out the video-specific meaning so video code doesn't have
        # to reason about which sense of has_alpha applies.
        self.has_video_alpha = self.is_video and bool(has_alpha)
        self.alpha_media_blob_offset = alpha_media_blob_offset
        self.alpha_media_blob_length = alpha_media_blob_length
        self.alpha_init_segment_length = alpha_init_segment_length
        self.alpha_init_segment_crc32 = alpha_init_segment_crc32
        self.alpha_segment_count = alpha_segment_count
        self.alpha_segment_table_offset = alpha_segment_table_offset
        # ({}, None, None, []) for any file predating VERSION 13, or one
        # whose source simply had no tags/EXIF/ICC profile/version history
        # to carry over -- see _pack_metadata/_unpack_metadata.
        self.tags, self.exif_bytes, self.icc_bytes, self.versions_meta = _unpack_metadata(metadata_raw)

        # A truncated/still-downloading file can cut the index table off
        # mid-entry -- drop only that dangling partial entry rather than
        # raising, so whatever tiles *did* arrive can still be decoded.
        self.entries = [Entry.unpack(idx_raw[i:i + ENTRY_SIZE])
                         for i in range(0, len(idx_raw) - ENTRY_SIZE + 1, ENTRY_SIZE)]
        self.frame_table = [FrameEntry(*struct.unpack(FRAME_FMT, ft_raw[i:i + FRAME_SIZE]))
                             for i in range(0, len(ft_raw) - FRAME_SIZE + 1, FRAME_SIZE)]
        # same truncation tolerance as the tile index above: a dangling
        # partial segment row at the end of a cut-off file is just dropped.
        self.segments = [struct.unpack(SEGMENT_FMT, seg_raw[i:i + SEGMENT_SIZE])
                          for i in range(0, len(seg_raw) - SEGMENT_SIZE + 1, SEGMENT_SIZE)]
        self.alpha_segments = [struct.unpack(SEGMENT_FMT, alpha_seg_raw[i:i + SEGMENT_SIZE])
                                for i in range(0, len(alpha_seg_raw) - SEGMENT_SIZE + 1, SEGMENT_SIZE)]

    def _read(self, entry):
        with open(self.path, "rb") as f:
            f.seek(entry.offset)
            data = f.read(entry.length)
        # A short read (truncated/mid-download file) or genuine bit rot both
        # show up here as a CRC mismatch -- one check covers both cases, and
        # catches it before wasting time on a doomed decompress attempt.
        if len(data) != entry.length or (zlib.crc32(data) & 0xFFFFFFFF) != entry.crc32:
            raise CorruptTileError(entry)
        return data

    def _decode_entry(self, entry):
        data = self._read(entry)
        try:
            return _decode_tile_bytes(data, entry, self.quality)
        except (zlib.error, OSError, ValueError) as exc:
            # CRC matched but the payload still doesn't decompress/reshape
            # cleanly -- extremely unlikely, but fail the same clean way.
            raise CorruptTileError(entry) from exc

    def thumbnail(self):
        """The small instant preview -- present for both images and video
        (a poster frame), always FMFF's own lossless tile codec."""
        color = alpha = None
        for e in self.entries:
            if e.layer != LAYER_THUMB:
                continue
            try:
                if e.plane == PLANE_COLOR:
                    color = self._decode_entry(e)
                else:
                    alpha = self._decode_entry(e)
            except CorruptTileError:
                if e.plane == PLANE_COLOR:
                    color = _broken_tile_fill(e.h, e.w, 3)
                else:
                    alpha = np.full((e.h, e.w), 255, dtype=np.uint8)
        if color is None:
            color = _broken_tile_fill(self.thumb_h or 1, self.thumb_w or 1, 3)
        if alpha is not None:
            return Image.fromarray(np.dstack([color, alpha]), "RGBA")
        return Image.fromarray(color, "RGB")

    def tile_byte_ranges(self, layer=LAYER_FULL):
        """(offset, length) pairs for every *image* tile payload (color
        and alpha, never a PLANE_MASK entry -- see add_mask) at the given
        layer -- images only. Suitable for HTTP Range requests, so a
        client can fetch a single tile without touching the rest of the
        file; also what the viewer's progressive-load tile counter counts
        against, which must agree with what full() actually blits (see
        its own PLANE_MASK exclusion) or the counter would never reach
        its total on a file with a version-0 mask."""
        return [(e.offset, e.length) for e in self.entries
                if e.layer == layer and e.plane != PLANE_MASK]

    @property
    def is_animated(self):
        """True for a CONTENT_IMAGE file with more than one frame -- see
        encode_image_sequence. Unrelated to is_video/is_audio: this is
        FMFF's own tile codec repeated across frames (Entry.frame +
        frame_table), not the segmented-media-blob path both of those use."""
        return self.content_mode == CONTENT_IMAGE and len(self.frame_table) > 1

    def frame_durations_ms(self):
        """Per-frame display duration in ms, derived from frame_table's
        cumulative start timestamps (see encode_image_sequence). The
        last frame has no "next" timestamp to diff against, so it just
        reuses the previous frame's duration (or a flat 100ms if there's
        only one frame) -- a minor approximation, not worth a whole
        extra stored field for."""
        starts = [f.timestamp_ms for f in self.frame_table]
        if len(starts) < 2:
            return [100] * len(starts)
        durations = [b - a for a, b in zip(starts, starts[1:])]
        durations.append(durations[-1])
        return durations

    def list_versions(self):
        """[{"index", "name", "note", "added", "size", "has_mask",
        "mask_size"}, ...], sorted by index -- one entry per version that
        exists, labeled from self.versions_meta where a name/note/
        timestamp was ever recorded and falling back to a plain default
        ("original" for 0, "version N" otherwise) where it wasn't. "size"
        is that version's own *incremental* cost -- the bytes of the
        color/alpha tiles it actually changed relative to the version
        before it (see add_version), zero for a version that changed no
        pixels at all (e.g. one added only to attach a note); mask bytes
        (see add_mask) are counted separately in "mask_size" rather than
        folded into "size", since a mask isn't part of the image itself.
        Version indices come from self.versions_meta (recorded even for
        a zero-changed-tiles version, which stores no LAYER_VERSION
        entries at all) unioned with whatever LAYER_VERSION frame numbers
        actually appear in the index, so neither source missing an index
        drops it. Always at least one entry (index 0) for a still,
        non-animated CONTENT_IMAGE file -- every such file has its
        LAYER_FULL tiles whether or not add_version has ever touched it;
        empty for any other content type or an animated file, neither of
        which supports versions (see add_version)."""
        if self.content_mode != CONTENT_IMAGE or self.is_animated:
            return []
        by_index = {v["index"]: v for v in self.versions_meta if "index" in v}
        indices = ({0} | {e.frame for e in self.entries if e.layer == LAYER_VERSION}
                   | set(by_index))
        out = []
        for idx in sorted(indices):
            meta = by_index.get(idx, {})
            layer = LAYER_FULL if idx == 0 else LAYER_VERSION
            version_entries = [e for e in self.entries if e.layer == layer and e.frame == idx]
            default_size = sum(e.length for e in version_entries if e.plane != PLANE_MASK)
            mask_size = sum(e.length for e in version_entries if e.plane == PLANE_MASK)
            default_name = "original" if idx == 0 else f"version {idx}"
            out.append({
                "index": idx,
                "name": meta.get("name") or default_name,
                "note": meta.get("note") or "",
                "added": meta.get("added") or "",
                "size": meta.get("size", default_size),
                "has_mask": mask_size > 0,
                "mask_size": mask_size,
            })
        return out

    def extract_mask(self, version=0, progress_cb=None):
        """Decode the selection mask attached to one version (see
        FMFFEncoder.add_mask), as a grayscale ("L") Image -- raises
        ValueError if that version has no mask.

        Unlike full()'s color/alpha reconstruction, this never falls
        back to an earlier version's mask: a mask belongs to exactly the
        version it was attached to, with no inheritance across the
        version chain (see add_mask's own docstring for why)."""
        if self.content_mode != CONTENT_IMAGE or self.is_animated or not self.has_fixed_tile_grid:
            raise ValueError(f"{self.path} has no versions/masks to extract from")
        layer = LAYER_FULL if version == 0 else LAYER_VERSION
        tile_entries = sorted(
            (e for e in self.entries if e.layer == layer and e.frame == version
             and e.plane == PLANE_MASK),
            key=lambda e: (e.ty, e.tx))
        if not tile_entries:
            raise ValueError(f"version {version} has no mask attached (see add-mask)")
        canvas = np.zeros((self.height, self.width), dtype=np.uint8)
        ts = self.tile_size
        for e in tile_entries:
            y0, x0 = e.ty * ts, e.tx * ts
            try:
                decoded = self._decode_entry(e)
            except CorruptTileError:
                decoded = _broken_tile_fill(e.h, e.w, 1)
            canvas[y0:y0 + e.h, x0:x0 + e.w] = decoded
            if progress_cb:
                progress_cb(e, y0, x0, decoded)
        return Image.fromarray(canvas, "L")

    def _nearest_full_version(self, version):
        """The largest version index <= `version` whose own PLANE_COLOR
        tiles cover every position in the tile grid -- 0 if none does
        (version 0's LAYER_FULL tiles always cover the whole grid, so
        that's always a safe fallback). Shared by full() (to bound how
        far back a decode has to replay) and add_version (to bound how
        far back its own change-comparison has to replay) -- see
        VERSION_SNAPSHOT_INTERVAL for why a later version can also
        qualify: a periodic full version is deliberately built to cover
        every tile, but an ordinary delta version that happens to touch
        every tile qualifies too, just as validly, by construction."""
        if version <= 0:
            return 0
        total_tiles = self.tiles_x * self.tiles_y
        color_counts = {}
        for e in self.entries:
            if e.layer == LAYER_VERSION and e.frame <= version and e.plane == PLANE_COLOR:
                color_counts[e.frame] = color_counts.get(e.frame, 0) + 1
        for v in range(version, 0, -1):
            if color_counts.get(v, 0) == total_tiles:
                return v
        return 0

    def full(self, progress_cb=None, frame=0, version=0):
        """Decode one frame of a still image (frame 0 for an ordinary
        single-frame file) -- images only, video has no per-tile data to
        assemble, see extract_media(). Dispatches to the JPEG-passthrough
        path automatically when that's how this file was encoded (which
        is never animated, so `frame` doesn't apply there);
        progress_cb is a no-op there since it never runs tile-by-tile.

        Good for random access to a single frame of a file that uses a
        fixed tile grid (see has_fixed_tile_grid) -- an ordinary still
        image, always. A file without one -- an actual multi-frame
        animation -- has LAYER_FULL entries that carry literal pixel
        (x, y) rectangles rather than tile-grid indices (see encode_image_sequence), so
        there's no per-position key to look up "whichever frame at or
        before N last touched this spot" the way a fixed grid allows --
        reconstructing frame N means replaying frames 0..N in order,
        which this delegates to full_sequence() for. Decoding *every*
        frame of a no-fixed-grid file should call full_sequence()
        directly instead of looping this: each call here would redo that
        whole 0..N replay from scratch.

        `version` (see add_version/list_versions) picks which named
        version to decode instead of `frame` -- the two never apply to
        the same file (a version is only ever added to a non-animated
        file, see add_version), so there's no ambiguity between them.
        version=0, the default, is the LAYER_FULL tiles every still
        image already had before this existed -- identical output to
        calling full() with no arguments at all on any file, versioned
        or not.

        A version > 0 only ever stores the tiles that changed relative
        to the version before it (see add_version), so reconstructing it
        in principle means starting from version 0's full canvas and
        replaying every version 1..version's own changed tiles on top,
        in order -- the same "start from a base, overlay each step's
        changes" shape full_sequence() already uses for animation
        frames, just keyed by version number instead of frame number.
        In practice this starts from the *nearest* version at or before
        `version` that happens to cover every tile (see
        _nearest_full_version and VERSION_SNAPSHOT_INTERVAL) instead of
        always version 0, so replay depth stays bounded by how often a
        full version occurs, not by `version` itself -- decoding version
        1000 of a long chain doesn't replay 1000 versions, only back to
        the nearest one that's a full snapshot. Alpha is a whole-file
        property (see add_version), so alpha_canvas is created once from
        self.has_alpha and shared across every step, not recomputed per
        version."""
        if self.is_jpeg_passthrough:
            return self._decode_jpeg_passthrough()
        if not self.has_fixed_tile_grid:
            return self.full_sequence()[frame]
        if version != 0:
            valid = {v["index"] for v in self.list_versions()}
            if version not in valid:
                raise ValueError(f"version {version} not found in {self.path} "
                                  f"(available: {sorted(valid)})")
        # Gray, not black, so any region a corrupt/truncated file never
        # covers at all (not just a tile that failed its own CRC) still
        # reads as "missing" rather than looking like real black content.
        canvas = np.full((self.height, self.width, 3), 128, dtype=np.uint8)
        alpha_canvas = np.full((self.height, self.width), 255, dtype=np.uint8) if self.has_alpha else None
        self.last_corrupt_tiles = 0
        ts = self.tile_size

        def _blit(step_entries):
            # Large batches (a real photo's worth of tiles) go through
            # the same multi-core pool encode_image already uses, not a
            # sequential Python loop -- profiling a real ~12MP photo
            # showed decoding it took 1.6s+ entirely on one core (see
            # _decode_tile_task's own docstring), which the viewer feels
            # as a visible pause between the window opening and the
            # picture actually appearing. imap_unordered rather than
            # map(): results stream back as each tile finishes instead
            # of only after the whole batch does, so progress_cb still
            # gets called tile-by-tile for the viewer's progressive
            # reveal -- just in whatever order workers finish them in,
            # not strictly top-to-bottom anymore.
            step_entries = sorted(step_entries, key=lambda e: (e.ty, e.tx, e.plane))
            if not step_entries:
                return
            jobs = [(self.path, e, self.quality) for e in step_entries]
            if len(step_entries) >= _MP_TILE_THRESHOLD:
                # imap_unordered defaults to chunksize=1 -- one IPC round
                # trip per tile, unlike map()'s own auto-computed batching
                # -- which measured dramatically slower here (sending
                # ~3000 tiles one at a time ate most of the parallel
                # speedup in IPC overhead). Matching map()'s own rule of
                # thumb (roughly 4 chunks per worker) instead recovered it.
                chunksize = max(1, len(jobs) // (8 * 4))
                results = _get_tile_pool().imap_unordered(_decode_tile_task, jobs, chunksize=chunksize)
            else:
                results = (_decode_tile_task(job) for job in jobs)
            for e, decoded in results:
                y0, x0 = e.ty * ts, e.tx * ts
                if decoded is None:
                    self.last_corrupt_tiles += 1
                    decoded = (_broken_tile_fill(e.h, e.w, 1) if e.plane == PLANE_ALPHA
                               else _broken_tile_fill(e.h, e.w, 3))
                if e.plane == PLANE_ALPHA:
                    alpha_canvas[y0:y0 + e.h, x0:x0 + e.w] = decoded
                else:
                    canvas[y0:y0 + e.h, x0:x0 + e.w, :] = decoded
                if progress_cb:
                    progress_cb(e, y0, x0, decoded)

        # PLANE_MASK entries (see add_mask) share the same (layer, frame)
        # bucket as color/alpha but must never be composited into the
        # image itself -- excluded here, not just left to _blit's own
        # plane check, since that check only knows "alpha or not".
        start = self._nearest_full_version(version) if version else 0
        if start == 0:
            _blit(e for e in self.entries
                  if e.layer == LAYER_FULL and e.frame == 0 and e.plane != PLANE_MASK)
        else:
            _blit(e for e in self.entries
                  if e.layer == LAYER_VERSION and e.frame == start and e.plane != PLANE_MASK)
        if version > start:
            by_version = {}
            for e in self.entries:
                if e.layer == LAYER_VERSION and start < e.frame <= version and e.plane != PLANE_MASK:
                    by_version.setdefault(e.frame, []).append(e)
            for v in range(start + 1, version + 1):
                _blit(by_version.get(v, []))

        if alpha_canvas is not None:
            return Image.fromarray(np.dstack([canvas, alpha_canvas]), "RGBA")
        return Image.fromarray(canvas, "RGB")

    def full_sequence(self, progress_cb=None):
        """Decode every frame of an animated image in one linear pass --
        the efficient counterpart to calling full(frame=i) in a loop
        (full(frame=i) on an animated file just calls this and indexes
        into the result, so a loop over all frames would redecode the
        whole animation once per frame: O(F^2), worse than it sounds).

        A frame identical to the previous one gets no entry at all (see
        encode_image_sequence), and a changed frame's entry is a single
        rectangle positioned at literal pixel (x, y) -- not a tile-grid
        cell the way a still image's entries are, since an animated file
        has no fixed grid. So this walks frames in order, bucketing
        entries by the frame they belong to ahead of time (O(total
        entries) once, not once per frame) and blitting each frame's
        rectangle (if it has one) directly onto its pixel position on a
        running canvas left over from the previous frame -- a tile
        position is never looked up because there isn't one; "reuse the
        previous frame's content" falls out for free from simply not
        touching whatever the rectangle doesn't cover. Total cost is
        O(total_entries) for the blit work plus O(F * width * height) to
        materialize each frame's image, i.e. linear in both frame count
        and entry count instead of their product."""
        if self.is_jpeg_passthrough:
            return [self._decode_jpeg_passthrough()]
        n_frames = max(1, len(self.frame_table))
        canvas = np.full((self.height, self.width, 3), 128, dtype=np.uint8)
        alpha_canvas = np.full((self.height, self.width), 255, dtype=np.uint8) if self.has_alpha else None
        self.last_corrupt_tiles = 0

        by_frame = [[] for _ in range(n_frames)]
        for e in self.entries:
            if e.layer == LAYER_FULL and e.frame < n_frames:
                by_frame[e.frame].append(e)

        # A file without a fixed tile grid carries literal pixel (x, y)
        # entries; one with a fixed grid carries tile-grid indices that
        # need scaling by tile_size -- see encode_image vs
        # encode_image_sequence.
        literal_coords = not self.has_fixed_tile_grid
        ts = self.tile_size
        frames_out = []
        for frame in range(n_frames):
            for e in sorted(by_frame[frame], key=lambda e: (e.ty, e.tx, e.plane)):
                y0, x0 = (e.ty, e.tx) if literal_coords else (e.ty * ts, e.tx * ts)
                try:
                    decoded = self._decode_entry(e)
                except CorruptTileError:
                    self.last_corrupt_tiles += 1
                    decoded = (_broken_tile_fill(e.h, e.w, 1) if e.plane == PLANE_ALPHA
                               else _broken_tile_fill(e.h, e.w, 3))
                if e.plane == PLANE_ALPHA:
                    alpha_canvas[y0:y0 + e.h, x0:x0 + e.w] = decoded
                else:
                    canvas[y0:y0 + e.h, x0:x0 + e.w, :] = decoded
            if alpha_canvas is not None:
                frames_out.append(Image.fromarray(np.dstack([canvas, alpha_canvas]).copy(), "RGBA"))
            else:
                frames_out.append(Image.fromarray(canvas.copy(), "RGB"))
            if progress_cb:
                progress_cb(frame, n_frames)
        return frames_out

    def extract_document(self, out_path=None):
        """Reconstruct the original PDF/.txt bytes exactly -- see
        FMFFEncoder.encode_document. The document counterpart to
        extract_media, using the same single-leading-tag-byte scheme
        _decode_jpeg_passthrough does (b"R" = the original bytes
        verbatim; one of _JPEG_COEFF_CODECS' own tags = which
        general-purpose compressor was raced and won). Returns the
        bytes either way; writes them to out_path too when given.
        Needs no rendering dependency (PyMuPDF, a font) at all -- unlike
        a thumbnail/preview, the stored bytes themselves never went
        through rendering to begin with."""
        with open(self.path, "rb") as f:
            f.seek(self.media_blob_offset)
            blob = f.read(self.media_blob_length)
        tag, payload = blob[:1], blob[1:]
        data = payload if tag == b"R" else _JPEG_COEFF_CODECS[tag][1](payload)
        if out_path is not None:
            Path(out_path).write_bytes(data)
        return data

    def _decode_jpeg_passthrough(self):
        """Reconstruct pixels from the JPEG's own coefficients (see
        FMFFEncoder.encode_image_jpeg_passthrough) using FMFF's own
        dequantize/IDCT/color-conversion math -- no jpeglib needed to
        *read* one of these files, only to have made it. Or, if
        recompression didn't win at encode time, the blob is just the
        original JPEG bytes verbatim -- Pillow decodes those directly."""
        with open(self.path, "rb") as f:
            f.seek(self.media_blob_offset)
            blob = f.read(self.media_blob_length)

        if blob[:1] == b"R":
            return Image.open(io.BytesIO(blob[1:])).convert("RGB")

        meta_len = struct.unpack_from("<I", blob, 1)[0]
        meta = json.loads(blob[5:5 + meta_len])
        tag = blob[5 + meta_len:6 + meta_len]
        payload = blob[6 + meta_len:]
        raw = _JPEG_COEFF_CODECS[tag][1](payload)

        shapes = meta["shapes"]
        qt = np.array(meta["qt"], dtype=np.float32)
        quant_tbl_no = meta["quant_tbl_no"]
        samp = meta["samp_factor"]
        max_h = max(s[0] for s in samp)
        max_v = max(s[1] for s in samp)

        elem_offset = 0
        planes = []
        for ci, (nby, nbx) in enumerate(shapes):
            count = nby * nbx * 64
            coeffs = np.frombuffer(raw, dtype=np.int16, count=count,
                                    offset=elem_offset * 2).reshape(nby, nbx, 8, 8)
            elem_offset += count
            coeffs = _dc_delta_decode(coeffs)
            deq = coeffs.astype(np.float32) * qt[quant_tbl_no[ci]]
            # See lossy_encode's forward transform for why this is @
            # instead of einsum -- same T.T @ block @ T per block, ~19x
            # faster for the same result.
            blocks = _T.T @ deq @ _T
            plane = _from_blocks(blocks, nby * 8, nbx * 8) + 128.0
            h_ratio, v_ratio = max_h // samp[ci][0], max_v // samp[ci][1]
            if h_ratio > 1 or v_ratio > 1:
                # Bilinear, not nearest-neighbor: subsampled chroma has to
                # be *interpolated* back up by any decoder (that detail is
                # genuinely gone, same as it would be for libjpeg's own
                # decode) -- nearest-neighbor produced blocky steps right
                # at sharp edges; bilinear is the standard choice and
                # matches libjpeg's own default "fancy upsampling" closely.
                ph, pw = plane.shape
                plane = np.array(Image.fromarray(plane.astype(np.float32), mode="F").resize(
                    (pw * h_ratio, ph * v_ratio), Image.BILINEAR))
            planes.append(plane[:self.height, :self.width])

        rgb = _ycc_to_rgb(np.dstack(planes))
        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")

    def _reconstruct_segmented(self, blob_offset, init_length, init_crc32, segments, what):
        """Shared by extract_media/extract_alpha_media: read the init
        chunk plus each listed segment from self.path starting at
        blob_offset, dropping any segment that fails its CRC32 (or that a
        truncated file doesn't have), and return the concatenated bytes.
        Returns (data, corrupt_count). A corrupt init chunk always raises
        -- see extract_media's docstring for why that one case can't
        gracefully degrade."""
        corrupt = 0
        pieces = []
        with open(self.path, "rb") as f:
            f.seek(blob_offset)
            init_bytes = f.read(init_length)
            if len(init_bytes) != init_length or (zlib.crc32(init_bytes) & 0xFFFFFFFF) != init_crc32:
                raise ValueError(
                    f"{what} init segment is corrupt or missing -- this file can't be "
                    f"played at all (unlike a corrupt/missing later segment, this one "
                    f"isn't something FMFF can gracefully skip past)")
            pieces.append(init_bytes)
            offset = blob_offset + init_length
            for length, expected_crc in segments:
                f.seek(offset)
                chunk = f.read(length)
                offset += length
                if len(chunk) != length or (zlib.crc32(chunk) & 0xFFFFFFFF) != expected_crc:
                    corrupt += 1
                    continue
                pieces.append(chunk)
        return b"".join(pieces), corrupt

    def extract_media(self, out_path):
        """Reconstruct the embedded video or audio (video/audio only) as
        a standalone, playable file at out_path -- a fragmented MP4
        (AV1 video + Opus audio, or Opus-only for a CONTENT_AUDIO file --
        see encode_audio). Any segment that fails its CRC32 check, or
        that a truncated/still-downloading file simply doesn't have yet,
        is dropped rather than aborting the whole extraction: what's
        left is still a valid file, just missing however many seconds
        that segment covered -- see last_corrupt_segments, and the
        module docstring's "Container layout for a video" section. The
        one thing this can't route around is a corrupt init segment (it
        carries the codec setup every later segment depends on), which
        raises a clear error instead of silently producing an unplayable
        file."""
        data, self.last_corrupt_segments = self._reconstruct_segmented(
            self.media_blob_offset, self.init_segment_length,
            self.init_segment_crc32, self.segments,
            "audio" if self.is_audio else "video")
        with open(out_path, "wb") as f:
            f.write(data)
        return out_path

    def extract_alpha_media(self, out_path):
        """Same as extract_media, but for the second, grayscale AV1 track
        that carries the alpha plane on a video encoded from a source
        with transparency (has_video_alpha) -- see encode_video's alpha
        handling. Raises if this file has no alpha track at all."""
        if not self.has_video_alpha:
            raise ValueError("this video has no alpha track")
        data, self.last_corrupt_alpha_segments = self._reconstruct_segmented(
            self.alpha_media_blob_offset, self.alpha_init_segment_length,
            self.alpha_init_segment_crc32, self.alpha_segments, "video alpha")
        with open(out_path, "wb") as f:
            f.write(data)
        return out_path


# --------------------------------------------------------------------- viewer

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff", ".tif", ".ico"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".opus", ".ogg", ".m4a", ".aac", ".wma"}

DARK_BG = "#1a1a1a"
DARK_PANEL = "#242424"
DARK_FG = "#c8c8c8"
DARK_BTN = "#333333"

# The window icon (title bar + taskbar while running) -- a separate thing
# from the .exe file's own icon (set at build time via PyInstaller's
# --icon; see this file's packaging notes), which Tk knows nothing about
# and doesn't set for you. Embedded as base64 rather than loaded from a
# logo.png sitting next to this file: a frozen .exe has no such file next
# to it at runtime unless separately bundled with --add-data (a build-time
# flag to remember and keep in sync), so inlining it here means the same
# few lines work identically from source and from a build, with nothing
# that can go stale or go missing after packaging. 64x64 PNG -- Tk's
# PhotoImage decodes PNG natively (Tcl/Tk 8.6+), no Pillow needed for this.
_APP_ICON_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAABkklEQVR4nO2aTW7CMBCFh6pH4FQcoVlwrK6z4Ha5"
    "Q7tKFYUxGY9/PtPMJ7GBjP38eLYHicvX9+1HTswHLYAmDKAF0IQBtACaMIAWQBMG0AJowgBaAE0YQAugCQNoATSf"
    "1gfnaXl67/64VhVDYDZAo4cpnjlyakwGrAPuB9EmstSVUtOUogSkJj0yRntWG8tjfG5NkQGlbEXN02JOiidRqZrD"
    "W6AkxmuN5n6L7eOpqX4NWkRoz1A3Svc+oNXB6GXIRqhX/EUqG3AkYv/5CGl4aUANgSMtViOrEVrxLsJS1zP+Is4t"
    "ME/L38si4v64JhuTGs1USU0yAZYOrTWtfweIODvBVvs4dUhansmp2VKlFW51wHnSl1ujGtD6xNbGt85ZWrNnyEao"
    "J8UGjHq/W3kyYOT4H43jIbZASfG7x19kZwCxIDL+Ip23wIiJcRtQYzH0ty+yMYA4/UcgbgGR/G/nv8RfJBIgl/iz"
    "9MkJA2gBNGEALYAmDKAF0IQBtACaMIAWQBMG0AJowgBaAE0YQAugOb0Bv6a867+vHHyNAAAAAElFTkSuQmCC"
)


def _app_icon_photo():
    """A tk.PhotoImage of the app icon -- call this only after a Tk root
    exists (it needs one to attach the image to), and keep a reference to
    the result for as long as the window using it is alive (Tk drops an
    image with no live Python reference, which would otherwise make the
    icon quietly vanish sometime after this function returns)."""
    return tk.PhotoImage(data=_APP_ICON_PNG_B64)


def _fmt_size(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _fmt_saving(orig, new):
    """'64% smaller' / '3% bigger' -- the sign flips depending on which
    way it went (JPEG passthrough can occasionally end up a hair bigger,
    see encode_image_jpeg_passthrough)."""
    if not orig:
        return "n/a"
    pct = (1 - new / orig) * 100
    return f"{pct:.0f}% smaller" if pct >= 0 else f"{-pct:.0f}% bigger"


class BatchConvertWindow:
    """Drag-and-drop / folder batch conversion: queue up any number of
    images and videos (individually, or by dropping/picking whole
    folders, recursed for supported files), convert each to .fmff next to
    its source file, and show a live per-file + running total size
    comparison so the win (or, honestly, occasional loss -- see
    _fmt_saving) is visible at a glance rather than something you have to
    go check file-by-file after the fact."""

    def __init__(self, parent_root, paths=()):
        self.root = parent_root
        self.top = tk.Toplevel(parent_root)
        self.top.title("FMFF Batch Convert")
        self.top.configure(bg=DARK_BG)
        self.top.geometry("760x440")

        toolbar = tk.Frame(self.top, bg=DARK_PANEL)
        toolbar.pack(fill="x")
        for text, cmd in (("Add Files...", self.add_files), ("Add Folder...", self.add_folder),
                          ("Clear", self.clear)):
            tk.Button(toolbar, text=text, command=cmd, bg=DARK_BTN, fg=DARK_FG,
                      activebackground="#454545", activeforeground=DARK_FG,
                      relief="flat", padx=10, pady=4).pack(side="left", padx=4, pady=4)
        self.start_btn = tk.Button(toolbar, text="Start", command=self._on_start_stop, bg="#2d5f3d", fg=DARK_FG,
                                    activebackground="#3a7a4f", activeforeground=DARK_FG,
                                    relief="flat", padx=14, pady=4)
        self.start_btn.pack(side="right", padx=4, pady=4)

        style = ttk.Style(self.top)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("FMFF.Treeview", background=DARK_PANEL, fieldbackground=DARK_PANEL,
                        foreground=DARK_FG, rowheight=22, borderwidth=0)
        style.configure("FMFF.Treeview.Heading", background=DARK_BTN, foreground=DARK_FG, relief="flat")
        style.map("FMFF.Treeview", background=[("selected", "#3a6ea5")])

        columns = ("file", "original", "fmff", "saved", "status")
        self.tree = ttk.Treeview(self.top, columns=columns, show="headings",
                                  style="FMFF.Treeview", height=15)
        for col, label, width in (("file", "File", 300), ("original", "Original", 90),
                                   ("fmff", "FMFF", 90), ("saved", "Saved", 100),
                                   ("status", "Status", 150)):
            self.tree.heading(col, text=label)
            self.tree.column(col, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self.summary = tk.Label(self.top, text="0 files queued", anchor="w", bg=DARK_PANEL, fg=DARK_FG)
        self.summary.pack(fill="x")

        self.rows = {}     # path -> tree item id
        self.sizes = {}    # path -> (orig_bytes, new_bytes), only once done
        self.q = queue.Queue()
        self._running = False
        self._closed = False
        self.stop_event = threading.Event()

        self.top.protocol("WM_DELETE_WINDOW", self.on_close)

        for p in paths:
            self._add_path(p)
        self._poll()

    # -- queueing -----------------------------------------------------------

    def _add_path(self, path):
        if os.path.isdir(path):
            for root_dir, _dirs, files in os.walk(path):
                for name in files:
                    self._add_file(os.path.join(root_dir, name))
        else:
            self._add_file(path)

    def _add_file(self, path):
        ext = Path(path).suffix.lower()
        if ((ext not in IMAGE_EXTS and ext not in VIDEO_EXTS and ext not in AUDIO_EXTS
                and ext not in DOCUMENT_EXTS)
                or path in self.rows):
            return
        try:
            size = os.path.getsize(path)
        except OSError:
            return
        item = self.tree.insert("", "end", values=(Path(path).name, _fmt_size(size), "", "", "queued"))
        self.rows[path] = item
        self._update_summary()

    def add_files(self):
        paths = filedialog.askopenfilenames(filetypes=[
            ("Images, video, audio, and documents",
                "*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tiff *.tif *.ico "
                "*.mp4 *.avi *.mov *.mkv *.webm *.m4v "
                "*.mp3 *.wav *.flac *.opus *.ogg *.m4a *.aac *.wma "
                "*.pdf *.txt"),
            ("All files", "*.*"),
        ])
        for p in paths:
            self._add_file(p)

    def add_folder(self):
        d = filedialog.askdirectory()
        if d:
            self._add_path(d)

    def clear(self):
        if self._running:
            return
        self.tree.delete(*self.tree.get_children())
        self.rows.clear()
        self.sizes.clear()
        self._update_summary()

    def _update_summary(self):
        n = len(self.rows)
        if self.sizes:
            total_orig = sum(o for o, _ in self.sizes.values())
            total_new = sum(v for _, v in self.sizes.values())
            self.summary.config(
                text=f"{n} file(s) queued, {len(self.sizes)} done -- "
                     f"{_fmt_size(total_orig)} -> {_fmt_size(total_new)} "
                     f"({_fmt_saving(total_orig, total_new)})")
        else:
            self.summary.config(text=f"{n} file(s) queued")

    # -- conversion -----------------------------------------------------------

    def _on_start_stop(self):
        if self._running:
            self.stop()
        else:
            self.start()

    def start(self):
        if self._running or not self.rows:
            return
        self._running = True
        self.stop_event.clear()
        self.start_btn.config(text="Stop", bg="#7a2d2d", activebackground="#a53a3a")
        threading.Thread(target=self._worker, daemon=True).start()

    def stop(self):
        """Signal the worker thread to stop as soon as possible: it will
        finish checking (not restart) the file it's on, terminate an
        in-progress ffmpeg encode via cancel_event, and mark every
        not-yet-started file "cancelled" instead of leaving them stuck at
        "queued" forever."""
        self.stop_event.set()
        self.start_btn.config(state="disabled")

    def on_close(self):
        """Closing the window mid-conversion must actually stop the
        background thread, not just hide the UI while it keeps running
        unattended in the background."""
        self._closed = True
        self.stop_event.set()
        self.top.destroy()

    def _worker(self):
        for path in list(self.rows.keys()):
            if self.stop_event.is_set():
                self.q.put(("status", (path, "cancelled")))
                continue
            self.q.put(("status", (path, "converting...")))
            try:
                out_path = str(Path(path).with_suffix(".fmff"))
                ext = Path(path).suffix.lower()
                enc = FMFFEncoder()
                if ext in VIDEO_EXTS:
                    stats = enc.encode_video(path, out_path, cancel_event=self.stop_event)
                elif ext in AUDIO_EXTS:
                    stats = enc.encode_audio(path, out_path, cancel_event=self.stop_event)
                else:
                    enc.quality = _default_quality_for(path, None)
                    stats = enc.encode(path, out_path)
                self.q.put(("done", (path, os.path.getsize(path), stats["size"])))
            except EncodingCancelled:
                self.q.put(("status", (path, "cancelled")))
            except Exception as exc:
                self.q.put(("error", (path, str(exc))))
        self.q.put(("all_done", None))

    def _poll(self):
        if self._closed:
            return
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "status":
                    path, text = payload
                    self.tree.set(self.rows[path], "status", text)
                elif kind == "done":
                    path, orig, new = payload
                    self.sizes[path] = (orig, new)
                    self.tree.set(self.rows[path], "original", _fmt_size(orig))
                    self.tree.set(self.rows[path], "fmff", _fmt_size(new))
                    self.tree.set(self.rows[path], "saved", _fmt_saving(orig, new))
                    self.tree.set(self.rows[path], "status", "done")
                    self._update_summary()
                elif kind == "error":
                    path, msg = payload
                    self.tree.set(self.rows[path], "status", f"error: {msg}"[:80])
                elif kind == "all_done":
                    self._running = False
                    self.start_btn.config(state="normal", text="Start",
                                           bg="#2d5f3d", activebackground="#3a7a4f")
        except queue.Empty:
            pass
        self.top.after(80, self._poll)


class MediaViewer:
    """General media viewer window: .fmff (progressive), images (Pillow,
    including animated GIF/WEBP) and video (OpenCV), plus a screen-region
    screenshot tool. Everything is letterboxed onto a fixed dark canvas."""

    CANVAS_W, CANVAS_H = 960, 640
    BG_RGB = (26, 26, 26)

    def __init__(self, initial_path=None, simulate_slow=0.0):
        if tk is None:
            raise RuntimeError("tkinter is not available in this Python install")
        self.initial_path = initial_path
        self.simulate_slow = simulate_slow
        self.q = queue.Queue(maxsize=8)
        self.stop_event = None
        self.anim_job = None
        self.current_pil = None
        self.current_exif = None
        self.current_icc = None
        self.current_source_path = None
        self.current_is_video = False
        self.current_is_document = False
        self.current_is_fmff = False
        self.playback_fps = 60
        self._fmff_buffer = None
        self._anim_frames = None

    # -- window / widgets ------------------------------------------------

    def run(self):
        self.root = root = TkinterDnD.Tk() if TkinterDnD is not None else tk.Tk()
        root.title("FMFF Media Viewer")
        root.configure(bg=DARK_BG)
        # default=True also covers every Toplevel spawned off this root
        # (BatchConvertWindow, the screenshot overlay, ...) that doesn't
        # set its own icon -- see _app_icon_photo. Kept on self, not just
        # a local var, so it isn't garbage-collected out from under the
        # window while still in use.
        self._icon_img = _app_icon_photo()
        root.iconphoto(True, self._icon_img)

        toolbar = tk.Frame(root, bg=DARK_PANEL)
        toolbar.pack(fill="x")
        for text, cmd in (("Open", self.on_open), ("Screenshot", self.on_screenshot),
                          ("Save as...", self.on_save), ("Batch...", self.on_batch)):
            tk.Button(toolbar, text=text, command=cmd, bg=DARK_BTN, fg=DARK_FG,
                      activebackground="#454545", activeforeground=DARK_FG,
                      relief="flat", padx=10, pady=4).pack(side="left", padx=4, pady=4)

        self.fps_var = tk.StringVar(value=f"{self.playback_fps} fps")
        tk.Button(toolbar, textvariable=self.fps_var, command=self.on_fps_click, bg=DARK_BTN, fg=DARK_FG,
                  activebackground="#454545", activeforeground=DARK_FG,
                  relief="flat", padx=10, pady=4).pack(side="left", padx=4, pady=4)

        self.canvas = tk.Canvas(root, width=self.CANVAS_W, height=self.CANVAS_H,
                                bg=DARK_BG, highlightthickness=0)
        self.canvas.pack()
        placeholder_text = ("Open a file, take a screenshot, or drag & drop files/folders here"
                             if TkinterDnD is not None else
                             "Open a file or take a screenshot")
        self.status = tk.Label(root, text="ready", anchor="w", bg=DARK_PANEL, fg=DARK_FG)
        self.status.pack(fill="x")

        self.photo = None
        self.image_id = self.canvas.create_image(0, 0, anchor="nw")
        self._blit_pil(self._placeholder_image(placeholder_text))

        if TkinterDnD is not None:
            root.drop_target_register(DND_FILES)
            root.dnd_bind("<<Drop>>", self.on_drop)

        root.after(40, self._poll)
        if self.initial_path:
            self.open_path(self.initial_path)
        root.mainloop()

    def set_status(self, text):
        self.status.config(text=text)

    def _placeholder_image(self, text):
        img = Image.new("RGB", (self.CANVAS_W, self.CANVAS_H), self.BG_RGB)
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((self.CANVAS_W - tw) // 2, (self.CANVAS_H - th) // 2), text, fill=(120, 120, 120))
        return img

    # -- letterboxed rendering -------------------------------------------

    def _letterbox(self, img):
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA")
        iw, ih = img.size
        scale = min(self.CANVAS_W / iw, self.CANVAS_H / ih) if iw and ih else 1.0
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = img.resize((nw, nh), Image.BILINEAR)

        frame = Image.new("RGB", (self.CANVAS_W, self.CANVAS_H), self.BG_RGB)
        pos = ((self.CANVAS_W - nw) // 2, (self.CANVAS_H - nh) // 2)
        if resized.mode == "RGBA":
            frame.paste(resized, pos, resized)
        else:
            frame.paste(resized, pos)
        return frame

    def _show_frame(self, frame):
        self.photo = ImageTk.PhotoImage(frame)
        self.canvas.itemconfig(self.image_id, image=self.photo)

    def _blit_pil(self, img, remember=True):
        if remember:
            self.current_pil = img
        self._show_frame(self._letterbox(img))

    def _set_busy(self, percent, label="Working"):
        """Frost the current frame with a semi-transparent white veil and a
        centered percentage, so a slow encode/convert doesn't look frozen."""
        base = self._letterbox(self.current_pil) if self.current_pil is not None \
            else Image.new("RGB", (self.CANVAS_W, self.CANVAS_H), self.BG_RGB)
        veil = Image.blend(base, Image.new("RGB", base.size, (255, 255, 255)), 0.55)
        draw = ImageDraw.Draw(veil)
        text = f"{label}... {percent}%" if percent is not None else f"{label}..."
        bbox = draw.textbbox((0, 0), text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((self.CANVAS_W - tw) // 2, (self.CANVAS_H - th) // 2), text, fill=(20, 20, 20))
        self._show_frame(veil)

    # -- background-task lifecycle ---------------------------------------

    def _cancel_background(self):
        if self.stop_event is not None:
            self.stop_event.set()
            self.stop_event = None
        if self.anim_job is not None:
            self.root.after_cancel(self.anim_job)
            self.anim_job = None
        self._anim_frames = None

    def _poll(self):
        latest_frame = None
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "fmff_tile":
                    entry, y0, x0, decoded = payload
                    if entry.plane == PLANE_ALPHA:
                        self._fmff_buffer[y0:y0 + entry.h, x0:x0 + entry.w, 3] = decoded
                    else:
                        self._fmff_buffer[y0:y0 + entry.h, x0:x0 + entry.w, :3] = decoded
                    self._fmff_tiles_done += 1
                elif kind == "fmff_done":
                    # First (and only) time the image is shown -- see
                    # _load_fmff_image for why there's no tile-by-tile blit.
                    # payload is full()'s actual return value, not the
                    # tile-accumulated buffer: modes with no tiles at all
                    # (JPEG passthrough) never touch that buffer.
                    self._blit_pil(payload)
                    self.set_status(f"loaded ({self._fmff_w}x{self._fmff_h})")
                elif kind == "fmff_anim_done":
                    # All frames decoded at once (see _load_fmff_animated) --
                    # then it's the exact same _anim_frames/_animate loop
                    # that plays an animated GIF/WebP already, so playback
                    # behaves identically no matter which format it came
                    # from.
                    frames, durations, n_frames = payload
                    self._anim_frames, self._anim_durations, self._anim_idx = frames, durations, 0
                    self._animate()
                    self.set_status(f"loaded animated ({self._fmff_w}x{self._fmff_h}, {n_frames} frames)")
                elif kind == "video_frame":
                    # Only the newest frame matters for display -- if the GUI
                    # thread fell behind, silently drop the stale backlog
                    # instead of blitting every queued frame one by one
                    # (that per-frame catch-up work is what froze the window).
                    latest_frame = payload
                elif kind == "busy":
                    percent, label = payload
                    self._set_busy(percent, label)
                elif kind == "busy_done":
                    if self.current_pil is not None:
                        self._blit_pil(self.current_pil, remember=False)
                elif kind == "doc_pages":
                    # A rendered PDF/.txt preview (see _load_document_file) --
                    # same _anim_frames/_animate loop an animated GIF
                    # already uses, so paging through it looks identical.
                    frames, durations = payload
                    self._anim_frames, self._anim_durations, self._anim_idx = frames, durations, 0
                    self._animate()
                    self.set_status(f"rendered {len(frames)} page(s)")
                elif kind in ("error", "status"):
                    self.set_status(payload)
        except queue.Empty:
            pass
        if latest_frame is not None:
            mode = "RGBA" if latest_frame.shape[2] == 4 else "RGB"
            self._blit_pil(Image.fromarray(latest_frame, mode), remember=True)
        self.root.after(40, self._poll)

    # -- opening media -----------------------------------------------------

    def on_open(self):
        path = filedialog.askopenfilename(filetypes=[
            ("All supported", "*.fmff *.png *.jpg *.jpeg *.bmp *.gif *.webp *.tiff *.tif *.ico "
                               "*.mp4 *.avi *.mov *.mkv *.webm *.m4v "
                               "*.mp3 *.wav *.flac *.opus *.ogg *.m4a *.aac *.wma "
                               "*.pdf *.txt"),
            ("FMFF", "*.fmff"),
            ("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tiff *.tif *.ico"),
            ("Videos", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
            ("Audio", "*.mp3 *.wav *.flac *.opus *.ogg *.m4a *.aac *.wma"),
            ("Documents", "*.pdf *.txt"),
            ("All files", "*.*"),
        ])
        if path:
            self.open_path(path)

    def on_batch(self):
        BatchConvertWindow(self.root)

    def on_drop(self, event):
        paths = [p for p in self.root.tk.splitlist(event.data) if p]
        if not paths:
            return
        # a single dropped file previews like Open does; a folder, or more
        # than one item, queues them for batch conversion instead -- that
        # matches what dropping each of those would intuitively mean.
        if len(paths) == 1 and os.path.isfile(paths[0]):
            self.open_path(paths[0])
        else:
            BatchConvertWindow(self.root, paths)

    def on_fps_click(self):
        val = simpledialog.askinteger("Playback FPS", "Frames per second:",
                                       initialvalue=self.playback_fps, minvalue=1, maxvalue=240,
                                       parent=self.root)
        if val:
            self.playback_fps = val
            self.fps_var.set(f"{val} fps")

    def open_path(self, path):
        self._cancel_background()
        self._fmff_buffer = None
        self.playback_fps = 60
        self.fps_var.set(f"{self.playback_fps} fps")
        ext = Path(path).suffix.lower()
        self.current_source_path = str(path)
        self.current_is_fmff = (ext == ".fmff")
        self.current_is_video = False
        self.current_is_audio = False
        self.current_is_document = False
        self.current_exif = None
        self.current_icc = None
        self.root.title(f"FMFF Media Viewer - {Path(path).name}")
        try:
            if ext == ".fmff":
                self._load_fmff(path)
            elif ext in VIDEO_EXTS:
                self.current_is_video = True
                self._load_video_file(path)
            elif ext in AUDIO_EXTS:
                self.current_is_audio = True
                self._load_audio_file(path)
            elif ext in DOCUMENT_EXTS:
                self.current_is_document = True
                self._load_document_file(path)
            else:
                self._load_image(path)
        except Exception as exc:
            self.set_status(f"failed to open: {exc}")

    def _load_fmff(self, path):
        d = FMFFDecoder(path)
        self.current_is_video = d.is_video
        self.current_is_audio = d.is_audio
        self.current_is_document = d.is_document
        if d.is_video:
            self._load_fmff_video(d)
        elif d.is_audio:
            self._load_fmff_audio(d)
        elif d.is_document:
            self._load_fmff_document(d)
        elif d.is_animated:
            self._load_fmff_animated(d)
        else:
            self._load_fmff_image(d)

    def _load_fmff_document(self, d):
        """An .fmff document (see FMFFEncoder.encode_document): the
        original PDF/.txt bytes are recovered byte-for-byte to a temp
        file and handed straight to _load_document_file -- the exact
        same page-rendering path a plain, not-yet-converted PDF/.txt
        already uses, so browsing looks identical either way and
        nothing about rendering pages is duplicated for the .fmff case.
        Save as... reads straight from the original .fmff file itself
        (see _save_document), not from this temp copy, which is thrown
        away with the rest of tmp_dir once the OS gets around to it."""
        ext = d.tags.get("doc_ext", ".pdf")
        tmp_dir = tempfile.mkdtemp(prefix="fmff_view_")
        doc_path = os.path.join(tmp_dir, "document" + ext)
        d.extract_document(doc_path)
        self._load_document_file(doc_path)

    def _load_fmff_animated(self, d):
        """An .fmff with more than one frame (see encode_image_sequence)
        -- decode every frame up front in the background (same reasoning
        as _load_fmff_image: no visible tile-by-tile fill-in), then hand
        the frame list to the exact same _anim_frames/_animate loop an
        animated GIF/WebP already uses, so playback looks identical
        regardless of source format."""
        self._fmff_w, self._fmff_h = d.width, d.height
        base = np.array(d.thumbnail().convert("RGBA").resize((d.width, d.height), Image.BILINEAR))
        if not d.has_alpha:
            base[:, :, 3] = 255
        self._blit_pil(Image.fromarray(base, "RGBA"))
        self.set_status(f"decoding {d.frame_count} frames...")

        stop_event = threading.Event()
        self.stop_event = stop_event

        def worker():
            durations = d.frame_durations_ms()
            # full_sequence() decodes every frame in one linear pass instead
            # of calling full(frame=i) per frame (which re-scans the whole
            # entry list each time -- quadratic in frame count, see its
            # docstring), so this can't check stop_event between frames the
            # way the old loop did; cancelling mid-decode just discards the
            # result below instead.
            frames = d.full_sequence()
            if not stop_event.is_set():
                self.q.put(("fmff_anim_done", (frames, durations, d.frame_count)))

        threading.Thread(target=worker, daemon=True).start()

    def _load_fmff_image(self, d):
        # Decode fully in the background, but only show the finished image --
        # no visible tile-by-tile fill-in. The thumbnail is still decoded as
        # the compositing base (so partial tiles land on something sane) but
        # is never itself blitted to the canvas, so nothing looks like it's
        # struggling; the (usually well under a second) decode just happens
        # behind a plain "decoding..." status instead.
        # Carried separately from the decoded pixels (see _load_image's
        # identical comment for why) so Save as... can still attach it --
        # this .fmff's own header is the only place it lives, decode()
        # never puts it back into the returned PIL Image's .info.
        self.current_exif = d.exif_bytes
        self.current_icc = d.icc_bytes
        self._fmff_w, self._fmff_h = d.width, d.height
        self._fmff_tiles_total = len(d.tile_byte_ranges())
        self._fmff_tiles_done = 0

        base = np.array(d.thumbnail().convert("RGBA").resize((d.width, d.height), Image.BILINEAR))
        if not d.has_alpha:
            base[:, :, 3] = 255
        self._fmff_buffer = base
        self.set_status("decoding...")

        stop_event = threading.Event()
        self.stop_event = stop_event

        def worker():
            def on_tile(entry, y0, x0, decoded):
                if stop_event.is_set():
                    return
                self.q.put(("fmff_tile", (entry, y0, x0, decoded)))
                if self.simulate_slow:
                    time.sleep(self.simulate_slow)
            result_img = d.full(progress_cb=on_tile)
            if not stop_event.is_set():
                self.q.put(("fmff_done", result_img))

        threading.Thread(target=worker, daemon=True).start()

    def _load_fmff_video(self, d):
        poster_mode = "RGBA" if d.has_video_alpha else "RGB"
        poster = np.array(d.thumbnail().convert(poster_mode).resize((d.width, d.height), Image.BILINEAR))
        self._blit_pil(Image.fromarray(poster, poster_mode))

        tmp_dir = tempfile.mkdtemp(prefix="fmff_view_")
        blob_path = os.path.join(tmp_dir, "video.mp4")
        try:
            d.extract_media(blob_path)
        except ValueError as exc:
            self.set_status(str(exc))
            return
        corrupt_note = (f", {d.last_corrupt_segments} of {d.segment_count} segment(s) "
                         f"corrupt/missing -- skipped" if d.last_corrupt_segments else "")

        if d.has_video_alpha:
            # No mainstream player (ffplay included) can composite two
            # separate video tracks into transparency on the fly -- so an
            # alpha video always plays in-window, decoded and blended
            # against the dark canvas frame-by-frame (see
            # _load_video_with_alpha), the same way an RGBA *image*
            # already does. That means no live audio for these even when
            # the file has an audio track (audio is still preserved in
            # the file itself -- Save as... / decode still gets it).
            alpha_path = os.path.join(tmp_dir, "alpha.mp4")
            try:
                d.extract_alpha_media(alpha_path)
            except ValueError as exc:
                self.set_status(str(exc))
                return
            alpha_note = (f", {d.last_corrupt_alpha_segments} of {d.alpha_segment_count} "
                           f"alpha segment(s) corrupt/missing" if d.last_corrupt_alpha_segments else "")
            self.set_status(f"playing in-window with alpha ({d.width}x{d.height}, "
                             f"silent -- alpha video can't use the external player"
                             f"{corrupt_note}{alpha_note})")
            self._load_video_with_alpha(blob_path, alpha_path)
            return

        # blob_path's audio track is Opus, straight out of extract_media --
        # exactly the bytes stored on disk (see encode_video). That's the
        # same situation _load_fmff_audio already documents and works
        # around: Windows' own Media Foundation pipeline (what the
        # built-in Movies & TV / Media Player app, and anything else
        # using the system's registered decoders, decodes through)
        # doesn't reliably handle an Opus track at all unless it's inside
        # Opus's own native Ogg container -- confirmed there by direct
        # testing (silent failure or an outright "not supported" error).
        # _load_fmff_audio's fix (remux to Ogg) can't apply here, since
        # dropping the MP4 container would drop the video track with
        # it -- so instead just the audio is transcoded to AAC (decoded
        # by literally every player, this one included) while the AV1
        # video stream is copied through untouched (`-c:v copy`, so this
        # is fast regardless of the video's length -- only the audio is
        # actually re-encoded). This is a disposable playback-only copy;
        # the .fmff file itself keeps its original Opus audio exactly as
        # encoded (Save as... / decode still gets that, not this AAC copy).
        play_path = blob_path
        ffmpeg = _find_ffmpeg()
        if ffmpeg is not None:
            aac_path = os.path.join(tmp_dir, "video_playback.mp4")
            result = subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                                      "-i", blob_path, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                                      aac_path], check=False, stderr=subprocess.DEVNULL)
            if result.returncode == 0 and os.path.exists(aac_path):
                play_path = aac_path

        # Handed off to the system's default player: a real separate
        # window with real hardware decode, real audio, and a real seek
        # bar -- FMFF's own in-window loop (_load_video) is silent and
        # renders into this same window's canvas, which is worse on both
        # counts, not just a stylistic difference. Falls back to that
        # in-window loop only if launching the default player fails
        # outright (e.g. nothing associated with .mp4 at all).
        self.set_status(f"playing in the system's default player{corrupt_note}")
        try:
            _open_with_default_player(play_path)
            return
        except OSError as exc:
            self.set_status(f"couldn't launch default player ({exc}) -- "
                             f"falling back to silent in-window preview{corrupt_note}")
        self._load_video(blob_path, status_suffix=corrupt_note)

    def _load_fmff_audio(self, d):
        """An .fmff audio file (see FMFFEncoder.encode_audio): show its
        cover-art thumbnail (the plain placeholder encode_audio falls
        back to when the source has none) as a static image, extract the
        embedded Opus track, and hand it off to the system's default
        player. Unlike _load_fmff_video, this doesn't try to keep
        playback inside FMFF's own window -- there's no picture to
        render in-window for audio, so the reason video stays in-window
        (this is FMFF's own viewer, playback shouldn't hand off
        elsewhere) doesn't apply; the system's player is strictly better
        here (real seek bar, real volume, etc.) with nothing to trade off.

        extract_media() alone produces a *fragmented* MP4 with an
        Opus-only audio track -- exactly the bytes stored on disk, valid
        and exactly what FFmpeg-based tools expect, but confirmed (direct
        testing on a real Windows install) to fail in the built-in
        Movies & TV / Media Player app with "this item was encoded in a
        format that's not supported" (0xc00d5212): Windows' own Media
        Foundation pipeline doesn't reliably handle Opus-in-fragmented-
        MP4, unlike FFmpeg's own demuxer. Ogg is Opus's own native,
        widely-supported container (the one Windows' "Web Media
        Extensions" / built-in Opus support actually targets), so the
        extracted track is remuxed there (-c:a copy -- no re-encode, just
        a container swap) before being handed off, which is the file
        that actually gets played."""
        self._blit_pil(d.thumbnail())

        tmp_dir = tempfile.mkdtemp(prefix="fmff_view_")
        blob_path = os.path.join(tmp_dir, "audio.mp4")
        try:
            d.extract_media(blob_path)
        except ValueError as exc:
            self.set_status(str(exc))
            return
        corrupt_note = (f", {d.last_corrupt_segments} of {d.segment_count} segment(s) "
                         f"corrupt/missing -- skipped" if d.last_corrupt_segments else "")

        play_path = blob_path
        ffmpeg = _find_ffmpeg()
        if ffmpeg is not None:
            opus_path = os.path.join(tmp_dir, "audio.opus")
            result = subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                                      "-i", blob_path, "-c:a", "copy", opus_path],
                                     check=False, stderr=subprocess.DEVNULL)
            if result.returncode == 0 and os.path.exists(opus_path):
                play_path = opus_path

        self.set_status(f"playing in the system's default player "
                         f"({d.duration_ms / 1000:.1f}s, {d.sample_rate} Hz, "
                         f"{d.channels}ch{corrupt_note})")
        try:
            _open_with_default_player(play_path)
        except OSError as exc:
            self.set_status(f"couldn't launch default player: {exc}")

    def _load_audio_file(self, path):
        """Open a plain (non-.fmff) audio file the same way _load_video_file
        opens a plain video: hand it straight to the system's default
        player. There's no in-window fallback to offer the way video has
        -- there's nothing for this viewer to draw for audio in the first
        place, so a failed launch is just reported rather than degrading
        to some silent preview of nothing."""
        self._blit_pil(self._placeholder_image(f"playing in default player\n{Path(path).name}"))
        self.set_status(f"playing in the system's default player: {Path(path).name}")
        try:
            _open_with_default_player(str(path))
        except OSError as exc:
            self.set_status(f"couldn't launch default player: {exc}")

    def _load_document_file(self, path):
        """Preview a plain (non-.fmff) PDF/.txt by rendering its pages the
        same way encode_document would (see _render_document_pages),
        without writing anything -- then handing them to the exact same
        _anim_frames/_animate loop an animated GIF already uses, so
        paging through a rendered document looks identical to any other
        multi-frame preview. Runs off the GUI thread: rendering a many-
        page PDF is slow enough to visibly freeze the window otherwise,
        the same reasoning every other loader here backs its decode with
        a background thread for."""
        self._blit_pil(self._placeholder_image(f"rendering pages...\n{Path(path).name}"))
        self.set_status("rendering pages...")

        stop_event = threading.Event()
        self.stop_event = stop_event

        def worker():
            try:
                pages = _render_document_pages(path)
            except Exception as exc:
                if not stop_event.is_set():
                    self.q.put(("status", f"failed to render {Path(path).name}: {exc}"))
                return
            if stop_event.is_set():
                return
            frames = [p.convert("RGBA") for p in pages]
            durations = [1500] * len(frames)
            self.q.put(("doc_pages", (frames, durations)))

        threading.Thread(target=worker, daemon=True).start()

    def _load_image(self, path):
        img = Image.open(path)
        # Captured before convert() below drops it -- see on_save's use of
        # this for why: Save as... -> .fmff for a still image goes through
        # encode_image(still, path) using this already-converted PIL
        # object when the source isn't a JPEG (which instead re-reads the
        # original file fresh via encode(), recapturing EXIF on its own),
        # and convert() doesn't carry .info over, so EXIF would otherwise
        # quietly be lost between opening and saving. Same deal for the
        # ICC color profile.
        self.current_exif = img.info.get("exif")
        self.current_icc = img.info.get("icc_profile")
        n_frames = getattr(img, "n_frames", 1)
        if n_frames > 1:
            frames, durations = [], []
            for i in range(n_frames):
                img.seek(i)
                frames.append(img.convert("RGBA"))
                durations.append(max(20, img.info.get("duration", 100)))
            self._anim_frames, self._anim_durations, self._anim_idx = frames, durations, 0
            self._animate()
            self.set_status(f"animated, {n_frames} frames")
        else:
            self._blit_pil(img.convert("RGBA"))
            self.set_status(f"loaded {img.size[0]}x{img.size[1]}")

    def _animate(self):
        frames = self._anim_frames
        if not frames:
            return
        self._blit_pil(frames[self._anim_idx])
        delay = self._anim_durations[self._anim_idx]
        self._anim_idx = (self._anim_idx + 1) % len(frames)
        self.anim_job = self.root.after(delay, self._animate)

    def _load_video_file(self, path):
        """Open a plain (non-.fmff) video file the same way _load_fmff_video
        opens one: prefer the system's default player, for real audio and
        actual seek/pause controls, falling back to the silent in-window
        preview only if that fails. ffplay was tried first here before --
        see _open_with_default_player for why that's gone."""
        self._blit_pil(self._placeholder_image("playing in default player"))
        self.set_status(f"playing in the system's default player: {Path(path).name}")
        try:
            _open_with_default_player(str(path))
            return
        except OSError as exc:
            self.set_status(f"couldn't launch default player ({exc}) -- "
                             "falling back to silent in-window preview")
        self._load_video(path)

    def _load_video(self, path, status_suffix=""):
        if cv2 is None:
            self.set_status("video support needs OpenCV: pip install opencv-python")
            return

        stop_event = threading.Event()
        self.stop_event = stop_event
        self.set_status(f"opening {Path(path).name}...")

        def put_frame(rgb):
            try:
                self.q.put(("video_frame", rgb), timeout=1.0)
            except queue.Full:
                pass  # viewer fell behind or is shutting down -- drop this frame

        def worker():
            # Opening a container (probing codec/streams) can itself take a
            # while for some files -- do it here, off the GUI thread, so the
            # window never blocks/"stops responding" just from loading.
            cap = _open_video_capture(path)
            if not cap.isOpened():
                self.q.put(("status", f"could not open video: {path}"))
                return
            hw = _HW_ACCEL_NAMES.get(cap.get(cv2.CAP_PROP_HW_ACCELERATION), "unknown")
            self.q.put(("status", f"playing video (playback {self.playback_fps} fps, looping, "
                                   f"decode: {hw}) -- click the fps button to change playback speed"
                                   f"{status_suffix}"))
            try:
                while not stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    put_frame(rgb)
                    time.sleep(1.0 / max(1, self.playback_fps))
            finally:
                cap.release()

        threading.Thread(target=worker, daemon=True).start()

    def _load_video_with_alpha(self, color_path, alpha_path):
        """Like _load_video, but for a video with a real alpha channel
        (see encode_video's alpha handling): reads the color and alpha
        AV1 tracks as two separate captures, composites them into one
        RGBA frame per step, and pushes that -- _poll blits it through
        the same alpha-aware path an RGBA *image* already uses (see
        _letterbox), so it renders against the dark canvas with real
        transparency instead of a solid rectangle. No external player
        (the system default included) can do this compositing, so this
        path is always the silent in-window one; see _load_fmff_video."""
        if cv2 is None:
            self.set_status("video support needs OpenCV: pip install opencv-python")
            return

        stop_event = threading.Event()
        self.stop_event = stop_event

        def put_frame(rgba):
            try:
                self.q.put(("video_frame", rgba), timeout=1.0)
            except queue.Full:
                pass

        def worker():
            cap = _open_video_capture(color_path)
            acap = _open_video_capture(alpha_path)
            if not cap.isOpened() or not acap.isOpened():
                self.q.put(("status", "could not open video/alpha streams"))
                cap.release()
                acap.release()
                return
            try:
                while not stop_event.is_set():
                    ok, frame = cap.read()
                    aok, aframe = acap.read()
                    if not ok or not aok:
                        # Independent encodes of the same source can end up
                        # a frame or two apart in length -- loop both back
                        # together rather than letting them drift out of
                        # sync a frame at a time.
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        acap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    alpha = cv2.cvtColor(aframe, cv2.COLOR_BGR2GRAY)
                    if alpha.shape[:2] != rgb.shape[:2]:
                        alpha = cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]))
                    put_frame(np.dstack([rgb, alpha]))
                    time.sleep(1.0 / max(1, self.playback_fps))
            finally:
                cap.release()
                acap.release()

        threading.Thread(target=worker, daemon=True).start()

    # -- screenshot --------------------------------------------------------

    def on_screenshot(self):
        self._cancel_background()
        self.root.withdraw()
        self.root.after(250, self._start_snip)

    def _start_snip(self):
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        overlay = tk.Toplevel(self.root)
        overlay.overrideredirect(True)
        overlay.geometry(f"{sw}x{sh}+0+0")
        overlay.attributes("-topmost", True)
        overlay.attributes("-alpha", 0.35)
        overlay.configure(bg="black")

        canvas = tk.Canvas(overlay, cursor="cross", bg="gray10", highlightthickness=0,
                          width=sw, height=sh)
        canvas.pack(fill="both", expand=True)

        state = {"x0": 0, "y0": 0, "rect": None}

        def on_press(evt):
            state["x0"], state["y0"] = evt.x, evt.y
            state["rect"] = canvas.create_rectangle(evt.x, evt.y, evt.x, evt.y,
                                                     outline="#4da6ff", width=2)

        def on_drag(evt):
            if state["rect"] is not None:
                canvas.coords(state["rect"], state["x0"], state["y0"], evt.x, evt.y)

        def on_release(evt):
            x0, y0, x1, y1 = state["x0"], state["y0"], evt.x, evt.y
            box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            overlay.destroy()
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                self.root.deiconify()
                self.set_status("screenshot cancelled")
                return
            self.root.after(150, lambda: self._finish_snip(box))

        def on_escape(_evt):
            overlay.destroy()
            self.root.deiconify()
            self.set_status("screenshot cancelled")

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        overlay.bind("<Escape>", on_escape)
        overlay.focus_force()

    def _finish_snip(self, box):
        img = ImageGrab.grab(bbox=box).convert("RGBA")
        self.root.deiconify()
        self._cancel_background()
        self.current_source_path = None
        self.current_is_video = False
        self.current_is_audio = False
        self.current_is_document = False
        self.current_exif = None
        self.current_icc = None
        self.current_is_fmff = False
        self._blit_pil(img)
        self.set_status(f"screenshot captured ({box[2]-box[0]}x{box[3]-box[1]}) - Save as... to keep it")

    # -- saving --------------------------------------------------------------

    def on_save(self):
        if self.current_is_video:
            path = filedialog.asksaveasfilename(
                defaultextension=".fmff",
                filetypes=[("FMFF", "*.fmff"), ("MP4", "*.mp4"), ("WebM", "*.webm")])
        elif self.current_is_audio:
            path = filedialog.asksaveasfilename(
                defaultextension=".fmff",
                filetypes=[("FMFF", "*.fmff"), ("Opus", "*.opus"), ("OGG", "*.ogg"),
                           ("M4A", "*.m4a"), ("MP3", "*.mp3"), ("WAV", "*.wav"),
                           ("FLAC", "*.flac")])
        elif self.current_is_document:
            path = filedialog.asksaveasfilename(
                defaultextension=".fmff",
                filetypes=[("FMFF", "*.fmff"), ("PDF", "*.pdf"), ("Text", "*.txt")])
        else:
            if self.current_pil is None:
                self.set_status("nothing to save")
                return
            path = filedialog.asksaveasfilename(
                defaultextension=".fmff",
                filetypes=[("FMFF", "*.fmff"), ("PNG", "*.png"), ("JPEG", "*.jpg"), ("GIF", "*.gif")])
        if not path:
            return

        dest_ext = Path(path).suffix.lower()
        is_video = self.current_is_video
        is_audio = self.current_is_audio
        is_document = self.current_is_document
        source_path, is_fmff, fps = self.current_source_path, self.current_is_fmff, self.playback_fps
        still = self.current_pil
        exif_bytes = self.current_exif
        icc_bytes = self.current_icc
        # Grab these before _cancel_background() below, which clears
        # _anim_frames -- otherwise an open animated GIF/WebP/.fmff would
        # always save as just whatever single frame happened to be on
        # screen at the moment Save as... was clicked, animation lost.
        anim_frames = self._anim_frames
        anim_durations = self._anim_durations if anim_frames else None

        # a live video/animation would keep overwriting the frame under the
        # busy veil (and race the encoder if it's an fmff video) -- freeze it
        self._cancel_background()
        is_animated = anim_frames is not None and len(anim_frames) > 1

        def progress_cb(percent):
            self.q.put(("busy", (percent, "Converting")))

        def work():
            try:
                if is_video:
                    self.q.put(("busy", (0, "Converting")))
                    _save_video(source_path, is_fmff, fps, path, dest_ext, progress_cb)
                elif is_audio:
                    self.q.put(("busy", (None, "Converting")))
                    _save_audio(source_path, is_fmff, path, dest_ext, progress_cb)
                elif is_document:
                    self.q.put(("busy", (None, "Converting")))
                    _save_document(source_path, is_fmff, path, dest_ext, progress_cb)
                elif dest_ext == ".fmff":
                    self.q.put(("busy", (None, "Encoding")))
                    quality = _default_quality_for(source_path or "", None)
                    enc = FMFFEncoder(quality=quality)
                    if is_animated:
                        enc.encode_image_sequence(anim_frames, anim_durations, path)
                    # If the open file is still an unedited JPEG on disk,
                    # go through encode() so it can try lossless coefficient
                    # passthrough first -- current_pil alone (just decoded
                    # pixels) can't do that, it has no access to the
                    # source's original DCT coefficients any more.
                    elif (source_path and Path(source_path).suffix.lower() in (".jpg", ".jpeg")
                            and Path(source_path).is_file()):
                        enc.encode(source_path, path)
                    else:
                        # encode_image reads img.info["exif"] itself -- see
                        # _load_image/_load_fmff_image's identical comments
                        # for why exif_bytes/icc_bytes are carried
                        # separately from `still` rather than already
                        # being on it.
                        if exif_bytes:
                            still.info["exif"] = exif_bytes
                        if icc_bytes:
                            still.info["icc_profile"] = icc_bytes
                        enc.encode_image(still, path)
                elif is_animated and dest_ext in (".gif", ".webp", ".png"):
                    # See cmd_decode's identical save_kwargs for why .webp
                    # specifically needs lossless=True here.
                    save_kwargs = {"lossless": True} if dest_ext == ".webp" else {}
                    if exif_bytes:
                        save_kwargs["exif"] = exif_bytes
                    if icc_bytes:
                        save_kwargs["icc_profile"] = icc_bytes
                    anim_frames[0].save(path, save_all=True, append_images=anim_frames[1:],
                                         duration=anim_durations, loop=0, **save_kwargs)
                else:
                    img = still.convert("RGB") if dest_ext in (".jpg", ".jpeg") else still
                    save_kwargs = {"lossless": True} if dest_ext == ".webp" else {}
                    if exif_bytes:
                        save_kwargs["exif"] = exif_bytes
                    if icc_bytes:
                        save_kwargs["icc_profile"] = icc_bytes
                    img.save(path, **save_kwargs)
                self.q.put(("busy_done", None))
                try:
                    new_size = os.path.getsize(path)
                    orig_size = (os.path.getsize(source_path)
                                 if source_path and os.path.exists(source_path) else None)
                    if orig_size and dest_ext == ".fmff":
                        self.q.put(("status", f"saved {path}  ({_fmt_size(orig_size)} -> "
                                               f"{_fmt_size(new_size)}, {_fmt_saving(orig_size, new_size)})"))
                    else:
                        self.q.put(("status", f"saved {path}  ({_fmt_size(new_size)})"))
                except OSError:
                    self.q.put(("status", f"saved {path}"))
            except Exception as exc:
                self.q.put(("busy_done", None))
                self.q.put(("status", f"save failed: {exc}"))

        threading.Thread(target=work, daemon=True).start()


def _save_video(source_path, is_fmff, fps, path, dest_ext, progress_cb):
    """Save a whole video (not just one displayed frame) either as .fmff or
    re-muxed into another video container. Runs off the GUI thread; reports
    0-100 through progress_cb so the viewer can show a busy overlay."""
    if dest_ext == ".fmff":
        if is_fmff:
            shutil.copyfile(source_path, path)
        else:
            def cb(i, total):
                if total:
                    progress_cb(min(100, int(i * 100 / total)))
            FMFFEncoder().encode_video(source_path, path, fps=fps, progress_cb=cb)
    elif dest_ext in VIDEO_EXTS:
        if is_fmff:
            dec = FMFFDecoder(source_path)
            # See cmd_decode's identical .mp4-with-tags handling: the raw
            # extracted stream never had the source's tags muxed in (they
            # live in FMFF's own metadata blob, not the AV1/Opus stream),
            # so restoring them needs an FFmpeg remux even for .mp4.
            if dest_ext == ".mp4" and not dec.tags:
                dec.extract_media(path)
            else:
                ffmpeg = _find_ffmpeg()
                if ffmpeg is None:
                    raise RuntimeError("saving to this format needs FFmpeg on PATH")
                with tempfile.TemporaryDirectory(prefix="fmff_save_") as tmp:
                    blob_path = os.path.join(tmp, "video.mp4")
                    dec.extract_media(blob_path)
                    # -map 0 (every stream in the input), not FFmpeg's
                    # default "best of each type" auto-selection -- a
                    # video .fmff can carry several audio/subtitle tracks
                    # (see encode_video), and -c copy alone would silently
                    # drop every one but the first of each type otherwise.
                    subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                                     "-i", blob_path, "-map", "0", "-c", "copy",
                                     *_ffmpeg_metadata_args(dec.tags), path], check=True)
            progress_cb(100)
        else:
            shutil.copyfile(source_path, path)
    else:
        raise ValueError(f"can't save a video as {dest_ext or '(no extension)'}")


def _save_audio(source_path, is_fmff, path, dest_ext, progress_cb):
    """Save the whole audio track (not whatever placeholder/cover-art
    image happens to be on screen -- see _load_audio_file/_load_fmff_audio,
    neither of which loads audio into self.current_pil the way an actual
    picture would) either as .fmff or remuxed/transcoded into another
    audio container -- the audio counterpart to _save_video. Runs off
    the GUI thread; reports through progress_cb so the viewer can show a
    busy overlay. Before this existed, Save as... on an audio file fell
    through to the plain still-image path and silently encoded whatever
    placeholder text or cover art was being displayed as a picture --
    the audio itself was never touched, and nothing errored to say so."""
    if dest_ext == ".fmff":
        if is_fmff:
            shutil.copyfile(source_path, path)
        else:
            FMFFEncoder().encode_audio(source_path, path)
        progress_cb(100)
    elif dest_ext in AUDIO_EXTS or dest_ext == ".mp4":
        if is_fmff:
            dec = FMFFDecoder(source_path)
            if dest_ext == ".mp4" and not dec.tags:
                dec.extract_media(path)
            else:
                ffmpeg = _find_ffmpeg()
                if ffmpeg is None:
                    raise RuntimeError("saving to this format needs FFmpeg on PATH")
                with tempfile.TemporaryDirectory(prefix="fmff_save_") as tmp:
                    blob_path = os.path.join(tmp, "audio.mp4")
                    dec.extract_media(blob_path)
                    # See cmd_decode's identical fallback for why: -c:a copy
                    # works only where the container can hold Opus verbatim
                    # (.opus/.ogg/.m4a); anything else (.mp3/.wav/.flac/...)
                    # needs an actual transcode. Either way the source's own
                    # tags (see encode_audio) are restored here too.
                    meta_args = _ffmpeg_metadata_args(dec.tags)
                    try:
                        subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                                         "-i", blob_path, "-c:a", "copy",
                                         *meta_args, path], check=True)
                    except subprocess.CalledProcessError:
                        subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                                         "-i", blob_path, *meta_args, path], check=True)
        else:
            shutil.copyfile(source_path, path)
        progress_cb(100)
    else:
        raise ValueError(f"can't save audio as {dest_ext or '(no extension)'}")


def _save_document(source_path, is_fmff, path, dest_ext, progress_cb):
    """Save the original document (not the rendered preview pages -- see
    _load_document_file/_load_fmff_document, neither of which loads a
    document into self.current_pil the way an actual picture would)
    either as .fmff or back out to a real PDF/.txt file -- the document
    counterpart to _save_audio/_save_video. is_fmff's original bytes are
    passed straight through either way: to .fmff they're already exactly
    that (a plain copyfile, same as _save_audio/_save_video do for their
    own already-.fmff case); to a document extension,
    FMFFDecoder.extract_document recovers the exact original bytes
    FMFFEncoder.encode_document stored, byte-for-byte."""
    if dest_ext == ".fmff":
        if is_fmff:
            shutil.copyfile(source_path, path)
        else:
            FMFFEncoder().encode_document(source_path, path)
    elif dest_ext in DOCUMENT_EXTS:
        if is_fmff:
            FMFFDecoder(source_path).extract_document(path)
        else:
            shutil.copyfile(source_path, path)
    else:
        raise ValueError(f"can't save a document as {dest_ext or '(no extension)'}")
    progress_cb(100)


# ------------------------------------------------------------------------ CLI

DEFAULT_QUALITY = 80
ALREADY_LOSSY_QUALITY = 60
ALREADY_LOSSY_EXTS = {".jpg", ".jpeg"}
_ANIMATED_CHECK_EXTS = {".gif", ".webp", ".apng", ".png"}


def _default_quality_for(input_path, requested):
    """Auto-lower the lossy-tile quality unless the caller set one
    explicitly, for two different kinds of source that both come out
    *larger* than the source at the full default quality otherwise:

    - An already-lossy source (JPEG): detail it already threw away can't
      come back, so preserving it at high fidelity just spends bytes for
      nothing -- re-encoding a real photo through FMFF at the default
      quality routinely comes out larger than the JPEG it started from.

    - An animated GIF/WebP/APNG source: every stored frame/region is
      raced losslessly (palette/PNG-filter) *and* lossily (DCT, see
      _race_color_candidates) and whichever is smaller is kept, so this
      never makes a low-color-count, easily-paletted frame worse. But a
      busy, densely-dithered animation (the kind GIF's own LZW happens
      to compress unusually well, and the kind that pushes FMFF's own
      candidates to pick the lossy one) can end up noticeably bigger
      than the source at quality=80 -- measured 42% bigger on one real
      57-frame test clip. That gap almost entirely tracked the lossy
      candidate's own size: quality=60 alone took the same file from
      42% bigger to 13% *smaller*, with no measured regression on
      animations that were already winning via the lossless/palette
      candidates instead (quality only changes the losing candidate's
      size for those)."""
    if requested is not None:
        return requested
    ext = Path(input_path).suffix.lower()
    if ext in ALREADY_LOSSY_EXTS:
        return ALREADY_LOSSY_QUALITY
    if ext in _ANIMATED_CHECK_EXTS:
        try:
            if getattr(Image.open(input_path), "n_frames", 1) > 1:
                return ALREADY_LOSSY_QUALITY
        except (OSError, ValueError):
            pass
    return DEFAULT_QUALITY


def cmd_encode(args):
    src_size = Path(args.input).stat().st_size

    if Path(args.input).suffix.lower() in VIDEO_EXTS:
        enc = FMFFEncoder(tile_size=args.tile_size, thumb_max=args.thumb_max)
        print(f"encoding {args.input} -> {args.output} via FFmpeg (AV1/libsvtav1 + Opus)")

        def progress(i, total):
            if i % 25 == 0:
                if total:
                    print(f"  frame {i}/{total} ({100 * i // total}%)...")
                else:
                    print(f"  frame {i}...")

        stats = enc.encode_video(args.input, args.output, crf=args.crf, speed=args.speed,
                                  fps=args.fps, progress_cb=progress)
        print(f"  {stats['width']}x{stats['height']}, {stats['frame_count']} frames @ "
              f"{stats['fps']:.2f} fps, audio={stats['audio_tracks']} track(s), "
              f"subtitles={stats['subtitle_tracks']} track(s)")
        print(f"  size: {src_size} -> {stats['size']} bytes "
              f"({_fmt_size(src_size)} -> {_fmt_size(stats['size'])}, {_fmt_saving(src_size, stats['size'])})")
    elif Path(args.input).suffix.lower() in AUDIO_EXTS:
        enc = FMFFEncoder(thumb_max=args.thumb_max)
        print(f"encoding {args.input} -> {args.output} via FFmpeg (Opus)")

        last_reported = [-5000]

        def progress(ms):
            # No reliable total to report a percentage against (see
            # _encode_fragmented_audio), so just a periodic elapsed-time
            # ping instead of video's frame/total style.
            if ms - last_reported[0] >= 5000:
                last_reported[0] = ms
                print(f"  {ms / 1000:.1f}s encoded...")

        stats = enc.encode_audio(args.input, args.output, bitrate=args.audio_bitrate,
                                  progress_cb=progress)
        if args.audio_bitrate is None and stats["bitrate"] != f"{DEFAULT_AUDIO_BITRATE_KBPS}k":
            src_kbps = stats["source_bit_rate"] / 1000
            print(f"  note: {args.input}'s own bitrate is ~{src_kbps:.0f}k -- using "
                  f"{stats['bitrate']} Opus instead of the default "
                  f"{DEFAULT_AUDIO_BITRATE_KBPS}k so the FMFF copy doesn't end up "
                  f"bigger than the source (pass --audio-bitrate to override)")
        print(f"  {stats['duration_ms'] / 1000:.1f}s, {stats['sample_rate']} Hz, "
              f"{stats['channels']}ch, {stats['bitrate']} Opus, {stats['segments']} segment(s)")
        print(f"  size: {src_size} -> {stats['size']} bytes "
              f"({_fmt_size(src_size)} -> {_fmt_size(stats['size'])}, {_fmt_saving(src_size, stats['size'])})")
    else:
        quality = _default_quality_for(args.input, args.quality)
        enc = FMFFEncoder(tile_size=args.tile_size, quality=quality, thumb_max=args.thumb_max)
        stats = enc.encode(args.input, args.output,
                            version_name=args.version_name, version_note=args.version_note)
        print(f"encoded {args.input} -> {args.output}")
        if stats.get("mode") == "jpeg-passthrough":
            if stats["recompressed"]:
                print(f"  JPEG passthrough: re-entropy-coded the source's own DCT coefficients "
                      f"losslessly (no quality change, quality={quality} was not used)")
            else:
                print(f"  JPEG passthrough: recompressing this file's coefficients didn't beat "
                      f"its own (likely already-optimized) Huffman tables, so the original JPEG "
                      f"bytes were stored as-is instead -- still lossless, just not smaller")
            print(f"  {stats['width']}x{stats['height']}")
        elif stats.get("mode") == "document-passthrough":
            if stats["compressed"]:
                print(f"  document passthrough ({stats['source_ext']}): recompressed losslessly "
                      f"with a general-purpose compressor (smaller than the source, quality="
                      f"{quality} was not used -- this is lossless, not a picture of the pages)")
            else:
                print(f"  document passthrough ({stats['source_ext']}): recompression didn't beat "
                      f"the source's own encoding, so the original bytes were stored as-is instead "
                      f"-- still exact, just not smaller")
            print(f"  {stats['width']}x{stats['height']} (first-page preview thumbnail only -- "
                  f"the whole document is stored, not rendered pages)")
        else:
            if args.quality is None and quality != DEFAULT_QUALITY:
                if Path(args.input).suffix.lower() in ALREADY_LOSSY_EXTS:
                    reason = ("is already lossy-compressed and JPEG passthrough wasn't usable "
                              "for it (progressive scan, or jpeglib not installed)")
                else:
                    reason = "is an animated source (see _default_quality_for)"
                print(f"  note: {args.input} {reason} -- using quality={quality} instead of "
                      f"the default {DEFAULT_QUALITY} so the FMFF copy doesn't end up bigger "
                      f"than the source (pass --quality to override)")
            frames_note = f", {stats['frames']} frames" if "frames" in stats else ""
            # An animated encode has no fixed tile count to report (see
            # encode_image_sequence) -- only a still image's stats carry "tiles".
            region_note = f"{stats['tiles']} tiles" if "tiles" in stats else "no fixed tile grid"
            print(f"  {stats['width']}x{stats['height']}, alpha={stats['has_alpha']}, "
                  f"{region_note}{frames_note}, quality={quality}")
            if args.version_name or args.version_note:
                print(f"  version 0 ({args.version_name or 'original'!r}): "
                      f"{args.version_note or '(no note)'!r}")
        print(f"  size: {src_size} -> {stats['size']} bytes "
              f"({_fmt_size(src_size)} -> {_fmt_size(stats['size'])}, {_fmt_saving(src_size, stats['size'])})")


def cmd_add_version(args):
    enc = FMFFEncoder()
    src_size = Path(args.fmff).stat().st_size
    try:
        result = enc.add_version(args.fmff, args.input, name=args.name, note=args.note)
    except ValueError as exc:
        raise SystemExit(str(exc))
    new_size = Path(args.fmff).stat().st_size
    print(f"added version {result['version_index']} ({result['name']!r}) to {args.fmff}")
    print(f"  version size: {_fmt_size(result['size'])}")
    print(f"  file size: {_fmt_size(src_size)} -> {_fmt_size(new_size)} "
          f"({_fmt_saving(src_size, new_size)})")


def cmd_add_mask(args):
    enc = FMFFEncoder()
    src_size = Path(args.fmff).stat().st_size
    version = None
    if args.version is not None:
        version = _resolve_version_arg(args.version, FMFFDecoder(args.fmff).list_versions())
    try:
        result = enc.add_mask(args.fmff, args.mask, version=version)
    except ValueError as exc:
        raise SystemExit(str(exc))
    new_size = Path(args.fmff).stat().st_size
    print(f"added mask to version {result['version_index']} of {args.fmff} "
          f"({_fmt_size(result['size'])})")
    print(f"  file size: {_fmt_size(src_size)} -> {_fmt_size(new_size)} "
          f"({_fmt_saving(src_size, new_size)})")


def _resolve_version_arg(value, versions):
    """CLI --version can name a version either by its numeric index or by
    its --name (see add_version/cmd_add_version) -- this turns whichever
    was given into the index list_versions()/full(version=...) expect,
    or raises a clear SystemExit listing what's actually in the file."""
    available = ", ".join(f"{v['index']}:{v['name']}" for v in versions)
    try:
        idx = int(value)
    except ValueError:
        matches = [v for v in versions if v["name"] == value]
        if not matches:
            raise SystemExit(f"no version named {value!r} -- available: {available}")
        return matches[0]["index"]
    if idx not in {v["index"] for v in versions}:
        raise SystemExit(f"no version {idx} -- available: {available}")
    return idx


def cmd_decode(args):
    dec = FMFFDecoder(args.input)
    out_ext = Path(args.output).suffix.lower()

    if getattr(args, "mask", False):
        # Still-image-only, and orthogonal to every other content-type
        # branch below -- resolved and handled up front rather than
        # threaded through the video/audio/document/animation logic that
        # follows, none of which a mask has anything to do with.
        versions = dec.list_versions()
        if not versions:
            raise SystemExit("--mask only works on a still-image .fmff (see add-mask)")
        version_index = (_resolve_version_arg(args.version, versions)
                          if args.version is not None else versions[-1]["index"])
        try:
            mask_img = dec.extract_mask(version=version_index)
        except ValueError as exc:
            raise SystemExit(str(exc))
        mask_img.save(args.output)
        print(f"decoded mask for version {version_index} of {args.input} -> {args.output} "
              f"({dec.width}x{dec.height})")
        return

    if dec.is_document:
        # No transcoding to do or choose between -- the stored bytes ARE
        # the original document (see FMFFEncoder.encode_document), so
        # this just writes them back out exactly, regardless of what
        # extension args.output happens to have.
        doc_ext = dec.tags.get("doc_ext", "")
        if out_ext and out_ext != doc_ext:
            print(f"  note: this document was originally {doc_ext or 'unknown'} -- writing "
                  f"the recovered bytes to {args.output} as given, extension mismatch and all")
        dec.extract_document(args.output)
        print(f"decoded {args.input} -> {args.output} "
              f"({dec.media_blob_length} bytes recovered, byte-for-byte)")
        return

    if dec.is_video:
        # The .mp4 fast path skips FFmpeg entirely (extract_media() alone
        # is already a valid, playable file) -- but that raw fragmented
        # blob never had the source's tags muxed into it to begin with
        # (they're carried separately, in FMFF's own metadata blob, not
        # inside the AV1/Opus stream -- see encode_video), so restoring
        # them means an FFmpeg remux even for a .mp4 output, same as any
        # other extension.
        if out_ext == ".mp4" and not dec.tags:
            dec.extract_media(args.output)
        elif out_ext in VIDEO_EXTS or (out_ext == ".mp4" and dec.tags):
            ffmpeg = _find_ffmpeg()
            if ffmpeg is None:
                raise SystemExit("writing this format needs FFmpeg on PATH "
                                  "(or pass a .mp4 output to just extract the reconstructed "
                                  "stream directly)")
            with tempfile.TemporaryDirectory(prefix="fmff_dec_") as tmp:
                blob_path = os.path.join(tmp, "video.mp4")
                dec.extract_media(blob_path)
                # -map 0 (every stream in the input), not FFmpeg's default
                # "best of each type" auto-selection -- a video .fmff can
                # carry several audio/subtitle tracks (see encode_video),
                # and -c copy alone would silently drop every one but the
                # first of each type otherwise (confirmed directly: a
                # real 2-audio/2-subtitle source came out of this remux
                # with only 1 audio track and no subtitles at all before
                # -map 0 was added here).
                subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                                 "-i", blob_path, "-map", "0", "-c", "copy",
                                 *_ffmpeg_metadata_args(dec.tags), args.output], check=True)
        else:
            dec.thumbnail().save(args.output)
            print(f"decoded poster frame of {args.input} -> {args.output} "
                  f"(pass a video extension for --output to get the full clip)")
            return
        print(f"decoded {args.input} -> {args.output} "
              f"({dec.width}x{dec.height} @ {dec.fps:.2f} fps, audio={dec.has_audio})")
        if dec.last_corrupt_segments:
            missing_s = dec.last_corrupt_segments * VIDEO_SEGMENT_SECONDS
            print(f"  warning: {dec.last_corrupt_segments} of {dec.segment_count} video "
                  f"segment(s) failed their CRC32 check and were dropped "
                  f"(~{missing_s}s of playback may be missing)")
        if dec.has_video_alpha:
            if getattr(args, "alpha_output", None):
                dec.extract_alpha_media(args.alpha_output)
                print(f"  alpha track -> {args.alpha_output} "
                      f"(grayscale AV1; composite it back over the color track yourself -- "
                      f"{args.output} alone has no transparency)")
                if dec.last_corrupt_alpha_segments:
                    print(f"  warning: {dec.last_corrupt_alpha_segments} of "
                          f"{dec.alpha_segment_count} alpha segment(s) failed their CRC32 "
                          f"check and were dropped")
            else:
                print(f"  note: this video has an alpha track that was NOT extracted -- "
                      f"{args.output} has no transparency; pass --alpha-output PATH to get it")
    elif dec.is_audio:
        if out_ext == ".mp4" and not dec.tags:
            dec.extract_media(args.output)
        else:
            ffmpeg = _find_ffmpeg()
            if ffmpeg is None:
                raise SystemExit("writing this format needs FFmpeg on PATH "
                                  "(or pass a .mp4 output to just extract the reconstructed "
                                  "Opus stream directly)")
            with tempfile.TemporaryDirectory(prefix="fmff_dec_") as tmp:
                blob_path = os.path.join(tmp, "audio.mp4")
                dec.extract_media(blob_path)
                # -c:a copy where the container allows keeping Opus as-is
                # (.m4a, .opus/.ogg); anything else (.mp3, .wav, .flac, ...)
                # needs FFmpeg to actually transcode, since those containers
                # can't hold an Opus stream verbatim -- letting FFmpeg pick
                # the codec for out_ext's default rather than forcing one.
                # Either way the source's own tags (see encode_audio) are
                # restored here too -- they never made it into the Opus
                # stream itself, only into FMFF's own metadata blob.
                meta_args = _ffmpeg_metadata_args(dec.tags)
                try:
                    subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                                     "-i", blob_path, "-c:a", "copy",
                                     *meta_args, args.output], check=True)
                except subprocess.CalledProcessError:
                    subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                                     "-i", blob_path, *meta_args, args.output], check=True)
        print(f"decoded {args.input} -> {args.output} "
              f"({dec.duration_ms / 1000:.1f}s, {dec.sample_rate} Hz, {dec.channels}ch)")
        if dec.last_corrupt_segments:
            missing_s = dec.last_corrupt_segments * VIDEO_SEGMENT_SECONDS
            print(f"  warning: {dec.last_corrupt_segments} of {dec.segment_count} audio "
                  f"segment(s) failed their CRC32 check and were dropped "
                  f"(~{missing_s}s of playback may be missing)")
    elif dec.is_animated and out_ext in (".gif", ".webp", ".png", ".apng"):
        # full_sequence() decodes all frames in one linear pass rather than
        # calling full(frame=i) per frame (see its docstring for why that
        # loop is quadratic in frame count -- the actual cause of a slow
        # open on an animated file with many frames).
        frames = dec.full_sequence()
        total_corrupt = dec.last_corrupt_tiles
        durations = dec.frame_durations_ms()
        # Pillow's WebP writer defaults to lossy (~quality 75) even for
        # frames that came out of an exact/lossless FMFF decode -- without
        # this, decoding to .webp silently re-introduces a second lossy
        # pass on top of whatever FMFF already did, which showed up as a
        # large, easy-to-mistake-for-a-decoder-bug color shift in
        # partially-transparent RGBA content specifically.
        save_kwargs = {"lossless": True} if out_ext == ".webp" else {}
        if dec.exif_bytes:
            save_kwargs["exif"] = dec.exif_bytes
        if dec.icc_bytes:
            save_kwargs["icc_profile"] = dec.icc_bytes
        frames[0].save(args.output, save_all=True, append_images=frames[1:],
                        duration=durations, loop=0, **save_kwargs)
        print(f"decoded {args.input} -> {args.output} "
              f"({dec.width}x{dec.height}, {dec.frame_count} frames)")
        if total_corrupt:
            print(f"  warning: {total_corrupt} tile(s) across all frames failed their CRC32 "
                  f"check and were filled with gray placeholders")
    else:
        if dec.is_animated:
            print(f"  note: {args.input} is animated ({dec.frame_count} frames) but {out_ext} "
                  f"doesn't support animation -- decoding frame 0 only "
                  f"(use .gif/.webp/.png for the full animation)")
        # --version picks a specific one by index or --name; with no
        # --version and more than one version stored, decode defaults to
        # the most recently added one (version control's usual "give me
        # the current state" default) rather than silently always meaning
        # version 0 -- see add_version/list_versions.
        version_index = 0
        versions = dec.list_versions()
        if args.version is not None:
            version_index = _resolve_version_arg(args.version, versions)
        elif len(versions) > 1:
            version_index = versions[-1]["index"]
        img = dec.full(version=version_index)
        # See the animated branch above for why .webp specifically needs
        # lossless=True: Pillow's WebP writer otherwise defaults to lossy
        # regardless of how exactly FMFF itself decoded the pixels.
        save_kwargs = {"lossless": True} if out_ext == ".webp" else {}
        if dec.exif_bytes:
            save_kwargs["exif"] = dec.exif_bytes
        if dec.icc_bytes:
            save_kwargs["icc_profile"] = dec.icc_bytes
        img.save(args.output, **save_kwargs)
        version_note = f", version {version_index}" if len(versions) > 1 else ""
        print(f"decoded {args.input} -> {args.output} ({dec.width}x{dec.height}{version_note})")
        if dec.last_corrupt_tiles:
            print(f"  warning: {dec.last_corrupt_tiles} tile(s) failed their CRC32 check "
                  f"and were filled with gray placeholders")


def _external_launch_command():
    """The argv[0..] prefix that re-invokes this same program from
    scratch -- cmd_register_filetype appends one subcommand plus "%1"
    to whatever this returns and writes the result into the registry.

    Two very different cases share this one call site: running from
    source (`python F.M.F.F.py ...`), where re-invoking means launching
    a *separate* python.exe against this .py file, versus a PyInstaller
    build (see this file's packaging notes), where the frozen .exe
    already *is* the whole program and takes subcommands directly --
    sys.executable there is the .exe itself, not some interpreter next
    to it, and there's no separate script path to point at (`__file__`
    inside a frozen build resolves into PyInstaller's own temporary
    unpack directory, not anywhere a double-click command should name).

    Only the from-source case bothers with pythonw.exe (next to
    whichever interpreter is running this, if there is one): that's
    what keeps double-click from flashing a console window open (a
    plain .mp4/.mp3/.png double-click never does either). A frozen
    build has no such windowed twin to swap in -- see this file's
    packaging notes for the console-flash trade-off that leaves an
    onefile build with instead."""
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}"'
    candidate = Path(sys.executable).with_name("pythonw.exe")
    exe = str(candidate) if candidate.exists() else sys.executable
    script_path = str(Path(__file__).resolve())
    return f'"{exe}" "{script_path}"'


def cmd_open_external(args):
    """Headless (no Tk window of FMFF's own) equivalent of double-
    clicking a plain image/video/audio file: convert the .fmff to
    whatever ordinary format its content already reduces to, and hand
    that off to an external player instead of FMFF's own viewer.

    Video, audio, document, and JPEG-passthrough content already ARE a
    standard file/codec stream sitting under FMFF's own header (see the
    module docstring's container-layout sections and encode_document)
    -- extracting that (via cmd_decode, reused as-is here rather than
    duplicated) is no lossier than cmd_decode already is on its own. A
    still or animated image is FMFF's own proprietary tile codec, with
    no equivalent standalone stream to just pull out -- getting pixels
    out of it always needs FMFF's own decoder, no way around that
    specific part -- but the *result* (an ordinary PNG/GIF/WebP written
    to a temp file) is standalone from that point on, so what actually
    reaches the external player is still a completely normal file it
    needs no FMFF-awareness to open.

    --choose shows Windows' own "Open with" picker (whatever's
    installed and registered, not a hand-maintained list here) instead
    of silently using the OS's remembered default handler for that
    output type."""
    _hide_console_window()
    dec = FMFFDecoder(args.input)
    if dec.is_video:
        out_ext = ".mp4"
    elif dec.is_audio:
        out_ext = ".opus" if _find_ffmpeg() else ".mp4"
    elif dec.is_jpeg_passthrough:
        out_ext = ".jpg"
    elif dec.is_document:
        out_ext = dec.tags.get("doc_ext", ".pdf")
    elif dec.is_animated:
        out_ext = ".webp" if dec.has_alpha else ".gif"
    else:
        out_ext = ".png"

    tmp_dir = tempfile.mkdtemp(prefix="fmff_open_")
    out_path = os.path.join(tmp_dir, "content" + out_ext)
    cmd_decode(argparse.Namespace(input=args.input, output=out_path, alpha_output=None, version=None))

    if args.choose:
        if os.name != "nt":
            raise SystemExit("--choose (Windows' 'Open with' picker) is Windows-only -- "
                              "omit it to use the OS default handler instead")
        subprocess.Popen(["rundll32.exe", "shell32.dll,OpenAs_RunDLL", out_path])
    else:
        _open_with_default_player(out_path)
    print(f"opened {args.input} externally as {out_path}")


_FMFF_PROGID = "FMFF.MediaFile"


def cmd_register_filetype(args):
    """Register .fmff as a real Windows file type for the CURRENT USER
    only (HKEY_CURRENT_USER\\Software\\Classes -- no admin rights
    needed, and nothing outside this one Windows account is touched),
    so double-clicking a .fmff file in Explorer behaves like double-
    clicking any other media file instead of Windows asking "how do you
    want to open this file" with no memory of the answer every time.

    --mode external (the default) points a double-click at
    `open-external` (see cmd_open_external): convert to an ordinary
    format and hand it straight to the system's default player for
    that format -- the same experience as opening a plain .mp4/.mp3/
    .png, FMFF's own viewer window never appears. --mode viewer points
    it at `view` instead, opening FMFF's own GUI (progressive tile
    loading, in-window playback, etc. -- everything `open-external`
    deliberately skips).

    Either way this only sets what a plain double-click does --
    right-click -> "Open with" still lists this registration as a
    choosable app regardless (Windows populates that list from any
    ProgID with an HKCU shell\\open\\command, not from whichever one is
    currently the default), so a specific other player is always still
    pickable by hand."""
    if winreg is None:
        raise SystemExit("registering a file type needs the `winreg` module, "
                          "which only exists on Windows")
    verb = "open-external" if args.mode == "external" else "view"
    command = f'{_external_launch_command()} {verb} "%1"'

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\Classes\.fmff") as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, _FMFF_PROGID)
    prog_key = rf"Software\Classes\{_FMFF_PROGID}"
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, prog_key) as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, "FMFF Media File")
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, prog_key + r"\shell\open\command") as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, command)

    print(f".fmff registered for the current user -> {verb}")
    print(f"  command: {command}")
    print("double-clicking a .fmff file now uses this -- no restart needed, though Explorer "
          "may need reopening for an icon change to show (this doesn't set one).")


def cmd_unregister_filetype(args):
    """Undo cmd_register_filetype -- deletes only the keys it created
    (HKCU .fmff and the FMFF ProgID, recursively), leaving every other
    registry entry untouched."""
    if winreg is None:
        raise SystemExit("`winreg` only exists on Windows")

    def _delete_tree(root, subkey):
        try:
            with winreg.OpenKey(root, subkey) as k:
                while True:
                    try:
                        child = winreg.EnumKey(k, 0)
                    except OSError:
                        break
                    _delete_tree(root, subkey + "\\" + child)
        except FileNotFoundError:
            return
        winreg.DeleteKey(root, subkey)

    _delete_tree(winreg.HKEY_CURRENT_USER, r"Software\Classes\.fmff")
    _delete_tree(winreg.HKEY_CURRENT_USER, rf"Software\Classes\{_FMFF_PROGID}")
    print(".fmff file-type registration removed for the current user")


def cmd_info(args):
    dec = FMFFDecoder(args.input)
    if dec.is_audio:
        print(f"FMFF v{dec.version}  audio")
    else:
        print(f"FMFF v{dec.version}  {dec.width}x{dec.height}  alpha={dec.has_alpha}")
    if dec.is_audio:
        print(f"audio: {dec.duration_ms / 1000:.1f}s, {dec.sample_rate} Hz, {dec.channels}ch")
        print(f"media blob: {dec.media_blob_length} bytes across {dec.segment_count} "
              f"CRC32-checked segment(s) of ~{VIDEO_SEGMENT_SECONDS}s each (Opus via FFmpeg, "
              f"not FMFF's own codec)")
        print(f"thumbnail {dec.thumb_w}x{dec.thumb_h} (cover art, if the source had any)")
    elif dec.is_video:
        duration = dec.frame_count / dec.fps if dec.fps else 0.0
        print(f"video: {dec.frame_count} frames @ {dec.fps:.2f} fps (~{duration:.1f}s), "
              f"audio={dec.has_audio}")
        print(f"media blob: {dec.media_blob_length} bytes across {dec.segment_count} "
              f"CRC32-checked segment(s) of ~{VIDEO_SEGMENT_SECONDS}s each "
              f"(AV1 + Opus via FFmpeg, not FMFF's own codec)")
        if dec.has_video_alpha:
            print(f"alpha track: {dec.alpha_media_blob_length} bytes across "
                  f"{dec.alpha_segment_count} CRC32-checked segment(s) (grayscale AV1, "
                  f"no audio) -- decode with --alpha-output to get it")
        print(f"thumbnail {dec.thumb_w}x{dec.thumb_h}")
    elif dec.is_jpeg_passthrough:
        with open(dec.path, "rb") as f:
            f.seek(dec.media_blob_offset)
            kind = f.read(1)
        if kind == b"R":
            print("JPEG passthrough: original JPEG bytes stored verbatim "
                  "(recompression didn't beat this file's own Huffman tables)")
        else:
            print("JPEG passthrough: source's own DCT coefficients, losslessly re-entropy-coded "
                  "(not FMFF's own tile codec)")
        print(f"blob: {dec.media_blob_length} bytes")
        print(f"thumbnail {dec.thumb_w}x{dec.thumb_h}")
    elif dec.is_document:
        with open(dec.path, "rb") as f:
            f.seek(dec.media_blob_offset)
            kind = f.read(1)
        doc_ext = dec.tags.get("doc_ext", "?")
        if kind == b"R":
            print(f"document passthrough ({doc_ext}): original bytes stored verbatim "
                  f"(recompression didn't beat the source's own encoding)")
        else:
            print(f"document passthrough ({doc_ext}): recompressed losslessly with a "
                  f"general-purpose compressor (smaller than the source)")
        print(f"blob: {dec.media_blob_length} bytes")
        print(f"thumbnail {dec.thumb_w}x{dec.thumb_h} (first-page preview only -- "
              f"the full document is the blob above, not a per-page render)")
    else:
        color_entries = [e for e in dec.entries if e.layer == LAYER_FULL and e.plane == PLANE_COLOR]
        if not dec.has_fixed_tile_grid:
            # No fixed tile grid (see encode_image_sequence) --
            # tile_size/tiles_x/tiles_y aren't meaningful here, so show
            # the changed-region sizes instead.
            sizes = [e.w * e.h for e in color_entries]
            avg_px = sum(sizes) / len(sizes) if sizes else 0
            n_frames_changed = len({e.frame for e in color_entries})
            kind = (f"animated image, {dec.frame_count} frames" if dec.frame_count > 1
                    else "single-region image (no fixed tile grid)")
            print(f"{kind}, quality={dec.quality}")
            # A changed frame can now be split across more than one region
            # (see _split_changed_regions), so region count and changed-frame
            # count can differ -- both are worth showing.
            print(f"  {n_frames_changed} changed frame(s) stored as {len(color_entries)} "
                  f"region(s), avg region ~{avg_px:.0f}px^2 of {dec.width * dec.height}px^2 "
                  f"full frame")
        else:
            print(f"still image, tile_size={dec.tile_size}  grid={dec.tiles_x}x{dec.tiles_y}  "
                  f"quality={dec.quality}")
        print(f"thumbnail {dec.thumb_w}x{dec.thumb_h}")
        print(f"index entries: {len(dec.entries)} (read from the first "
              f"{HEADER_SIZE + len(dec.entries) * ENTRY_SIZE} bytes only)")
        lossless = sum(1 for e in color_entries if e.mode == MODE_LOSSLESS)
        lossy = sum(1 for e in color_entries if e.mode == MODE_LOSSY)
        palette = sum(1 for e in color_entries if e.mode == MODE_PALETTE)
        noun = "regions" if not dec.has_fixed_tile_grid else "tiles"
        print(f"color {noun}: {lossless} lossless, {lossy} lossy, {palette} palette"
              f"{' (summed across all frames)' if dec.frame_count > 1 else ''}")
    if dec.tags:
        tag_str = ", ".join(f"{k}={v}" for k, v in dec.tags.items())
        print(f"tags: {tag_str}")
    if dec.exif_bytes:
        print(f"EXIF: {len(dec.exif_bytes)} bytes (camera/GPS/orientation, passed through verbatim)")
    if dec.icc_bytes:
        print(f"ICC profile: {len(dec.icc_bytes)} bytes, passed through verbatim")
    versions = dec.list_versions()
    # Only worth a section when there's actually more than the one version
    # every still image already has, or that lone version was explicitly
    # named/noted (see FMFFEncoder.encode's --version-name/--version-note)
    # -- an ordinary unnamed still stays silent about this exactly like it
    # would have before add_version existed.
    has_extra_info = any(v["note"] or v["added"] or v["has_mask"] for v in versions)
    if len(versions) > 1 or has_extra_info:
        print(f"versions ({len(versions)}):")
        for v in versions:
            added = f", added {v['added']}" if v["added"] else ""
            mask = f", mask {_fmt_size(v['mask_size'])}" if v["has_mask"] else ""
            note = f" -- {v['note']}" if v["note"] else ""
            print(f"  [{v['index']}] {v['name']!r}: {_fmt_size(v['size'])}{mask}{added}{note}")


def cmd_view(args):
    _hide_console_window()
    MediaViewer(args.input, simulate_slow=args.simulate_slow).run()


def main():
    p = argparse.ArgumentParser(prog="fmff", description="FMFF encoder / decoder / viewer")
    sub = p.add_subparsers(dest="command", required=True)

    pe = sub.add_parser("encode", help="encode an image (FMFF's own codec), video (AV1/Opus via "
                                        "FFmpeg), or audio (Opus via FFmpeg) into .fmff")
    pe.add_argument("input")
    pe.add_argument("output")
    pe.add_argument("--quality", type=int, default=None,
                     help=f"image only: lossy tile quality 1-100 (default {DEFAULT_QUALITY}, "
                          f"or {ALREADY_LOSSY_QUALITY} when re-encoding an already-lossy "
                          f"source like JPEG or an animated GIF/WebP/APNG)")
    pe.add_argument("--tile-size", type=int, default=None,
                     help=f"still image only (default {STILL_TILE_SIZE}) -- an animated "
                          f"source has no fixed tile grid and ignores this")
    pe.add_argument("--thumb-max", type=int, default=64)
    pe.add_argument("--fps", type=float, default=None,
                     help="video only: resample to this frame rate (default: keep source fps)")
    pe.add_argument("--crf", type=int, default=30,
                     help="video only: AV1 quality, 0-63, lower = better/bigger (default 30)")
    pe.add_argument("--speed", type=int, default=8,
                     help="video only: SVT-AV1 encoder preset, 0-13, higher = faster (default 8)")
    pe.add_argument("--audio-bitrate", default=None,
                     help=f"audio only: Opus bitrate, e.g. 96k/128k/192k (default "
                          f"{DEFAULT_AUDIO_BITRATE_KBPS}k, auto-lowered for an "
                          f"already-lossy source with a lower bitrate of its own -- "
                          f"see _default_audio_bitrate)")
    pe.add_argument("--version-name", default=None,
                     help="still image only: label this file's version 0 (default: 'original' "
                          "once any version metadata exists -- see add-version)")
    pe.add_argument("--version-note", default=None,
                     help="still image only: freeform note for version 0 (see --version-name)")
    pe.set_defaults(func=cmd_encode)

    pav = sub.add_parser("add-version",
                          help="append a named version (e.g. a retouched copy) of a still image "
                               "to an existing .fmff, in place -- see README's 'Versions' section")
    pav.add_argument("fmff", help="existing still-image .fmff to add a version to")
    pav.add_argument("input", help="image for the new version -- must match the file's own "
                                    "width/height")
    pav.add_argument("--name", default=None, help="label for this version (default: 'version N')")
    pav.add_argument("--note", default=None,
                      help="freeform note describing this version / what changed")
    pav.set_defaults(func=cmd_add_version)

    pam = sub.add_parser("add-mask",
                          help="attach a selection mask (or any single-channel per-pixel "
                               "annotation) to one version of a still-image .fmff, in place -- "
                               "see README's 'Selection masks' section")
    pam.add_argument("fmff", help="existing still-image .fmff to attach a mask to")
    pam.add_argument("mask", help="grayscale mask image -- must match the file's own "
                                   "width/height")
    pam.add_argument("--version", default=None,
                      help="which version this mask belongs to, by index or --name "
                           "(default: the most recently added version)")
    pam.set_defaults(func=cmd_add_mask)

    pd = sub.add_parser("decode", help="decode .fmff into a normal image, video, or audio file")
    pd.add_argument("input")
    pd.add_argument("output")
    pd.add_argument("--alpha-output", metavar="PATH",
                     help="also extract the alpha track (video-with-transparency .fmff only) "
                          "as its own grayscale video file, e.g. for compositing elsewhere")
    pd.add_argument("--version", default=None,
                     help="still image only: which version to decode -- an index (0 = original) "
                          "or a --name from add-version (default: the most recently added "
                          "version, or plain version 0 if there's only the one)")
    pd.add_argument("--mask", action="store_true",
                     help="still image only: decode the selected version's attached mask (see "
                          "add-mask) instead of the image itself -- errors if it has none")
    pd.set_defaults(func=cmd_decode)

    pi = sub.add_parser("info", help="print header/index summary")
    pi.add_argument("input")
    pi.set_defaults(func=cmd_info)

    pv = sub.add_parser("view", help="open the media viewer (fmff / images / video)")
    pv.add_argument("input", nargs="?", default=None,
                     help="optional file to open on start (.fmff, image, or video)")
    pv.add_argument("--simulate-slow", type=float, default=0.0,
                     help="seconds to sleep per fmff tile, to visualize progressive loading")
    pv.set_defaults(func=cmd_view)

    px = sub.add_parser("open-external",
                         help="convert a .fmff to an ordinary format (mp4/opus/png/gif/webp) in a "
                              "temp file and hand it to an external player, instead of FMFF's own "
                              "viewer -- what register-filetype's default --mode wires a "
                              "double-click to")
    px.add_argument("input")
    px.add_argument("--choose", action="store_true",
                     help="show Windows' own 'Open with' picker instead of using the OS's "
                          "remembered default player for the converted file's type")
    px.set_defaults(func=cmd_open_external)

    pr = sub.add_parser("register-filetype",
                         help="register .fmff as a real Windows file type for the current user "
                              "only, so double-clicking one in Explorer works like any other "
                              "media file instead of prompting every time")
    pr.add_argument("--mode", choices=["external", "viewer"], default="external",
                     help="what a plain double-click does: 'external' (default) converts and "
                          "hands off to a normal player, exactly like opening a plain media "
                          "file; 'viewer' opens FMFF's own GUI instead")
    pr.set_defaults(func=cmd_register_filetype)

    pu = sub.add_parser("unregister-filetype", help="undo register-filetype")
    pu.set_defaults(func=cmd_unregister_filetype)

    argv = sys.argv[1:]
    known_commands = {"encode", "add-version", "add-mask", "decode", "info", "view",
                       "open-external", "register-filetype", "unregister-filetype"}
    if not argv:
        # No arguments at all -- a plain double-click of the .exe itself
        # (not via a file association, which always passes a path -- see
        # the branch below), the same way launching any other GUI app's
        # icon with nothing to open takes you to an empty window rather
        # than an error. Without this, argparse's required subparser
        # demands a command and refuses to run at all, which is fine
        # from a terminal (you'd just type the command you meant) but
        # not something a double-click can recover from.
        argv = ["view"]
    elif argv[0] not in known_commands and Path(argv[0]).exists():
        # bare `fmff.py somefile.ext` (e.g. from a file-association double-click)
        # opens straight in the viewer instead of requiring `view` first.
        argv = ["view"] + argv

    args = p.parse_args(argv)
    args.func(args)


def _hide_console_window():
    """Hides (not frees -- stdio keeps working exactly as before, just
    the window itself disappears) this process's own console window.
    Called right before entering the GUI (see cmd_view) and at the top
    of cmd_open_external, so the .exe can stay an ordinary console-
    subsystem build -- encode/decode/info/register-filetype/... keep
    working identically from any shell (cmd, PowerShell, or otherwise),
    with none of the fragility of trying to detect "was this launched
    from a terminal or a double-click" at startup and reopen stdio onto
    whatever's found (AttachConsole(ATTACH_PARENT_PROCESS) can do that,
    but proved unreliable to get right across every launch context --
    not worth it when hiding the window after the fact is this much
    simpler and has no such edge cases) -- while a double-click, file-
    association, or `view`/`open-external` launch still ends up
    window-less rather than sitting behind a visible console for as
    long as the program runs. Windows-only, and a no-op wherever
    there's no console to hide (e.g. running from source inside an
    IDE's own output pane, which owns no console window of its own to
    begin with)."""
    if os.name != "nt":
        return
    import ctypes
    hwnd = ctypes.windll.kernel32.GetConsoleWindow()
    if hwnd:
        SW_HIDE = 0
        ctypes.windll.user32.ShowWindow(hwnd, SW_HIDE)


_JOB_HANDLE = None  # kept alive for the process's whole lifetime, see below


def _setup_worker_job_object():
    """Ties every _get_tile_pool worker's life to this process's, so a
    hard crash or a Task-Manager "End task" on the main .exe can never
    leave orphaned worker processes running forever in the background
    (each one just sits idle waiting on the pool's task queue, so it
    doesn't show up as CPU usage -- only as unexplained extra
    F.M.F.F.exe entries and memory that never comes back).

    A normal exit doesn't need this: _shutdown_tile_pool already runs
    via atexit and terminates every worker cleanly. atexit only fires on
    a normal interpreter shutdown, though -- not on a forceful kill or a
    native-level crash (e.g. inside an image/video C extension), which
    is exactly when orphans were happening.

    The fix is a Windows Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE:
    this process is assigned to the job *before* any worker exists, and
    Windows automatically makes every child process created afterwards
    (each pool worker) a member of the same job too. Whenever the job's
    last open handle is closed -- which Windows does on its own the
    moment this process ends, however it ends -- every remaining member
    process is killed with it. The handle is stashed in the module-level
    _JOB_HANDLE precisely so nothing closes it early: closing it before
    the process exits would trigger that same kill immediately.

    Windows-only, and best-effort -- e.g. a host that already placed
    this process in a job without nested-job support (pre-Windows-8,
    or some sandboxes) will fail the assignment, in which case this
    quietly does nothing rather than treating it as fatal."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    global _JOB_HANDLE
    try:
        kernel32 = ctypes.windll.kernel32

        class _BASIC_LIMIT(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _IO_COUNTERS(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class _EXTENDED_LIMIT(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BASIC_LIMIT),
                ("IoInfo", _IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JobObjectExtendedLimitInformation = 9

        # ctypes assumes a 32-bit int return by default, which truncates
        # HANDLE (a pointer, 64-bit on x64) -- silently turning a
        # perfectly valid handle into something that reads as falsy, so
        # every call here needs its real Windows signature spelled out.
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return

        info = _EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
                job, JobObjectExtendedLimitInformation,
                ctypes.byref(info), ctypes.sizeof(info)):
            kernel32.CloseHandle(job)
            return

        if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
            kernel32.CloseHandle(job)
            return

        _JOB_HANDLE = job  # deliberately never closed -- see docstring
    except OSError:
        pass


if __name__ == "__main__":
    # Must be the very first thing that runs, before anything else --
    # including argument parsing -- when this has been frozen into a
    # Windows .exe (see this file's packaging notes). A frozen build has
    # no separate python.exe/script.py for multiprocessing.Pool's workers
    # (see _get_tile_pool) to launch the way it does running from source
    # -- each worker instead re-executes this same .exe from scratch,
    # and without freeze_support() telling it "this particular
    # relaunch is a worker bootstrap, not the app starting over", every
    # one of those workers would re-run main() itself: for an
    # encode_image_sequence job big enough to reach _MP_TILE_THRESHOLD,
    # that means each of the first pool's workers spinning up its OWN
    # full worker pool, recursively, without limit -- the exact runaway-
    # process/CPU-pegging failure mode a from-source multiprocessing
    # bug already caused once in this project (missing `if __name__ ==
    # "__main__":` in a one-off test script, not this file) -- so this
    # is not a hypothetical to skip.
    mp.freeze_support()
    # Must happen before the first _get_tile_pool() call (anywhere inside
    # main()) so every pool worker it spawns is already a child of a
    # job-assigned process and inherits membership automatically -- see
    # _setup_worker_job_object's docstring for why this exists at all.
    _setup_worker_job_object()
    # A Windows console's own encoding (cp1252, or another single-byte
    # codepage depending on locale) is what sys.stdout/stderr default to
    # here -- not UTF-8 -- so printing a path with non-Latin-1 characters
    # (Cyrillic, say) raises UnicodeEncodeError and crashes before doing
    # anything, independent of and in addition to the matching read-side
    # fix in _ffprobe_info (ffmpeg/ffprobe's own output is UTF-8
    # regardless of the console). errors="replace" is a last-resort
    # safety net for whatever this console genuinely can't render, not
    # the reason this works -- reconfiguring to UTF-8 is.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    sys.exit(main())
