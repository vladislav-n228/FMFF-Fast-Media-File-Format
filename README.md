# FMFF -- Fast Media File Format

An experimental image/video/audio/document container (`.fmff`) with a
reference Python encoder, decoder, and viewer -- also distributed as a
standalone Windows `.exe`, no Python install required (see "Windows
.exe" below).

**The one thing FMFF does that no mainstream image format does at all:**
keep more than one version of the same still image -- an original photo,
a retouched copy, a selection mask, whatever else got derived from it --
in one file, where a new version only costs bytes for the tiles that
actually changed: **0 bytes** for a version that changed nothing, a few
KB for a small retouch, not a second full copy every time. PNG, WebP, MP4,
and PDF have no concept of this at all -- the closest analogue elsewhere
is a version-control system, not an image format. See "Versions" below
for exactly how it works and what it costs on a real file.

Beyond that headline feature, this is a container where each content
type is handled very differently on purpose, with tradeoffs disclosed
rather than hidden -- see "Benchmarks" and "Status / limitations" below
before deciding whether it's useful for you. FMFF's own hybrid tile codec
(stills/animation) is genuinely this project's own design; video and
audio instead wrap FFmpeg's own AV1/Opus encoders rather than reinventing
them (see "Video" and "Audio" below for why that's a deliberate choice,
not a shortcut). This is a personal/hobby project, not a production
codec -- see "Status / limitations" for the honest scope.

## Images -- FMFF's own hybrid codec

This part is genuinely FMFF's own format, built around a few ideas:

- **Instant indexing** -- a fixed-size header at byte 0 plus a tile index
  right after it, so a reader knows where everything is without scanning
  the file.
- **Hybrid lossless/palette/lossy/raw tiles** -- each tile is raced four
  ways: a PNG-style lossless filter (zlib), a lossless palette/indexed-
  color encoding (a small per-tile color table plus a 1-byte-per-pixel
  index, for any tile with 256 or fewer distinct colors), a JPEG-style
  lossy DCT codec (whichever of zlib/bz2/zstd is smaller -- see below),
  and the tile's own
  raw, uncompressed bytes as a last-resort floor; whichever of the four
  comes out smallest is kept. Flat/text/line-art regions (and anything
  that started out palette-based, GIF included) stay pixel-exact and
  small, while photo/gradient regions compress hard. The raw candidate
  matters for small/edge tiles specifically -- found by digging into a
  real report of file bloat that turned out to have a different root
  cause (the "source" file was actually WebP wearing a `.png` extension,
  not a bug here -- see "Benchmarks" below), but the investigation
  surfaced a real, separate gap along the way: a tiny tile (the partial
  ones a non-tile-size-multiple image edge always has) could have every
  compressed candidate cost *more* than its own raw pixel data once
  zlib/bz2's own per-stream framing was added on top, with no floor to
  catch it -- confirmed directly: a 1x1 tile that used to cost more than
  its 3 raw bytes now costs exactly 3. The palette and raw candidates can
  each only ever win the race or lose it, never make a tile worse -- the
  other two are still computed and compared
  every time.
- **A third entropy-coding candidate, `zstandard` (optional -- `pip
  install zstandard`), races alongside zlib/bz2** -- the same tag-and-
  race pattern this file already used for lossless/palette/lossy/raw,
  just one level down, on the general-purpose compression step every
  lossy tile and every palette index stream already goes through. Level
  12, chosen the same way `_ZLIB_LEVEL` was: measured against zlib/bz2
  on real byte-packed tile data from an actual photo, not guessed --
  level 19+ measured ~0.4% smaller than 12 for roughly double the time,
  the same "diminishing returns past a point" shape; level 12 was both
  smaller *and* faster than zlib level 6 in that same measurement. Real
  effect on the benchmark images (see "Benchmarks" below): under 1% to
  ~9% smaller on top of everything else here, none of them bigger, and
  biggest on palette-heavy content (screenshot/pixel-art) since palette
  index bytes race through the same three compressors too. Optional and
  gracefully degrading like
  jpeglib/pymupdf/tkinterdnd2 elsewhere in this file: encoding without it
  installed just means zlib/bz2 alone contest that race (still correct,
  slightly bigger on average); a file that does have a zstd-compressed
  tile needs it installed to decode *that* tile, with a clear error
  naming exactly that instead of a bare crash if it's missing.
- **The lossy candidate now subsamples chroma, orders its coefficients
  the way real JPEG does, and packs them distribution-aware instead of
  at a fixed width** -- three separate gaps a from-scratch look at this
  codec's own code turned up, each closing part of the size gap against
  WebP at matched quality (see "Benchmarks" below): it used to transform
  Cb/Cr at the same full resolution as luma (no other lossy photo codec
  spends bits on chroma detail that way) -- now Cb/Cr are downsampled 2x
  per axis before transforming (4:2:0, same as JPEG/WebP). It used to
  serialize each block's quantized DCT coefficients in plain raster
  order, fixed at 2 bytes each, straight into zlib/bz2 -- now each
  block's coefficients are zigzag-reordered with the DC term delta-coded
  against the previous block's (the same DPCM trick this file already
  used for JPEG passthrough, see "JPEG sources" below), a block's
  trailing run of zeros past its last nonzero coefficient is cut
  outright (an EOB marker, not just compressed), and each surviving
  coefficient is usually one byte instead of a fixed two, with a rare
  out-of-range one escaping to its own slot instead of forcing every
  coefficient to pay for the exceptional one -- all still fully
  vectorized (no per-coefficient Python loop), short of real JPEG's own
  bit-level Huffman/arithmetic coding but built on the same two ideas.
  Verified bit-for-bit lossless relative to the pre-repack coefficients
  (same DCT, same quantization -- only how the result is packed into
  bytes changed) before ever measuring size. Together, measured on two
  synthetic photo-like benchmark images: output shrank 51-52% at the
  same quality setting versus before any of this, cutting the size gap
  against WebP at matched quality roughly in half again on top of the
  chroma/zigzag fix alone.
- **The base quantization table is the "ImageMagick community" table,
  not the 1992 JPEG Annex K default** -- checked against mozjpeg's own
  `jcparam.c` before assuming anything (mozjpeg's real advantage turned
  out to be trellis quantization and better Huffman tables, not a
  different default matrix -- its default is the same 1992 table this
  codec used before, so an earlier draft of this note was wrong to credit
  the swap to "mozjpeg's tables"). mozjpeg does bundle several published
  alternative tables as a selectable, non-default option, including one
  credited there to an ImageMagick community thread reporting a better
  quality/size trade-off for photographic content; swapped in and
  measured here rather than taken on faith. Real effect, on the same
  controlled before/after images (see "Benchmarks" below): meaningfully
  smaller on the smooth-photo and random-noise cases (-9%, -6%), roughly
  flat on flat/palette-friendly content, very slightly bigger (+1.9%) on
  the busier/textured photo case -- a genuine, disclosed trade-off, not a
  strict win everywhere.
- **True-color alpha** -- transparency is stored per-pixel (not 1-bit),
  losslessly.
- **Chunked/streamable** -- every tile has its own (offset, length) in the
  index, so a client could fetch a single tile via an HTTP Range request.
- **Thumbnail scaled to the source, not just to a fixed cap** -- capped at
  a third of the source's longest side (not always up to 64px), and given
  the same three-way lossless/palette/lossy race as a regular tile.
  Without this, a small and/or detailed source image's "instant preview"
  could rival or beat the size of the entire rest of the file -- it was
  58% of one 128x128 test file before this.
- **Error resilience** -- every tile carries a CRC32; a corrupt or
  not-yet-downloaded tile is detected and swapped for a gray placeholder
  instead of failing the whole image, and a truncated file still decodes
  whatever tiles did arrive. The header also carries a version field that
  a mismatched reader now rejects with a clear error instead of
  misparsing the file.
- **Multi-core encoding for large images** -- every tile's lossless/
  palette/lossy race is independent of every other tile's, so above a size
  threshold they're spread across a `multiprocessing` pool instead of
  encoded one at a time. Profiling a real encode showed the actual cost
  is ~60% `zlib.compress` and ~10% `bz2` -- both serial, CPU-only,
  per-tile algorithms with no GPU equivalent -- versus ~25% for the DCT
  math itself, which is why this is multiple CPU cores rather than a
  GPU kernel: it attacks the part that's actually slow. ~2.9x faster on
  a 3000x2000 test image on a 4-core/8-thread machine. Small images skip
  the pool (worker-process startup cost isn't worth it below the
  threshold), and the pool itself is created once and reused for the
  life of the process, not respawned per file, so batch-converting many
  large images only pays that startup cost once.
- **Multi-core decoding for large images, too** -- decoding was single-
  process-only until profiling a real ~12MP photo (a typical phone/
  camera resolution) showed opening it took 1.6s+ entirely on one core,
  visible as a real pause in the viewer between the window appearing and
  the picture actually rendering. Every tile's decode is exactly as
  independent as its encode, so it now goes through the same pool
  (`imap_unordered`, so tiles still stream back and reveal progressively
  in the viewer instead of only appearing all at once when the whole
  batch finishes). Measured on that same 12MP photo: ~1.2s on a cold
  pool (the first file opened in a session, paying worker-startup cost),
  **~0.5s once the pool is warm** (every file after that in the same
  session) -- a ~3.2x speedup for the common case of opening more than
  one image without restarting the app. A first attempt using
  `ThreadPoolExecutor` instead of processes measured *slower* than the
  original single-threaded code (the GIL isn't released enough during
  all these small, fast per-tile operations for threads to pay off) --
  worth knowing if this is revisited, so that mistake doesn't get
  repeated.
- **Animation** -- a multi-frame source (animated GIF/WebP/APNG) encodes
  as a single `.fmff` with more than one frame instead of collapsing to
  a still. This isn't a bolted-on addition: the container already had
  per-entry frame indices and a frame table (`Entry.frame`, the
  `frame_table`/`FRAME_FMT` fields in the header) from an earlier,
  pre-FFmpeg version of FMFF's own video codec -- unused by a single
  still image, which only ever fills in frame 0. Animation just puts
  that existing structure to use: every stored frame gets the same
  hybrid lossless/palette/lossy race and its own CRC32 as a plain still,
  so one corrupt frame degrades to a gray placeholder instead of
  breaking the whole animation. Beats GIF's 256-color palette outright,
  and is simpler/more broadly supported than APNG or animated WebP
  tooling. `encode` detects a multi-frame source automatically; `decode`
  writes a multi-frame `.gif`/`.webp`/`.png` when the output extension
  supports it, or frame 0 alone (with a note) otherwise. A frame
  pixel-identical to the previous one gets no entry at all -- decoding
  frame N just reuses whatever the previous frame already had. This
  matters a lot: a typical GIF (mostly-static background, a small part
  actually moving) encoded completely independently per frame re-stored
  that whole unchanged background every single frame, coming out 3-5x
  *larger* than the source GIF on a mostly-static test clip -- not
  competitive, a real regression, not just a missed optimization, since
  GIF's own LZW (and tools that re-save GIFs, which often crop each
  frame to just its changed region outright) already exploits exactly
  that redundancy.
- **Each changed frame stores one rectangle sized to its own change, not
  a fixed tile grid** -- earlier versions of this reused the still-image
  tile grid for animation (chop every frame into fixed tiles, skip a
  tile identical to the same grid cell in the previous frame, race
  lossless/palette/lossy for the rest). That's a real improvement over
  storing every frame independently, but it's still bound by the grid:
  a small, localized change (a spinner, a bit of moving text) that
  straddles a few grid tiles re-stores all of those tiles in full, most
  of whose area didn't actually change -- and a single tile covering the
  whole frame (tried as a fix) re-stores the *entire* frame the moment
  anything, anywhere, changes. Neither is content-shape-aware. So this
  doesn't tile the frame at all any more: for each frame after the
  first, the tight bounding box of every pixel that actually differs
  from the previous frame is computed directly, and exactly that
  rectangle -- positioned at its own (x, y), not a grid cell -- gets
  raced through the same lossless/palette/lossy codec and stored as a
  single entry. A small localized change stores a small rectangle; a
  frame-wide change stores close to the whole frame; either way it's one
  entry, so the fixed per-entry cost (index row, CRC32, compression
  per-stream overhead) is paid at most once per changed frame no matter
  the change's shape. On a real 550x400, 30-frame GIF with a small
  moving region on a large static canvas, no fixed tile size (from 32px
  up to one tile per frame) got the `.fmff` within 50% of the 40KB
  source -- this got it *smaller* than the source.
- **...or several rectangles, when the change isn't one blob** -- a
  single bounding box is only optimal when everything that changed is
  clustered together. Content like a logo whose highlight sweeps through
  more than one corner of an otherwise-static frame changes a handful of
  small, far-apart spots each frame; one bbox around all of them
  re-stores whatever untouched background sits between them too. So the
  changed-pixel mask is first connected-component-labeled on a coarse
  block grid (cheap -- cost tracks frame area, not changed-pixel count),
  nearby components are coalesced (two small rectangles close together
  cost more, each paying its own index row + CRC32 + compression
  overhead, than one slightly larger one covering both), and the result
  is however many tight, non-overlapping rectangles the change actually
  needs -- capped at a dozen or so per frame, past which this falls back
  to the single whole-frame bbox instead of paying for a pile of small
  entries. Measured on a real 129-frame, 1080x1080 animated logo where
  the highlight moves between corners: 17 of the 129 frames needed more
  than one region (up to 12 in a single frame), and the file came out
  6.6% smaller than the single-bbox approach on the same source.
- **Faster on the large regions rectangle-per-frame storage produces** --
  storing one rectangle per changed frame instead of many small fixed
  tiles (above) means a single "tile" can now be a real fraction of a
  large frame, and two costs that were negligible at old tile sizes
  turned out not to scale down gracefully: profiling a slow encode on a
  1080x1080, 129-frame source found `zlib.compress` alone was ~72% of
  total time, because every tile/region is compressed at zlib's max
  level (9) -- worth it for a small tile, wasteful for a big one, since
  9 buys very little extra ratio past level 6 on real image data (one
  large region measured: 0.062s for 34,945 bytes at level 9 versus
  0.009s for 36,350 bytes at level 6 -- 7x faster for 4% bigger).
  Decoding the same file was dominated by the DCT math itself: the
  lossy codec's block transform used `np.einsum` for a batched
  per-8x8-block matrix multiply, which doesn't route through BLAS the
  way plain `@` does for the exact same computation -- switching it to
  `@` measured ~19x faster with numerically identical output. Together
  these took that 1080x1080 test file from 27s encode / 32s decode to
  11s / 13s.
- **Lossy quality auto-lowered for animated sources, same reasoning as
  JPEG passthrough's fallback** -- a low-color-count or simple frame
  wins the lossless/palette race regardless of quality (quality only
  affects the size of the *lossy* candidate, so it can't hurt those
  cases either way). But a busy, densely-dithered animation -- the kind
  GIF's own LZW happens to compress unusually well, and the kind that
  makes FMFF's own lossy candidate the one that wins the race -- came
  out noticeably bigger than the source at the default quality (80):
  42% bigger on one real 57-frame test clip. That gap tracked the lossy
  candidate's own size almost exactly: quality 60 alone took the same
  file from 42% bigger to 13% *smaller*, with zero measured regression
  on animations already winning via lossless/palette. So an animated
  GIF/WebP/APNG source now defaults to quality 60 instead of 80 (same
  value and the same "already-lossy source" reasoning JPEG passthrough's
  fallback already used) unless `--quality` is passed explicitly.
- **Fast full-animation decoding** -- decoding a whole animation frame by
  frame by re-scanning the entire index on every single frame (to find
  whichever earlier frame last touched each part of the picture) costs
  O(frames x index size) -- quadratic in frame count, and the real
  reason opening an animated file with many frames could take much
  longer to read than it took to encode. `FMFFDecoder.full_sequence()`
  decodes every frame in one linear pass instead: it buckets the index
  by frame once, then walks frames in order, blitting each frame's
  rectangle (if it has one) directly onto its own pixel position on a
  running canvas -- a frame with no rectangle simply leaves the canvas
  as the previous frame left it. The viewer and the `decode` CLI command
  both use this for animations. On one 600-frame test clip, decoding the
  whole thing this way was 5-6x faster than the naive per-frame loop,
  and the gap widens with more frames since the old path was quadratic
  and this one isn't.

## Versions -- an edit history for one still image, in one file

No mainstream media container (WebP, PNG, MP4, PDF, ...) has a built-in
way to keep more than one version of the same content in one file with
small, per-version overhead -- an original photo plus a retouched copy,
a selection mask, or anything else derived from it either lives as
separate files or isn't kept at all. FMFF's own tile codec (see "Images"
above) already indexes every tile by `(offset, length)`, so this turned
out to need no new binary layout at all: a version (`add-version`) sits
in the same tile index under a new layer code (`LAYER_VERSION` instead
of `LAYER_FULL`), sharing the base image's tile grid and quality. A
reader that only ever looked for `LAYER_FULL`/`LAYER_THUMB` entries --
any FMFF build before this existed included -- simply never sees the
extra entries and keeps decoding the base image exactly as it always
did. Version labels/notes/timestamps ride in the same small JSON
metadata blob tags and EXIF already use (see "Metadata" below), under a
`"versions"` key.

**Each version stores only what actually changed**, not a second full
copy of the image -- the same "unchanged gets no entry at all" saving
this file's own animation encoding already gives an identical frame
(see "Each changed frame stores one rectangle" above), here applied to
a version chain instead of a time axis. Concretely: every tile is
re-encoded and compared, byte for byte, against whatever's currently
the effective encoding at that position (version 0's tile, or the
latest earlier version that touched it) -- comparing *encoded bytes*
rather than raw pixels, since a lossy tile's own decode is never
bit-exact and comparing against a decoded-and-requantized previous
version would flag nearly everything as "changed" from quantization
noise alone. A tile whose fresh encode matches isn't stored again.
Measured on a real 1600x1200 test photo (392KB as a single version):
a version that only retouches a small corner cost **3.5KB**, and a
version pixel-identical to its predecessor (relabeling only, no edit)
cost **0 bytes** of tile data -- versus a version that color-grades the
*entire* photo, which still cost close to a second full copy (393KB),
because that much of the image genuinely did change. This doesn't
invent savings that aren't there; it just stops charging for the tiles
that didn't need re-storing.

- **Scoped to still images on purpose.** Video, audio, JPEG-passthrough,
  and documents already store their content as one opaque blob apiece
  (see their own sections above) with no per-tile addressing to append a
  version into cheaply -- adding versioning there would mean a second,
  much heavier mechanism, not a small extension of an existing one. An
  *animated* image is excluded for a related reason: its entries carry
  literal per-frame pixel rectangles instead of a fixed tile grid (see
  "Each changed frame stores one rectangle" above), so there's no shared
  per-version geometry the way a plain still's tile grid gives for free.
- **Every version shares one quality setting.** A lossy tile's
  dequantization at decode time uses the single, file-wide `quality`
  header field -- there's nowhere in the format for a per-version
  quality override, so `add-version` always encodes at the file's own
  already-stored quality rather than accepting its own `--quality`.
- **Alpha is a whole-file decision, not a per-version one.** The
  header's `alpha=` flag is set once, from version 0, and every later
  version conforms to it (a fully-opaque plane filled in if its own
  source has none, or its alpha silently dropped if the file has none)
  -- letting alpha appear/disappear version-by-version would make "did
  this tile actually change" ambiguous right at that boundary, for no
  real benefit most still images need.
- **The trade-off**: every tile is still fully re-encoded during
  `add-version` to find out whether it changed (there's no way to know
  without racing it through the codec) -- the saving is in what gets
  *written* to disk, not in encode time.
- **Periodic full versions bound replay depth.** Reconstructing version N
  means replaying every version between it and the nearest earlier
  version that covers every tile -- left unbounded, a very long,
  heavily-edited chain would cost more and more to decode as it grew.
  So every 8th version (`VERSION_SNAPSHOT_INTERVAL`) is stored in full
  regardless of what changed, the same "periodic checkpoint" idea
  video's own I-frames use -- decoding version 1000 of a long chain
  doesn't replay 1000 versions, only back to the nearest multiple of 8.
  Measured on a real 20-version chain (each a small local edit): without
  this, decoding version 20 would replay all 20 deltas on top of version
  0; with it, decoding version 20 only replays back to version 16 (the
  nearest checkpoint) -- **104 tiles touched instead of 120**, a gap
  that only widens as a chain gets longer. The cost is one full version's
  worth of extra storage every 8 versions, not every version.
- The viewer isn't CLI-only for this any more: a toolbar dropdown lists
  every version of the open file and switches the displayed image to
  whichever one is picked, and an **Add version...** button appends a new
  one (pick the new version's image, optionally name/note it) without
  leaving the GUI -- both just call the same `add-version`/`list_versions`
  the CLI already used. `add-mask` (see "Selection masks" below) is still
  CLI-only.

```
python F.M.F.F.py encode photo.png photo.fmff --version-name original --version-note "as shot"
python F.M.F.F.py add-version photo.fmff retouched.png --name retouched --note "background removed, color graded"
python F.M.F.F.py add-version photo.fmff final.png --name final
python F.M.F.F.py info photo.fmff                          # lists every version, with its size/note/timestamp
python F.M.F.F.py decode photo.fmff out.png                 # no --version: decodes the most recently added one
python F.M.F.F.py decode photo.fmff out.png --version 0     # by index -- 0 is always the original
python F.M.F.F.py decode photo.fmff out.png --version retouched   # or by --name
```

## Selection masks -- an extra channel that isn't transparency

A still image already has a per-pixel alpha channel for transparency
(see "Images" above); a *mask* is the same idea -- a single 8-bit
channel, same tile grid, same lossless codec -- but never composited
into the picture. It's for a selection used to make a retouch, a
subject/background split, or any other per-pixel annotation a caller
wants back out unchanged, attached to a specific version (see
"Versions" above) via `add-mask`. This is close to a clone of the
existing alpha-plane machinery on purpose: a new tile plane
(`PLANE_MASK`) reuses the same lossless encode/decode path alpha
already has, just tagged so `full()`/`full_sequence()` skip it when
reconstructing the actual image -- only `decode --mask` reads it.

A mask belongs to exactly the version it was attached to, with no
inheritance across the version chain the way an unedited color/alpha
tile is inherited (see "Versions" above) -- a selection drawn for one
retouch isn't implicitly still correct for a different, later one, and
there'd be no reliable way to tell a deliberately-reused mask from one
that simply wasn't revisited. Calling `add-mask` again for a version
that already has one replaces it.

```
python F.M.F.F.py add-mask photo.fmff mask.png                    # attaches to the most recently added version
python F.M.F.F.py add-mask photo.fmff mask.png --version 0        # or a specific version, by index or --name
python F.M.F.F.py info photo.fmff                                  # shows "mask NNN B" next to any version that has one
python F.M.F.F.py decode photo.fmff mask_out.png --mask             # extracts the mask instead of the image
python F.M.F.F.py decode photo.fmff mask_out.png --mask --version 0
```

- CLI only, still-image-only, one mask per version -- same scope as
  Versions above, for the same reasons (no fixed tile grid to share on
  video/audio/JPEG-passthrough/documents or an animated image).
- The mask image must match the file's own width/height exactly.

## JPEG sources -- lossless coefficient passthrough

Re-running a JPEG through FMFF's own tile codec means re-quantizing pixels
that a lossy codec already quantized once -- that's a second lossy pass on
data whose easy redundancy is already gone, and it routinely comes out
*larger* than the JPEG, not smaller. So a JPEG source instead gets pulled
apart at the coefficient level: FMFF reads the JPEG's own already-quantized
DCT coefficients straight out of it (via `jpeglib`/libjpeg -- no IDCT, no
requantization, no pixel ever touched), delta-codes each block's DC
coefficient against the previous block's (neighboring patches of a photo
usually have similar average brightness, so this turns the DC stream
mostly small/zero instead of raw values -- a free few more percent in
every case tested), and re-entropy-codes the result with whichever of a
few general-purpose compressors is smallest. This reliably beats JPEG's
baseline Huffman coding by 10-40% on ordinarily-encoded JPEGs.
Reconstructed pixels match what the JPEG itself decodes to within a few
levels (bilinear chroma-upsampling filter choice -- the same kind of small
difference any two compliant JPEG decoders can show versus each other).

This needs `pip install jpeglib` and only handles baseline (non-
progressive), 3-component JPEG; anything else (progressive scan, missing
`jpeglib`) automatically falls back to the normal tile codec with quality
lowered to keep the result from growing past the source (see the
`_default_quality_for` note in the code).

Recompression doesn't always win, though: a JPEG already saved with
*optimized* (per-image) Huffman tables -- common from phone cameras and
apps that recompress photos for sharing -- can already be close to what a
general-purpose byte compressor can do to it. When recompression doesn't
actually come out smaller, FMFF stores the original JPEG bytes verbatim
instead of the recompressed form, so this path never makes the *content*
bigger than the source. It can still end up a little larger than the bare
source file, though: the container's own fixed costs (header, index, the
instant-preview thumbnail) have to go somewhere, and there's no way to
losslessly store N source bytes in fewer than N -- so in that fallback
case the file is source size plus a small, fixed amount for those (well
under 1% for any real photo; more noticeable only for a very small JPEG).

## Video -- wrapped AV1 + Opus via FFmpeg

Video does **not** use FMFF's own codec. A `.fmff` video file holds one
opaque, complete AV1 (video) + Opus (audio) stream produced by FFmpeg,
plus a small instant thumbnail (poster frame) encoded the same way
stills are.

This is a deliberate trade-off, not a shortcut: an early version of FMFF
used its own intra-only (no motion compensation) tile codec for video
too, and it could never get within an order of magnitude of a real video
codec's file size -- that's an architectural ceiling, not something more
tuning fixes. AV1 gives real inter-frame compression, real speed
(FFmpeg/SVT-AV1 parallelizes internally in C -- no per-frame Python loop),
and audio, all from a mature codec, at the cost of video no longer being
frame-independently seekable or carrying FMFF's own hybrid lossless tiles
the way images (and that earlier version) did. Both AV1 and Opus are
royalty-free, so this carries none of the patent licensing baggage a
codec like H.264/HEVC would.

Unlike a plain MP4/MKV/WebM, the video isn't stored as one opaque blob.
FFmpeg is asked for **fragmented MP4** output -- the same mechanism
browsers use for MSE/DASH streaming -- which naturally splits into a
small init chunk (codec setup) followed by consecutive, keyframe-aligned
fragments a few seconds long. FMFF keeps each fragment's length and
CRC32 in its own segment table instead of re-merging them into one
blob. That gives video the same kind of error resilience images already
had: a fragment that fails its CRC32 (or that a truncated/still-
downloading file simply doesn't have yet) gets dropped, and everything
else is reconstructed into a still-valid, still-playable file -- the
video just skips those few seconds, instead of the whole file refusing
to open the way a damaged-index MKV/MP4 would.

One thing worth knowing if you hit a pink/magenta tint in dark scenes,
found and fixed by direct A/B testing against a real source file with a
fade-to-black: FFmpeg is told the color space explicitly (`bt709`, the
standard for consumer HD video) rather than leaving it unset -- an AV1
stream with no color tags of its own left it to whichever decoder
played it back to guess, and different decoders' guesses for
"unspecified" didn't agree.

That same round of testing also turned up a second, unrelated problem
with how the viewer *plays* video, not how FMFF encodes it: it used to
shell out to `ffplay` for real audio/seek controls, but on at least one
real system `ffplay`'s own SDL2 rendering produced visible blocky
corruption in dark scenes -- confirmed with a plain, never-touched-by-
FMFF source video too, and confirmed absent in every other player on
that system (Windows' own default player included), so it's a bug in
ffplay/SDL on that system, not anything FMFF encodes or decodes.
Opening a plain video file, or a `.fmff` video, both hand off to
whatever the OS has associated with `.mp4` -- the same thing double-
clicking it in Explorer/Finder does, in its own window with real audio,
seeking, and pause, rather than FMFF's own silent hardware-decoded loop
(which is now only ever a fallback for when launching the default
player fails outright, or the one exception below). An earlier version
kept a `.fmff` video inside FMFF's own window "on purpose"; in practice
that just meant no audio and no seek bar for the one case (FMFF's own
files) where a good player experience mattered most, for no real
benefit, so this now matches how a plain video file already behaved.

One thing worth knowing if a `.fmff` video plays with no sound in
Windows' own Movies & TV app specifically: the same Opus-in-fragmented-
MP4 issue described for audio below applies to a video's audio track
too (Movies & TV's Media Foundation pipeline doesn't reliably decode
Opus unless it's wrapped in Ogg, which a video container obviously
can't be). Since dropping the container isn't an option for video the
way it is for audio, the viewer instead transcodes just the audio track
to AAC (`-c:v copy`, so the video itself is a fast stream copy, not
re-encoded) before handing the file to the default player -- the stored
`.fmff` keeps its original Opus audio either way, this is a disposable
playback-only copy.

A related but separate problem showed up specifically in Windows'
Photos/Movies & TV app: even with the AAC audio fix above in place, the
seek bar stayed dead and no real duration showed. The file
`extract_media()` reconstructs is FFmpeg's own *streaming-style*
fragmented MP4 (`frag_keyframe+empty_moov` -- written segment by
segment as it's decoded, so nothing ever knew the total length up
front): its `moov` box carries no duration and there's no `sidx`
index either, which is exactly what a lightweight player reads to
populate a seek bar. FFmpeg itself can still work out the real
duration by scanning the whole file, but Windows' own player doesn't
bother -- confirmed directly by inspecting the box layout of the raw
extracted file (`moov` before `mdat`, `moof` boxes present, no
duration in the header) versus the same file after one more `-c copy`
remux (`moov` after `mdat`, no `moof`, real duration in the header).
That extra remux -- no re-encode, just a container finalize -- now
always runs as its own explicit step before handing the file to the
default player, rather than only ever happening as a side effect of
the AAC transcode succeeding; if that transcode fails for any reason,
the seek bar still works even though the file falls back to Opus
audio.

`extract_media()` plus these two remux/transcode passes used to run
right on the GUI thread, with nothing shown while they worked -- fine
for a short clip, but for a genuinely heavy video this was confirmed to
freeze the window long enough for Windows to flag it "Not Responding",
with no percentage or other feedback the whole time. All of it now runs
in a background thread instead: reading/CRC-checking each segment
reports real `extracting... N%` progress against the segment count,
and the two playback-prep passes report real `preparing playback... N%`
by parsing FFmpeg's own `-progress pipe:1` output against the video's
known frame count (confirmed by direct testing that FFmpeg reports
`frame=` progress even for a pure `-c copy` pass, since it's counting
packets, not decoding) -- the GUI thread itself is never blocked on
anything video-sized. The same fix applies to opening a `.fmff`
image (the thumbnail-to-full-resolution upscale that used to run before
the tile-decode thread even started could itself stall a large image)
and to browsing a PDF/`.txt` (see "Documents" below), which now shows
the same `decoding... N%` wording instead of a plain, unquantified
"rendering pages...".

Saving a video `.fmff` back out via Save as... / `decode`, not just
playing it live, hit the identical Opus-in-MP4 problem on the *export*
side: both used to stream-copy the audio track untouched into whatever
container the output extension asked for, which is exactly right for
`.webm`/`.mkv` (Opus is WebM's own native audio codec, well supported
wherever either format is used at all) but silently produced a
plays-with-no-sound file for `.mp4`/`.mov`/`.m4v` -- the same MP4-
family containers Windows can't reliably pull Opus out of. Those three
extensions now force the audio through an AAC transcode on export too
(the video itself stays a stream copy either way, so this doesn't add
real encode time), the same fix live playback already gets above.

Save as... on a `.fmff` video to another video container had a matching
progress-display bug: that remux ran as one blocking call with nothing
parsed from it, so the busy overlay sat at a static "0%" for the whole
export, however long it took, then jumped straight to 100% -- easy to
mistake for a frozen/broken export on a heavy file even though nothing
was actually wrong. This export path now gets the same real `-progress
pipe:1` frame-count tracking described above instead.

### Subtitles

Every text-based subtitle track (SRT, ASS/SSA, WebVTT, or already-
mov_text) present in the source -- not just the first one, each is
mapped explicitly by its own stream index -- rides along in the exact
same fragmented MP4 as the AV1 video and Opus audio, converted to MP4's
own `mov_text` timed-text format. This needed no new header fields, no
second segment table, nothing like video alpha's separate track below
-- subtitles are just
more bytes inside the one blob `extract_media()` already hands back
whole, so a decoded `.fmff` video with subtitles plays them back in any
player that reads `mov_text` (which is to say: any of them) with zero
FMFF-specific code on the reading side. An image-based subtitle track
(PGS/DVD subs, common on Blu-ray/DVD rips) is left out rather than
attempted and failed -- `mov_text` has no way to carry a picture, and
asking FFmpeg to convert one to it errors outright instead of degrading,
so that case is detected up front (see `_TEXT_SUBTITLE_CODECS`) and
simply skipped, rather than failing the whole video encode over a
subtitle track that was never going to survive the conversion anyway.

### Video alpha (real transparency)

A video source with an actual alpha channel (a `.mov` in QuickTime
Animation/`qtrle` or ProRes 4444, a PNG sequence, anything FFmpeg decodes
to `rgba`/`argb`/`yuva420p`/etc.) is detected automatically at encode
time. Neither AV1 nor MP4 has a standard alpha-channel convention of
their own -- that gap is exactly why this is one of the few things
essentially no mainstream video format handles well without a
proprietary/heavy codec (ProRes 4444) or a browser-specific hack (WebM
VP8/9's alpha side-channel, which most tools outside a browser don't
read). FMFF's approach is the most literal one that actually works
end-to-end with free tools: the alpha plane is pulled out with FFmpeg's
`alphaextract` filter and encoded as a second, independent grayscale AV1
track -- segmented and CRC32-checked exactly like the color track (its
own `--alpha-output` in the CLI, its own corruption counter) -- and
recombined at decode time.

There's no existing player that composites two separate tracks back into
transparency automatically, so this viewer plays an alpha video entirely
in-window (reads both tracks frame-by-frame, blends them against the
dark canvas), the same way it already displays an RGBA *image* -- not
through the external player a normal `.fmff` video uses, and
without live audio even if the file has an audio track (the audio is
still there in the file -- Save as... / `decode` still get it, just not
this live in-window preview). One implementation note if you're reading
the code: the alpha track is encoded with `libaom-av1`, not
`libsvtav1` -- SVT-AV1 (at least in the FFmpeg build this project
targets) produces a bitstream `libdav1d` can't decode for
`alphaextract`'s single-plane grayscale output, confirmed independent of
fragmentation; `libaom-av1` handles the same input cleanly.

**Requires FFmpeg on PATH**, built with `libsvtav1` + `libaom-av1` (video
alpha only) + `libopus` (most mainstream builds have all three). On
Windows:
```
winget install BtbN.FFmpeg.LGPL.8.1
```
An LGPL build is used deliberately, though it's not actually a licensing
requirement here: FMFF only ever shells out to `ffmpeg.exe`/`ffprobe.exe`
as separate processes, it never links against FFmpeg, so which
build/license variant you install doesn't affect FMFF's own code. Run
`python F.M.F.F.py doctor` after installing to confirm `libsvtav1`/
`libaom-av1`/`libopus` all actually made it into the build FFmpeg found
on PATH, rather than finding out the hard way mid-encode.

## Audio -- wrapped Opus via FFmpeg

Audio (MP3, WAV, FLAC, Opus, OGG, M4A, AAC, WMA) is handled exactly like
video's own audio track, just without a picture: FFmpeg encodes it to
Opus as fragmented MP4, split into a small init chunk plus a run of
independently CRC32-checked segments (see "Video -- wrapped AV1 + Opus
via FFmpeg" above -- this is the *same* mechanism, not a new one, so a
corrupt/missing segment degrades the same way, dropping only whatever
seconds it covered instead of failing the whole file). It's a new
content type in the header only so a reader can tell "this segmented
blob is a picture" from "this segmented blob is sound" -- not a fourth
storage scheme of its own. Opus is royalty-free and was already a hard
FFmpeg dependency here regardless, since video's audio track uses it
too.

If the source carries embedded cover art (ID3 `APIC`, a FLAC picture
block, ...), FFmpeg exposes it as an attached picture stream, which gets
pulled out and stored as the file's thumbnail through the exact same
path images and video posters use -- opening an audio `.fmff` shows the
album art for free, with the same plain-gray fallback video uses when
there isn't one.

`encode`/`decode`/`info` all handle audio the same way they handle
video, just with `--audio-bitrate` instead of `--crf`/`--speed`. Left
unset, the target Opus bitrate is auto-picked from the *source's* own
bitrate (see `_default_audio_bitrate`) rather than a flat 128k: Opus is
more efficient than most older lossy codecs at a given bitrate, so
matching an already-lossy source's own number exactly tends to come out
*bigger*, not smaller, once FMFF's own segment-table/index overhead is
added on top of an already source-sized stream -- measured on a real
128kbps CBR MP3, a matched 128k Opus target came out 6% bigger than the
source, while 75% of the source's own bitrate (96k here, and capped at
128k either way so a well-encoded 320k MP3 doesn't get pushed *above*
the normal default) came out 20% smaller. A high source bitrate --
lossless WAV/FLAC, which reports its raw PCM bitrate, typically
four-figure kbps -- isn't "already lossy at some low target" the same
way, so that case just gets the plain 128k default. `decode` transcodes
to whatever container the output extension needs: `-c:a copy` (no
re-encode) for `.opus`/`.ogg`, Opus's own native container, where it's
both legal *and* actually going to get decoded; `.mp4`/`.m4a` could
hold the exact same Opus bytes just as legally, but -- same Windows
Media Foundation limitation as a video's audio track, see "Video"
above -- they don't reliably decode there, so those two are forced
through a real AAC transcode instead of tried as a copy first;
anything else (`.mp3`/`.wav`/`.flac`, ...) can't hold Opus at all, so
it always gets a real transcode regardless. The viewer opens either a
plain audio file or an audio `.fmff` by handing it straight to the
system's default player (real seek bar, volume, etc.) -- the same thing
a `.fmff` video now does too (see "Video" above), just with no picture
to blit in-window while it's playing.

What actually gets handed off for an audio `.fmff` isn't the raw
extracted stream, though: `extract_media()` alone produces a
*fragmented* MP4 with an Opus-only track -- exactly what's stored on
disk, and exactly what FFmpeg-based tools expect, but confirmed by
direct testing to fail in Windows' own Movies & TV / Media Player app
("this item was encoded in a format that's not supported", `0xc00d5212`)
-- Media Foundation's own pipeline doesn't reliably handle Opus inside a
*fragmented* MP4 the way FFmpeg's demuxer does. Ogg is Opus's own
native container (the one Windows' built-in Opus support actually
targets), so the viewer remuxes into that (`-c:a copy`, no re-encode,
just a container swap) before handing it off -- that's the file that
actually gets played. `decode`'s own `.opus`/`.ogg` output already used
a proper (non-fragmented) Ogg container, so this was a viewer-playback-only
issue, not a problem with `.fmff` files or `decode` output themselves.

## Documents -- the original file, not rendered pages

PDF and `.txt` are the one case in this file that isn't media at all --
text and layout, not pixels/frames/samples. An earlier version of this
rasterized every page to a picture and stored the result as an ordinary
multi-frame `.fmff`, the same container an animated GIF uses. That was a
real, deliberate trade-off at the time, but it was the wrong one: no
selectable/searchable text, no hyperlinks, no forms/formulas, nothing a
real PDF/text viewer gives you beyond what you can see -- and, even after
tuning (disabling anti-aliasing, using the no-fixed-grid path instead of
the still-image tile grid), a rendered page still cost more bits than the
already-compressed text/vector data it came from, so the result was
routinely *bigger* than the source despite throwing all of that away. A
worse result that's also less useful had nothing left to recommend it.

FMFF now stores the document as itself: the original bytes, raced against
a general-purpose compressor (the same lossless race JPEG passthrough
uses for its own coefficients -- see "JPEG sources" above) and kept
however comes out smaller. A PDF's internal streams are usually already
Flate-compressed, so the original bytes often win outright there; a
`.txt` source (not compressed to begin with) typically shrinks a lot.
Either way this can never end up bigger than the source by more than this
container's own small fixed overhead (header + index + one small preview
thumbnail) -- measured on a real 3-page PDF, the `.fmff` came out **28%
smaller** than the source; a real `.txt` file (this README, in fact) came
out **61% smaller**. `decode` (or the viewer's Save as...) recovers the
exact original bytes, byte-for-byte -- search, hyperlinks, and formulas
all keep working in a real PDF/text viewer, because that's what's
actually stored.

Rendering doesn't disappear, it just moves from being the storage format
to being a *preview*: encoding still renders the first page (only the
first -- cheap even for a huge PDF) for an instant-preview thumbnail, and
the viewer re-renders every page on demand straight from the recovered
original bytes when actually browsing a document `.fmff`, the exact same
rendering path a plain, not-yet-converted PDF/`.txt` already uses. Anti-
aliasing is still off for that rendering, for the same reason as before
(smoother edges cost real bits a preview doesn't need), but it no longer
affects the stored file's size at all -- only how the on-demand preview
looks.

Scoped deliberately to formats a lightweight, no-external-application
dependency can rasterize *for the preview*: `pymupdf` (`pip install
pymupdf`, a real library, no separate program) for PDF, nothing extra at
all for `.txt`. Office formats (`.docx`/`.xlsx`/...) would need an
external renderer (LibreOffice, run headless) -- a much heavier
dependency than anything else this project asks for, left out of this
pass on purpose. Note that rendering is now optional in a way it never
used to be: storing and recovering a document's bytes needs no rendering
dependency at all -- `pymupdf` only matters for the PDF preview thumbnail
and for browsing pages in the viewer, never for the data itself.

## Metadata -- tags and EXIF

Until now, none of this survived converting to `.fmff`: an audio/video
source's own tags (artist, album, title, date, genre, ...) and a photo's
EXIF (camera make/model, capture settings, orientation, and -- notably --
GPS coordinates) were silently dropped, because every encode path here
works from decoded pixels/samples, not from the source container's other
metadata. Both are now carried through as one small optional JSON blob
appended after everything else in the file (`metadata_offset`/
`metadata_length` in the header -- zero/zero when there's nothing to
store, true for every file from before this).

- **Audio/video tags**: whatever `ffprobe` reports as the container's own
  format-level tags, stored verbatim as JSON key/value pairs -- no
  curated allowlist to fall out of date. `decode`/Save as... restore them
  with `-metadata key=value` on the FFmpeg remux that runs for
  essentially every output format now. `.mp4`/`.mov`/`.m4v`/`.m4a` no
  longer get a tags-free fast path that skips FFmpeg entirely the way
  they used to: those extensions need the remux regardless, to force
  audio to AAC instead of leaving it as Opus (see "Video"/"Audio"
  above). The CLI's `decode` keeps a narrower version of that old fast
  path only as a last-resort fallback for when FFmpeg genuinely isn't
  installed -- it prints a clear warning that tags won't be restored
  and the Opus audio may not play everywhere, instead of silently
  degrading.
- **Image EXIF**: the source's raw EXIF block, byte-for-byte, not
  re-derived field by field -- the same passthrough philosophy JPEG's own
  DCT-coefficient path already uses elsewhere in this file. GPS
  coordinates, camera make/model, and orientation are just fields inside
  that same block, so all three are covered by one mechanism rather than
  needing separate handling. Restored via Pillow's own `exif=` save
  parameter wherever the output format supports it (JPEG/TIFF/PNG/WebP);
  passing it to a format that doesn't (GIF/BMP/ICO) is a harmless no-op,
  not an error.

Both are best-effort: a source with nothing to carry (most `.txt`/PDF
documents, a WAV with no tags, a screenshot with no EXIF) simply gets an
empty blob, and every encode path already tested throughout this file
keeps working exactly as it did before, tags or not.

## Batch conversion and drag-and-drop

The viewer window has a **Batch...** button, and (when `tkinterdnd2` is
installed) accepts drag-and-drop directly: dropping a single file previews
it like Open does, but dropping a folder or several items at once opens a
batch queue instead. The batch window recurses into folders for supported
images/videos, converts each to `.fmff` next to its source file, and shows
a live per-file and running-total size comparison (`119.0 KB -> 41.4 KB,
65% smaller`) as it goes -- the same `original -> fmff (N% smaller)`
summary also shows up after a single-file Save as..., and in the CLI's
`encode` output.

The **Start** button toggles to **Stop** once a batch is running: it
signals the worker to stop, which finishes checking (not restarting) the
current file, marks every not-yet-started file `cancelled` instead of
leaving them stuck at `queued`, and -- for a video that's mid-encode --
actually kills the in-progress `ffmpeg` subprocess rather than waiting for
it to finish, so nothing partial gets written. Closing the batch window
does the same thing automatically, so leaving the window open isn't the
only way to stop a running batch.

## Glue -- concatenating .fmff clips into one

The viewer window also has a **Glue...** button: pick any number of
`.fmff` video files, or any number of `.fmff` audio files (not a mix
of both), either one at a time or by picking a whole folder (every
`.fmff` inside it gets added, in name order), arrange them in the
order they should play, and concatenate them into one new `.fmff`.

This isn't a raw byte-level splice of the stored AV1/Opus segments.
Each input is unpacked back to its own embedded fragmented MP4 (the
same `extract_media()` "Video"/"Audio" above already use to play or
export a file), FFmpeg's concat demuxer joins those into one file, and
*that* gets fed straight back through `encode_video`/`encode_audio`
exactly like any other source file would be. So the result is a
freshly re-encoded clip, not a stream-copied splice carrying over
whatever timestamp discontinuities a raw concat join tends to leave at
each cut point.

Every input has to already match on the one property that actually
matters for concatenation to make sense: the same resolution for
video, or the same sample rate and channel count for audio. FFmpeg's
own concat demuxer doesn't refuse a mismatch itself -- it just hands
back a file that glitches or fails to seek properly past the join --
so Glue checks up front and raises a clear error (listing the
mismatched values) instead of producing that silently.

Viewer only for now -- there's no CLI subcommand for this yet.

## Benchmarks -- FMFF vs PNG/WebP, with real numbers

Numbers, not adjectives. [`benchmarks/bench_images.py`](benchmarks/bench_images.py)
generates six small, seeded synthetic test images (two photo-like ones --
one smooth/soft, one busier/textured -- a UI/screenshot mockup, a
low-color pixel-art sprite, black-and-white line art, and pure random
noise as a deliberately-included worst case), encodes each one with
FMFF's own image codec, Pillow's PNG encoder, and Pillow's WebP encoder
(both lossless and quality-matched lossy), and reports the real, measured
file sizes and encode times side by side -- no image dataset is
downloaded or committed to this repo; the script generates its own test
images from a fixed seed, so anyone can reproduce (or challenge) these
exact numbers on their own machine, and `--images-dir` swaps in real
images instead of the synthetic stand-ins if you'd rather benchmark
actual content. (One bug in that reproducibility claim already found and
fixed: the per-category seed used Python's built-in `hash()` on a string,
which is randomized per process since Python 3.3 -- so two "identical"
runs quietly generated slightly different images. Swapped for `zlib.crc32`,
which isn't process-randomized; confirmed by running the whole benchmark
twice in a row and diffing the output byte-for-byte.)

Run it yourself: `python benchmarks/bench_images.py`. Full results (with
methodology notes and per-image pixel-fidelity numbers) land in
`benchmarks/results.md`; raw data in `benchmarks/results.csv`. One run's
headline numbers, FMFF quality=80 vs WebP quality=80 (matched, so the
lossy-vs-lossy comparison is fair):

| content type | FMFF vs PNG | FMFF vs WebP q80 |
|---|---|---|
| smooth photo-like | 92% smaller | 147% bigger |
| detailed/textured photo-like | 86% smaller | 95% bigger |
| UI/screenshot mockup | 113% bigger | 115% bigger |
| pixel art / low-color sprite | 139% bigger | 16% smaller |
| line art | 10% bigger | 54% smaller |
| random noise (worst case) | 78% smaller | 2% smaller |

**The honest reading of this**: FMFF's own lossy tile codec still isn't
competitive with WebP's on raw compression ratio for photographic
content, but it's a lot closer than it started out -- the very first
version of this benchmark showed 301-431% bigger than WebP on the two
photo-like cases; it's 95-147% now, after fixing five concrete gaps found
by reading the codec's own code (see "Hybrid lossless/palette/lossy
tiles" above for what each one was): no chroma subsampling, no zigzag
coefficient ordering or DC delta-coding, a fixed 2-bytes-per-coefficient
encoding with no EOB-style cut of a block's trailing zero run, a
quantization table swap verified on real before/after numbers rather than
assumed (see "The base quantization table" above -- a small, mixed
result: better on smooth photo/noise content, very slightly worse on
busy/textured photo content, disclosed rather than cherry-picked), and an
optional `zstandard` candidate added to the compression race (see "A
third entropy-coding candidate" above). Even the random-noise worst case,
which has no exploitable structure for any of this to find, went from
132% bigger than WebP at the very start to 2% *smaller*. What's left of
the remaining photo-content gap is believed to be WebP's more modern
predictive lossy design against FMFF's still-fundamentally-JPEG-style
per-tile DCT, entropy-coded with a general-purpose compressor (zlib/bz2/
zstd) over a byte-packed stream rather than true coefficient-aware
bit-level Huffman/arithmetic coding -- consistent with "encoding is pure
Python per-tile" in "Status / limitations" below (closing that last gap
for real would mean a bit-level Huffman/arithmetic coder, which this
project has so far avoided as a large jump in implementation risk -- a
subtly broken bit-packer corrupts images silently -- for a shrinking
remaining return, and spatial prediction between blocks the way WebP
itself does would mean giving up the independent, streamable/resilient
tile design this container is actually built around -- see "Chunked/
streamable" and "Error resilience" above). FMFF beats PNG on most content
(PNG has no lossy mode to fall back to when one would help) and wins
outright on line art and pixel art; it loses on the flat/UI screenshot
case, where WebP's whole-image lossless prediction beats FMFF's smaller,
per-tile palette encoding. FMFF's actual reason to exist
isn't winning a compression-ratio contest against a mature,
heavily-optimized format -- it's versioning, true-color alpha, and
per-tile error resilience, none of which PNG or WebP has at all (see the
top of this README and "Versions" below). Take the compression numbers
at face value rather than a pitch that hides them; video/audio aren't
included in this benchmark since they're just FFmpeg's own AV1/Opus
encoders wrapped in this container (see "Video" and "Audio" above) --
comparing those would mostly measure this container's own small fixed
overhead, not a codec FMFF wrote.

## Status / limitations

This is a personal/hobby project, not a production tool:

- Images: 8-bit RGB/RGBA only. Header fields for bit depth / color space
  are reserved for a future HDR extension but not implemented.
- Images: versions (see "Versions" above) are still-image-only and share
  one quality setting across every version in a file -- `add-version`
  (CLI, or the viewer's own **Add version...** button) always uses the
  file's own already-stored quality, never a per-version override.
- Images: selection masks (see "Selection masks" above) are one mask per
  version, CLI-only, same still-image-only scope as versions.
- Images: encoding is pure Python per-tile -- spread across CPU cores for
  large images (see "Multi-core encoding" above), but still Python-level
  work per tile, not a compiled codec, so it won't match a native
  encoder's speed. Compression ratio, not just speed, also trails a
  mature format like WebP on ordinary photographic content -- see
  "Benchmarks" above for real numbers instead of a vague "it's smaller"
  claim.
- Video: the hybrid lossless-tile idea is specific to the image codec
  and doesn't apply to the AV1 stream inside a video `.fmff` (segment-
  level CRC32/corruption resilience does apply, though -- see "Video --
  wrapped AV1 + Opus via FFmpeg" above). Alpha is supported (see "Video
  alpha" above) but only detected from a source that already has one;
  nothing converts an opaque video into a transparent one.
- Video playback: both a plain video file and a `.fmff` video open the
  system's default player (real audio+video, seeking, in its own
  window) -- FMFF's own silent, no-seek-bar, hardware-decoded in-window
  loop is now only a fallback for when launching the default player
  fails outright. A video with alpha is the one exception: it always
  uses that in-window path (with real transparency, but silent even if
  the source has audio), since no mainstream player composites two
  separate tracks into transparency on the fly -- see "Video alpha"
  above.
- Audio: cover art is carried over as the thumbnail, and source tags
  (artist, album, track number, ...) are preserved and restored on
  decode too (see "Audio -- wrapped Opus via FFmpeg" above). Re-encoding
  to Opus is
  also always a lossy transcode, same as re-encoding any lossy source
  through another lossy codec would be, even from an already-lossy
  source like MP3 (there's no coefficient-passthrough trick for audio
  the way JPEG images get one).
- Subtitles: only text-based tracks are carried over (see "Subtitles"
  above) -- an image-based one (PGS/DVD subs) is silently left out of
  the encode rather than attempted and failed. Every text-based subtitle
  track survives when the source has several, not just one (every stream
  is mapped explicitly by its own index rather than relying on FFmpeg's
  "best of each type" auto-selection -- the same explicit-mapping fix
  that also means every audio track survives, not just one), and styling
  beyond plain text is only as faithful as `mov_text` itself supports
  (fairly little -- MP4's timed-text format is much simpler than
  ASS/SSA's own).
- Documents: PDF and `.txt` only -- Office formats (`.docx`/`.xlsx`/...)
  would need an external renderer (LibreOffice) not implemented here. The
  document itself is stored byte-for-byte (see "Documents" above), so
  none of the old rasterization trade-offs apply any more -- text stays
  selectable/searchable, hyperlinks and forms keep working, and the
  `.fmff` is typically *smaller* than the source, not bigger. The only
  thing that's still a rendered picture is the small first-page preview
  thumbnail and the viewer's on-demand page browsing, neither of which
  affects what's actually stored.

**Don't use `.fmff` as the only copy of anything you care about** -- keep
your originals.

## Windows .exe -- no Python install needed

The whole thing (encoder, decoder, CLI, and GUI viewer) is also
distributed as a single standalone `F.M.F.F.exe` (built with PyInstaller,
`pyinstaller F.M.F.F.spec`) attached to each [GitHub
Release](https://github.com/vladislav-n228/FMFF-Fast-Media-File-Format/releases)
-- not committed to the repo itself (a 100+MB binary has no business
living in git history; see the `.gitignore`). FFmpeg still has to be on
PATH separately either way -- it's a real external program FMFF only
ever shells out to, packaging this into one `.exe` doesn't bundle it.

A `multiprocessing.Pool`-based encode (see "Multi-core encoding" above)
needs one specific piece of care to work correctly once frozen:
`multiprocessing.freeze_support()` has to run as the very first thing in
`if __name__ == "__main__":`, or every worker process would re-run the
*whole program* from scratch on Windows instead of just bootstrapping as
a worker -- for a large-enough encode job, that means each worker
spawning its own full worker pool, recursively, without limit. This is
the exact failure mode a from-source multiprocessing bug (unrelated to
this file, a one-off test script missing its own `__main__` guard)
already caused once during development, so it's called out here as a
"don't remove this line" rather than a hypothetical.

**Worker processes can't outlive a crash any more, either.** Tile-pool
workers used to be cleaned up only via `atexit` (`_shutdown_tile_pool`),
which runs on a normal exit but not when the process is killed outright
-- Task Manager's "End task", a hard crash inside an image/video C
extension, anything that skips Python's own shutdown sequence. Each
worker just sits idle waiting on the pool's task queue forever in that
case: no CPU usage, so it's easy to miss, but every one of them is
memory that never comes back, and Task Manager can't group an orphan
under the app it no longer has any relationship to -- which is exactly
what showed up as several ungrouped, unexplained `F.M.F.F.exe` entries
after a crash, alongside the correctly-grouped, still-running instance.
The app now assigns itself to a Windows Job Object with
`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` (`_setup_worker_job_object`) before
any pool can exist; Windows automatically makes every process launched
afterwards -- each pool worker included -- a member of that same job,
and tears down every member the moment the job's one handle is closed,
which Windows does on its own as soon as this process ends, no matter
how. Confirmed by force-killing the main process mid-encode with 8
active workers: before this, all 8 kept running indefinitely; after,
`taskkill /F` on the main process took every worker down with it.
Windows-only and best-effort -- a host that already placed this process
in a job without nested-job support (older than Windows 8, some
sandboxes) fails the assignment silently rather than crashing the app,
since a normal exit was already covered by `atexit` regardless.

Double-clicking the `.exe` with no arguments opens the GUI viewer, same
as `view` with no file; the console window that a plain console-
subsystem build would otherwise show for that hides itself immediately
on startup for `view`/`open-external` specifically (`encode`/`decode`/
`info`/`register-filetype`/... still print normally when run from an
actual terminal).

Two extra CLI subcommands exist only for this packaged/double-click
scenario, not as core format features:

- **`open-external`** -- converts a `.fmff` to whatever ordinary format
  its content already reduces to (video/audio/JPEG passthrough are
  already a standard codec stream under FMFF's own header, so that's
  just extracted; a still/animated image has no standalone equivalent,
  so it's decoded through FMFF's own codec and re-saved as a plain PNG/
  GIF/WebP) and hands the result to the system's default player --
  headless, no FMFF GUI window at all. `--choose` shows Windows' native
  "Open with" picker instead of silently using the remembered default.
  The GUI viewer has the same option too, without going headless: an
  **Open externally** toolbar button converts whatever's currently
  loaded the same way and hands it to the OS default app, while the
  viewer window itself stays open. Useful for any content type, but
  especially a PDF/`.txt`, since the viewer's own page preview has no
  text layer, search, or hyperlinks (see "Documents" above) the way a
  real document viewer does.
- **`register-filetype [--mode external|viewer]`** -- registers `.fmff`
  as a real Windows file type for the current user only (`HKEY_CURRENT_
  USER\Software\Classes`, no admin rights, nothing outside this one
  account touched), so double-clicking a `.fmff` file in Explorer works
  like any other media file instead of prompting "how do you want to
  open this" every time with no memory of the answer. `--mode external`
  (the default) wires a double-click to `open-external`; `--mode viewer`
  opens FMFF's own GUI instead. `unregister-filetype` undoes it.

## Requirements

Run `python F.M.F.F.py doctor` any time to check all of the below at
once -- what's found, what's missing, and the exact command to fix it
(including checking that an installed FFmpeg actually has `libsvtav1`/
`libaom-av1`/`libopus` compiled in, not just that `ffmpeg` itself is on
PATH -- a real gap plain "FFmpeg not found" error messages used to miss
entirely). Faster than reading a traceback to guess which of several
optional libraries an error actually came from.

- Python 3.9+
- `numpy`, `Pillow`
- `opencv-python` (viewer only: opening/playing ordinary image and video
  files, and the silent-preview video fallback -- video playback itself
  prefers the system's default player, see "Status / limitations" above)
- `ffmpeg` / `ffprobe` on PATH (video and audio only: encoding, decoding,
  and remuxing; `libaom-av1` specifically is needed too for a video-with-
  alpha source -- see "Video alpha" above)
- `jpeglib` (optional, JPEG sources only: enables lossless coefficient
  passthrough; without it, JPEG encoding falls back to the normal tile
  codec at a lowered quality)
- `pymupdf` (optional, PDF sources only, preview rendering only:
  `pip install pymupdf` -- without it, encoding a PDF raises a clear
  error (no first-page thumbnail to make), but this is only ever about
  the preview -- storing/recovering the original PDF bytes themselves
  needs no rendering dependency at all; `.txt` needs nothing beyond
  Pillow itself)
- `tkinterdnd2` (optional, viewer only: enables dragging files/folders
  onto the window; without it, use the Open/Batch... buttons instead)
- `zstandard` (optional, images only: `pip install zstandard` -- adds a
  3rd candidate to the tile compression race alongside zlib/bz2, usually
  both smaller and faster than either; without it, tiles are still
  correct, just raced between zlib/bz2 alone. A file that does have a
  zstd-compressed tile needs this installed to decode *that* tile --
  `doctor` flags this specifically if it comes up)

## Usage

```
python F.M.F.F.py encode input.png output.fmff [--quality 80] [--tile-size 64]
python F.M.F.F.py encode input.png output.fmff --version-name original --version-note "as shot"  # label version 0, see "Versions"
python F.M.F.F.py add-version output.fmff retouched.png --name retouched --note "..."  # append another version, in place
python F.M.F.F.py decode output.fmff result.png --version 0        # a specific version by index or --name (default: most recent)
python F.M.F.F.py add-mask output.fmff mask.png [--version 0]      # attach a selection mask to a version, see "Selection masks"
python F.M.F.F.py decode output.fmff mask.png --mask [--version 0] # extract a version's mask instead of the image
python F.M.F.F.py encode input.gif output.fmff        # animated GIF/WebP/APNG source -> multi-frame .fmff, detected automatically
                                                       # (--tile-size is still-image only -- an animated source has no fixed tile grid, see "Each changed frame stores one rectangle" above)
python F.M.F.F.py encode input.mp4 output.fmff [--crf 30] [--speed 8] [--fps 30]
python F.M.F.F.py decode output.fmff result.png
python F.M.F.F.py decode output.fmff result.gif        # a multi-frame .fmff decodes to a full animation for .gif/.webp/.png, frame 0 alone otherwise
python F.M.F.F.py decode output.fmff result.mp4        # remuxes via FFmpeg, forcing AAC audio for .mp4/.mov/.m4v (Windows can't reliably decode Opus there); other video extensions keep Opus
python F.M.F.F.py decode output.fmff result.mp4 --alpha-output alpha.mp4   # also pulls out the alpha track, if the source had one
python F.M.F.F.py encode input.mp3 output.fmff [--audio-bitrate 96k]  # unset: auto-picked from the source's own bitrate, see "Audio" above
python F.M.F.F.py decode output.fmff result.opus      # -c:a copy for Opus's own native containers (.opus/.ogg); .mp4/.m4a get an AAC transcode instead (Windows can't reliably decode Opus there), anything else (.mp3/.wav/.flac/...) gets a real transcode
python F.M.F.F.py encode input.pdf output.fmff        # stores the original PDF bytes themselves (recompressed if smaller), see "Documents" above
python F.M.F.F.py encode input.txt output.fmff        # same passthrough approach for a text file
python F.M.F.F.py decode output.fmff result.pdf       # recovers the exact original bytes, byte-for-byte -- not a rendered page
python F.M.F.F.py info output.fmff
python F.M.F.F.py view [file]        # image / video / audio / document / .fmff viewer, Open/Screenshot/Save in the window
python F.M.F.F.py output.fmff        # bare path also opens the viewer directly
python F.M.F.F.py open-external output.fmff [--choose]   # headless: convert + hand off to an external player, no FMFF window -- see "Windows .exe" above
python F.M.F.F.py register-filetype [--mode external|viewer]   # associate .fmff with double-click in Explorer (current user only)
python F.M.F.F.py unregister-filetype
python F.M.F.F.py doctor             # check FFmpeg (+ its encoders) and every optional library, with exact fixes for anything missing
```

The same commands work with `F.M.F.F.exe` in place of `python F.M.F.F.py`
when running the packaged build instead of from source (see "Windows
.exe" above).

## License

MIT -- see [LICENSE](LICENSE).
