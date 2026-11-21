# PSD Prep

Open a PSD, then **remove layers**, **crop**, and **convert CMYK → sRGB**. Save the result as a
**layered PSD** and/or a **flattened PNG/TIFF with alpha**. It's built for Rocky Linux and works
without internet access.

It also opens **Illustrator (.ai) and PDF files**: they're rendered to pixels with a
transparent background, and each artboard can be exported as its own PNG/TIFF with alpha.

## Use

```
PSDPrep/PSDPrep                 # GUI (or double-click)
PSDPrep/PSDPrep file.psd        # GUI with a file open (drag & drop works too)
```

* **Opening files**: *Open PSD…*, drag & drop, or **paste a path** into the box in the toolbar
  and press Enter. Quotes, `file://` links, `~` and `$VARIABLES` are all fine. **Ctrl+V**
  anywhere in the window also opens a copied path, or a file you copied in the file manager.
* **Layers panel**: untick a layer to remove it. Removing a group removes everything in it.
  The eye column toggles visibility. **Ctrl/Shift-click** selects several layers: clicking
  the eye (or tick) of one of them then sets all of them. **Show selected** / **Hide selected**
  and the right-click menu (Show / Hide / Remove / Keep) do the same.
  **Show all** / **Hide all** turn every eye on or off.
  **Reset to original** puts the layers back exactly as the file was saved: the same layers
  visible and hidden, nothing removed. **Remove hidden** unticks every hidden layer, and
  **Keep all** ticks everything again.
* **Crop**: drag on the preview to draw the box, then drag its edges or move it. You can also type
  X/Y/W/H or pick an aspect (16:9, 1:1, 4:5, 9:16…). By default pixels outside the crop are
  deleted; untick *Delete cropped pixels* to keep them, like Photoshop does.
* **Colour → Convert to**:
  * **sRGB** (default).
  * **Rec.709 (gamma 2.4)**: BT.1886, what a Rec.709 monitor and Flame's "Rec.709 video" use.
  * **Rec.709 (camera OETF)**: Nuke's `rec709`.
  * **No conversion**: the default for anything that isn't CMYK. CMYK files open set to
    your last choice (sRGB the first time).

  CMYK uses the file's embedded profile. If there isn't one, it uses the profile bundled with
  the app (searched before any system profiles, so every workstation matches); the Original panel
  shows which one, with Photoshop's defaults (Relative Colorimetric + black point compensation). Untagged
  RGB is treated as sRGB, so going to Rec.709 is exact maths (same primaries, only the curve
  changes). The outputs get a matching ICC profile embedded.
* **Proving the conversion**:
  * **Hover** the preview: the status bar shows the pixel's source values (e.g.
    `CMYK 100 0 0 0 %`) and the converted ones (`sRGB 0 159 227`), plus their ΔE. You can
    compare with Photoshop's Info panel.
  * **Check conversion** measures every visible pixel against what the source ICC profile says
    the colour is (CIE Lab), as **ΔE2000**. ΔE < 1 is invisible; < 2 is only visible side by side.
    Colours the target can't show (pure CMYK cyan is outside sRGB and Rec.709) are reported
    separately: they're clipped to the nearest colour, as Photoshop does.
  * **Highlight pixels with ΔE > 2** hatches those areas in magenta in the preview.
  * A broken conversion (for example inverted CMYK) shows up as ΔE in the tens to ~100.
* **Original panel** shows the source's colour mode (RGB / CMYK / Gray / Lab), bit depth,
  embedded colour profile and its colour space.
* **The original is protected.** Every write is checked against the source file (resolved
  paths and hard links included) and refused if it matches, from the GUI and the command line.
* **Layered PSD → Nuke-friendly** (on by default). Nuke's *Read > Breakout Layers* rebuilds
  the image very simply. It skips groups and ignores the eye (hidden layers), clipping, fill
  opacity and mask defaults, and it stops at the first layer with a non-ASCII name. With this
  on, the saved PSD is baked so Nuke rebuilds exactly what you see:
  * hidden layers are left out;
  * groups are dissolved. A group with its own opacity, blend mode or a soft (feathered) mask
    becomes one merged layer. So does a clipping stack whose base isn't Normal at 100%, or that
    is clipped to a group. That's the only way Nuke's per-layer merge can match;
  * masks, fill and clipping are baked into each layer's alpha;
  * names are made ASCII and unique (`Cyan circé` → `Cyan circe`);
  * layers are trimmed to the canvas.

  The image is the same; only the layer structure is simplified. A test runs Nuke's breakout
  logic on the result and checks that it matches. Untick it to keep the Photoshop structure
  (groups, masks, hidden layers).
* **Save to**, with the colour space always at the end of the name:
  * **Next to the original, auto-named** (default): writes into the source folder as
    `<name>_<suffix>_v1_sRGB.png`, then `_v2`… The suffix defaults to `cleanup` and you can edit
    it. The version is one more than the highest found in that folder, so nothing is ever
    overwritten.
  * **Other folder and name**: pick a folder and type a name, e.g. folder `/jobs/deliveries`,
    name `Client Logo` → `Client_Logo_v1_Rec709.png`. It's still versioned and still never
    overwrites.
  * **Ask where to save each time**: file dialogs, with the tagged name pre-filled.

  The tag is the colour space you chose (`sRGB`, `Rec709`, `Rec709-OETF`). With *No conversion*
  it's the file's own space (`CMYK`, `AdobeRGB`, `RGB`…). The panel shows the next name.
* **Artboards** (Photoshop's "pages") are listed when the file has them. Tick the ones to export,
  drag to reorder, or use *Reading order* (rows top to bottom, left to right) or *Layers order*.
  *Export artboards* writes each one as its own PSD and/or flattened file, numbered in that
  order: `<name>_cleanup_v1_01_<Artboard>_sRGB.psd`, `_02_…`. Each one is cropped to its artboard,
  keeps only its own layers, and gets the artboard background colour. In the PSD that colour is
  a bottom layer; tick *Transparent background* to skip it. Clicking an artboard outlines it in
  the preview.
  Flattened files are always sRGB, have straight (unpremultiplied) alpha and an embedded sRGB
  profile, at 8 or 16 bits.

### Illustrator and PDF files

* Open or drop a `.ai` or `.pdf`. A small window asks for the size, since vector art has none:
  * **Resolution (DPI)**, e.g. 300 for print quality at the artwork's real size, or
  * **Longest side (px)**, e.g. 4000, for an exact pixel size (usually best for comp/screen use).

  It shows the resulting pixel size before rendering. **Change resolution…** in the toolbar
  renders it again.
* Each Illustrator artboard (PDF page) shows up in the **Artboards** list. Export them
  numbered as PNG/TIFF, `logo_cleanup_v1_01_Artboard_1.png`, … The background is
  transparent: alpha is 0 where there's no artwork, with soft anti-aliased edges and real
  transparency where the art is see-through.
* The preview, crop and auto-naming work as for PSDs. PSD saving is turned off for these files.
* Limits:
  * `.ai` files must have been saved with **Create PDF Compatible File** (Illustrator's default).
    Files without it, and very old pre-CS files, show a message instead.
  * Illustrator layers can't be read, so each artboard is one flat image.
  * Artboard names aren't stored in the PDF part, so they're numbered *Artboard 1, 2…* in
    Illustrator's order.
  * Colour is converted to RGB by the PDF engine (PDFium, the one in Chrome). Artwork with an
    embedded profile converts accurately; plain untagged CMYK uses a simple conversion.

### "Image from" (flattened output)

* **Auto** (default): if no layers were removed or hidden, the tool uses Photoshop's own saved
  composite, which includes layer styles and adjustment layers. Otherwise it rebuilds the image
  from the layers.
* When the image is rebuilt from the layers, it supports normal, multiply, screen, overlay,
  soft/hard light, darken/lighten, difference, exclusion, add, linear burn, subtract, divide,
  dodge and burn. It also handles opacity, fill opacity, layer masks (including combined
  vector + pixel masks, as pixels), clipping masks (clipped layers take the base's opacity and
  blend mode, and can clip to groups), groups, and solid-colour fill layers. **Not rendered:**
  layer styles (FX), adjustment layers, shape layers drawn only by a vector path,
  gradient/pattern fills, and the HSL modes (hue/saturation/color/luminosity). The layer list
  and the log flag these.

### Command line (same engine)

```
PSDPrep in.psd --list
PSDPrep in.psd --remove "Retouch*" --remove-hidden --crop 0,0,1920,1080 \
        --out-psd out.psd --out-flat out.png --depth 16
PSDPrep in.psd --profile /path/CoatedFOGRA39.icc --out-flat out.tif
PSDPrep in.psd --auto both                     # -> in_cleanup_v1_sRGB.psd + .png next to in.psd
PSDPrep in.psd --to rec709 --auto flat --folder /jobs/out --name "Client Logo"
                                               # -> /jobs/out/Client_Logo_v1_Rec709.png
PSDPrep in.psd --check --to rec709             # print the ΔE2000 conversion report
PSDPrep in.psd --nuke --auto psd               # layered PSD baked for Nuke's Breakout Layers
PSDPrep in.psd --auto flat --artboards         # -> in_cleanup_v1_01_<name>_sRGB.png, _02_…
PSDPrep logo.ai --long-side 4000 --auto flat --artboards   # Illustrator: one PNG per artboard
PSDPrep logo.ai --dpi 300 --out-flat logo.tif            # all artboards laid out in one image
```

## Limits (v0.8)

* Supports 8 and 16-bit Grayscale/RGB/CMYK/Lab PSD and PSB. 32-bit, indexed, duotone and
  bitmap files aren't supported yet.
* In the saved PSD, text layers, smart objects and solid-colour fill layers are kept as pixels
  (they lose editability). Adjustment layers, layer styles, vector masks and vector-only shape
  layers are dropped. Groups, pixel masks, opacity, fill opacity, blend modes, visibility and
  clipping are kept.
* 16-bit layers are written in the standard layer section rather than Photoshop's own `Lr16`
  block. Other apps read this fine; it hasn't been confirmed in Photoshop yet.
* Files with very large layer data (over ~4 GB) are written as PSB automatically.
* With "Ask where to save", replacing an existing file always asks first. The command line
  refuses to replace existing outputs unless you pass `--overwrite`. The original is never
  replaced.
* 16-bit CMYK is converted through a 4D LUT sampled from littleCMS (18 nodes per channel).
  Against the exact 8-bit path the difference is ≤ ~1.5/255.

## Standalone app (double-click, nothing to install)

The app is built once on a machine with internet. That gives you a folder with Python, Qt,
numpy and Pillow inside, so the Rocky machine needs no Python, no pip and no internet. It works
the same way as the slater.

**GitHub Actions (easiest).** Every push to `main` builds the app inside a Rocky 8 container and
self-tests it headless. The download lands on one fixed page:
**Releases → latest → `PSDPrep-linux-x86_64.tar.gz`**
(you can also go to *Actions → Build PSD Prep → Run workflow* to run it by hand).

Copy the file to the Rocky machine, then:

```
tar -xzf PSDPrep-linux-x86_64.tar.gz
PSDPrep/PSDPrep            # first run: from a terminal, or double-click "Run PSD Prep.sh"
```

After the first run, **PSD Prep is in the Activities menu**. The app registers itself and keeps
the entry pointing at wherever the folder lives. To also get a double-click icon on the
Desktop, run `Install shortcut.sh` (or `PSDPrep/PSDPrep install-shortcut`). Both use absolute
paths, so move the folder first and then run it again.

Builds run on Rocky 8 and 9. The window title shows the build date and commit.

**Docker** on any computer with internet: `./build_docker.sh` writes the same
`dist/PSDPrep-linux-x86_64.tar.gz`.

**CMYK profiles:** files with an embedded profile always use their own. For untagged CMYK files,
the build includes Ghostscript's generic `default_cmyk.icc`. To ship your own profiles (Coated
FOGRA39, SWOP…), drop the `.icc` files into `profiles/` before building, or later into
`PSDPrep/profiles/` next to the app. The first one found is used for untagged CMYK files.

From source (Python 3.11+ with the pinned libraries in `requirements.txt`; Rocky 8's own `python3` is 3.6 and too old): `python3.11 app_main.py`. Tests:
`python3 tests/test_roundtrip.py`. Self-test: `PSDPrep selftest`.

## Layout

```
psdprep/psdfile.py   PSD/PSB reader + writer (numpy only; Pillow speeds up RLE decode)
psdprep/ops.py       layer tree, remove, crop, ICC conversion, compositor
psdprep/imgio.py     PNG / TIFF writers (8/16-bit, alpha, ICC)
psdprep/pipeline.py  settings -> outputs
psdprep/gui.py       PySide6 UI (falls back to PySide2)
psdprep/cli.py       command line / entry point
psdprep/artboards.py artboard detection / ordering
psdprep/vector.py    Illustrator/PDF rendering (pypdfium2)
psdprep/colorspaces.py sRGB / Rec.709 targets, ICC profiles, ΔE2000
psdprep/check.py     conversion check and pixel probe
psdprep/naming.py    auto names, versions, original-file guard
psdprep/descriptor.py Photoshop descriptor parser
psdprep/selftest.py  headless self-test (run by CI on the built app)
build.py, ci/, .github/workflows/build.yml   standalone build (Rocky 8 container)
```
