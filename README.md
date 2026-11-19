# PSD Prep

Open a PSD, then **remove layers**, **crop**, and **convert CMYK → sRGB**. Save the result as a
**layered PSD** and/or a **flattened PNG/TIFF with alpha**. It's built for Rocky Linux and works
without internet access.

## Use

```
PSDPrep/PSDPrep                 # GUI (or double-click)
PSDPrep/PSDPrep file.psd        # GUI with a file open (drag & drop works too)
```

* **Layers panel**: untick a layer to remove it. Removing a group removes everything in it.
  The eye column toggles visibility. **Remove hidden** unticks every hidden layer.
* **Crop**: drag on the preview to draw the box, then drag its edges or move it. You can also type
  X/Y/W/H or pick an aspect (16:9, 1:1, 4:5, 9:16…). By default pixels outside the crop are
  deleted; untick *Delete cropped pixels* to keep them, like Photoshop does.
* **Colour**: *Convert to sRGB* uses the embedded profile first. If a CMYK file has no profile,
  it uses the profile chosen in the *CMYK profile* menu. Intent and BPC default to Photoshop's
  defaults (Relative Colorimetric + black point compensation). The output PSD gets an sRGB
  profile embedded.
* **Original panel** shows the source's colour mode (RGB / CMYK / Gray / Lab), bit depth,
  embedded colour profile and its colour space.
* **The original is protected.** Every write is checked against the source file (resolved
  paths and hard links included) and refused if it matches, from the GUI and the command line.
* **Save to → Next to the original, auto-named** (the default) writes into the source folder
  as `<name>_<suffix>_v1.psd` / `.png`, then `_v2`, `_v3`… The suffix defaults to `cleanup` and
  you can edit it. The version is one more than the highest found in that folder, so nothing is
  ever overwritten. The panel shows the next name. *Ask where to save* uses file dialogs instead.
* **Artboards** (Photoshop's "pages") are listed when the file has them. Tick the ones to export,
  drag to reorder, or use *Reading order* (rows top to bottom, left to right) or *Layers order*.
  *Export artboards* writes each one as its own PSD and/or flattened file, numbered in that
  order: `<name>_cleanup_v1_01_<Artboard>.psd`, `_02_…`. Each one is cropped to its artboard,
  keeps only its own layers, and gets the artboard background colour. In the PSD that colour is
  a bottom layer; tick *Transparent background* to skip it. Clicking an artboard outlines it in
  the preview.
  Flattened files are always sRGB, have straight (unpremultiplied) alpha and an embedded sRGB
  profile, at 8 or 16 bits.

### "Image from" (flattened output)

* **Auto** (default): if no layers were removed or hidden, the tool uses Photoshop's own saved
  composite, which includes layer styles and adjustment layers. Otherwise it rebuilds the image
  from the layers.
* When the image is rebuilt from the layers, it supports normal, multiply, screen, overlay,
  soft/hard light, darken/lighten, difference, exclusion, add, linear burn, subtract, divide,
  dodge and burn. It also handles opacity, layer masks, clipping masks and groups. **Not
  rendered:** layer styles (FX), adjustment layers, and the HSL modes (hue/saturation/color/
  luminosity). The layer list flags these.

### Command line (same engine)

```
PSDPrep in.psd --list
PSDPrep in.psd --remove "Retouch*" --remove-hidden --crop 0,0,1920,1080 \
        --out-psd out.psd --out-flat out.png --depth 16
PSDPrep in.psd --profile /path/CoatedFOGRA39.icc --out-flat out.tif
PSDPrep in.psd --auto both                     # -> in_cleanup_v1.psd + .png next to in.psd
PSDPrep in.psd --auto flat --artboards         # -> in_cleanup_v1_01_<name>.png, _02_…
```

## Limits (v0.1)

* Supports 8 and 16-bit Grayscale/RGB/CMYK/Lab PSD and PSB. 32-bit, indexed, duotone and
  bitmap files aren't supported yet.
* In the saved PSD, text layers, smart objects and fill layers are kept as pixels (they lose
  editability). Adjustment layers and layer styles are dropped. Groups, masks, opacity, blend
  modes, visibility and clipping are kept.
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
PSDPrep/PSDPrep            # or double-click "PSD Prep.desktop" (first time: right-click > Allow Launching)
```

Builds run on Rocky 8 and 9. The window title shows the build date and commit.

**Docker** on any computer with internet: `./build_docker.sh` writes the same
`dist/PSDPrep-linux-x86_64.tar.gz`.

**CMYK profiles:** files with an embedded profile always use their own. For untagged CMYK files,
the build includes Ghostscript's generic `default_cmyk.icc`. To ship your own profiles (Coated
FOGRA39, SWOP…), drop the `.icc` files into `profiles/` before building, or later into
`PSDPrep/profiles/` next to the app. You can also pick one with the **…** button.

From source (on a machine that has the libraries): `python3 app_main.py`. Tests:
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
psdprep/naming.py    auto names, versions, original-file guard
psdprep/descriptor.py Photoshop descriptor parser
psdprep/selftest.py  headless self-test (run by CI on the built app)
build.py, ci/, .github/workflows/build.yml   standalone build (Rocky 8 container)
```
