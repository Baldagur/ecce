# PSD Prep

Open a PSD, then **remove layers**, **crop**, and **convert CMYK → sRGB**. Save the result as a
**layered PSD** and/or a **flattened PNG/TIFF with alpha**. It's built for Rocky Linux and works
without internet access.

## Use

```
./psdprep                 # GUI (or double-click)
./psdprep file.psd        # GUI with a file open (drag & drop works too)
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
* **Save PSD…** / **Export flattened…** / **Save both…**. The original file is never overwritten.
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
psdprep in.psd --list
psdprep in.psd --remove "Retouch*" --remove-hidden --crop 0,0,1920,1080 \
        --out-psd out.psd --out-flat out.png --depth 16
psdprep in.psd --profile /path/CoatedFOGRA39.icc --out-flat out.tif
```

## Limits (v0.1)

* Supports 8 and 16-bit Grayscale/RGB/CMYK/Lab PSD and PSB. 32-bit, indexed, duotone and
  bitmap files aren't supported yet.
* In the saved PSD, text layers, smart objects and fill layers are kept as pixels (they lose
  editability). Adjustment layers and layer styles are dropped. Groups, masks, opacity, blend
  modes, visibility and clipping are kept.
* 16-bit CMYK is converted through a 4D LUT sampled from littleCMS (18 nodes per channel).
  Against the exact 8-bit path the difference is ≤ ~1.5/255.

## Build the double-click app (offline Rocky)

On any machine with internet, download the wheels for the Rocky Python version:

```
pip download -d wheels --only-binary=:all: --platform manylinux_2_28_x86_64 \
    --python-version 3.11 numpy Pillow PySide6-Essentials pyinstaller
```

Copy the repo and the `wheels/` folder to Rocky, then:

```
python3.11 -m pip install --user --no-index --find-links wheels numpy Pillow PySide6-Essentials pyinstaller
PYTHON=python3.11 ./build.sh        # -> dist/psdprep
```

Put any CMYK `.icc` profiles in `profiles/` before building and they get bundled. The app also
finds system profiles (e.g. Ghostscript's `/usr/share/ghostscript/*/iccprofiles`).

Run from source: `python3 main.py`. Tests: `python3 tests/test_roundtrip.py`.

## Layout

```
psdprep/psdfile.py   PSD/PSB reader + writer (numpy only; Pillow speeds up RLE decode)
psdprep/ops.py       layer tree, remove, crop, ICC conversion, compositor
psdprep/imgio.py     PNG / TIFF writers (8/16-bit, alpha, ICC)
psdprep/pipeline.py  settings -> outputs
psdprep/gui.py       PySide6 UI (falls back to PySide2)
psdprep/cli.py       command line / entry point
```
