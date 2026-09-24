"""
Real-ALMA degradation pairs for Block 2 (PLAN.md): simobserve + tclean on a Jy/pixel model.

Produces one (clean, dirty) pair per `--totaltime` level, laid out like `sg_synth/`
(`run_<id>_<step>_rt_00/run_..._{clean,dirty}.fits`) so every existing loader and
scoring notebook reads them unchanged. Integration time is the degradation axis: thermal
noise scales as 1/sqrt(time), so a list of times is a list of input-degradation levels.

    dirty = tclean(niter=0) of simobserve's noisy MS          [Jy/beam]
    clean = the same sky model convolved to that dirty cube's restoring beam  [Jy/beam]

Runs only in a CASA environment (casatools + casatasks + casadata, Python 3.12):

    ~/Projects/exxa-casa-venv/bin/python tools/alma_simobserve.py \\
        --model "self-gravitating cube and dirty cube/kinematic_data/lines.fits" \\
        --out alma_sim --totaltime 60 --bin 3

The model must be Jy/PIXEL (the un-convolved truth). `lines.fits` is; `clean_sg.fits` is
already beam-convolved (Jy/beam) and would convolve the beam in twice.
"""
import argparse
import os
import shutil

import numpy as np

C_KMS = 299792.458


def read_fits(path, partial=False):
    """Header dict and big-endian data array of a single-HDU FITS, no astropy needed.

    partial=True accepts a truncated file (an unfinished download): the leading axis is cut to
    the whole planes actually present and the header's NAXIS is left as written.
    """
    hdr, off = {}, 0
    with open(path, "rb") as f:
        done = False
        while not done:
            blk = f.read(2880)
            off += 2880
            for i in range(0, 2880, 80):
                card = blk[i:i + 80].decode("ascii", "replace")
                if card.startswith("END "):
                    done = True
                    break
                if card[8:10] == "= ":
                    hdr[card[:8].strip()] = card[10:].split(" /")[0].strip().strip("'").strip()
    dtype = {-32: ">f4", -64: ">f8"}[int(hdr["BITPIX"])]
    shape = tuple(int(hdr[f"NAXIS{k}"]) for k in range(int(hdr["NAXIS"]), 0, -1))
    if partial:
        plane = int(np.prod(shape[1:])) * np.dtype(dtype).itemsize
        shape = ((os.path.getsize(path) - off) // plane,) + shape[1:]
    return hdr, np.memmap(path, dtype=dtype, mode="r", offset=off, shape=shape)


def write_fits(path, data, cards):
    """Minimal FITS writer: float32 cube plus the given (key, value) cards."""
    def card(k, v):
        # FITS fixed format: strings start in column 11 and are padded to 8 characters
        # inside the quotes; numbers and T/F end in column 30. casacore rejects anything else.
        if isinstance(v, bool):
            return f"{k:<8}= {'T' if v else 'F':>20}".ljust(80)
        if isinstance(v, str):
            return f"{k:<8}= {repr(v.ljust(8)).replace(chr(34), chr(39)):<20}".ljust(80)
        return f"{k:<8}= {v!r:>20}".ljust(80)
    head = [card("SIMPLE", True), card("BITPIX", -32), card("NAXIS", data.ndim)]
    head += [card(f"NAXIS{i + 1}", n) for i, n in enumerate(data.shape[::-1])]
    head += [card(k, v) for k, v in cards] + ["END".ljust(80)]
    text = "".join(head)
    text += " " * (-len(text) % 2880)
    raw = np.ascontiguousarray(data, dtype=">f4").tobytes()
    with open(path, "wb") as f:
        f.write(text.encode("ascii"))
        f.write(raw + b"\0" * (-len(raw) % 2880))


def build_skymodel(model_path, out_path, direction, bin_n, pad, crop, scale=1.0):
    """Jy/pixel or Jy/beam cube (velocity OR frequency axis) -> FREQ-axis Jy/pixel cube.

    Two input kinds, told apart by the header:
      * the project's own cubes (`lines.fits`): VELO-LSR, RA/Dec 0,0 (no real sky position,
        so `direction` is used), Jy/pixel;
      * DSHARP fiducial cubes (`AS209_CO.fits`): FREQ, a real RA/Dec, Jy/beam, a degenerate
        Stokes axis in front. Its own coordinates are kept.
    Channels carrying no emission are cropped (keeping `pad` either side), the rest are averaged
    in groups of `bin_n` (flux per pixel is intensity, so averaging preserves it), and the
    spatial axes are cropped to the central `crop` x `crop` pixels.
    Returns (n_channels, cell in mas, image size, direction string for simobserve).
    """
    hdr, cube = read_fits(model_path)
    if cube.ndim == 4:                       # (stokes, chan, y, x) with one Stokes plane
        assert cube.shape[0] == 1, cube.shape
        cube = cube[0]
    bunit = hdr.get("BUNIT", "").upper()
    assert bunit in ("JY/PIXEL", "JY/BEAM"), f"{model_path}: BUNIT {hdr.get('BUNIT')!r}, need Jy/pixel or Jy/beam"
    # Jy/beam -> Jy/pixel: divide by the beam area in pixels, pi/(4 ln2) * bmaj * bmin / pixel^2.
    # The result is only as sharp as that beam, the honest ceiling of a model built from a CLEANed image.
    beam_pixels = 1.0
    if bunit == "JY/BEAM":
        beam_pixels = np.pi / (4 * np.log(2)) * float(hdr["BMAJ"]) * float(hdr["BMIN"]) / float(hdr["CDELT1"]) ** 2
        print(f"Jy/beam model: dividing by {beam_pixels:.2f} pixels per beam")

    ny, nx = cube.shape[1:]
    crop = min(crop, ny, nx)
    y0, x0 = (ny - crop) // 2, (nx - crop) // 2
    window = (slice(None), slice(y0, y0 + crop), slice(x0, x0 + crop))
    flux = np.array([np.abs(np.nan_to_num(np.asarray(cube[k, window[1], window[2]], dtype="f4"))).sum(dtype="f8")
                     for k in range(cube.shape[0])])
    on = np.where(flux > flux.max() * 1e-3)[0]
    lo, hi = max(on.min() - pad, 0), min(on.max() + pad, cube.shape[0] - 1)
    n = (hi - lo + 1) // bin_n * bin_n
    sub = np.nan_to_num(np.asarray(cube[(slice(lo, lo + n),) + window[1:]], dtype="f4")) * (scale / beam_pixels)
    sub = sub.reshape(n // bin_n, bin_n, crop, crop).mean(axis=1)

    ctype3 = hdr["CTYPE3"]
    nu0 = float(hdr.get("RESTFREQ") or hdr["RESTFRQ"])
    c3, d3, r3 = float(hdr["CRVAL3"]), float(hdr["CDELT3"]), float(hdr["CRPIX3"])
    centre = lo + (bin_n - 1) / 2 + 1 - r3          # first output channel's centre, in input channels
    if ctype3.startswith("FREQ"):
        nu_first, dnu = c3 + centre * d3, d3 * bin_n
    else:                                            # VELO / VRAD in km/s, radio convention nu = nu0 (1 - v/c)
        assert ctype3.startswith(("VELO", "VRAD")), ctype3
        nu_first, dnu = nu0 * (1 - (c3 + centre * d3) / C_KMS), -nu0 * d3 * bin_n / C_KMS
    dv = abs(dnu) / nu0 * C_KMS

    real_sky = abs(float(hdr["CRVAL1"])) > 1e-9 or abs(float(hdr["CRVAL2"])) > 1e-9
    ra, dec = (float(hdr["CRVAL1"]), float(hdr["CRVAL2"])) if real_sky else direction
    cell_deg = abs(float(hdr["CDELT1"]))
    write_fits(out_path, sub, [
        ("BUNIT", "Jy/pixel"), ("BTYPE", "Intensity"),
        ("CTYPE1", "RA---SIN"), ("CRVAL1", ra), ("CDELT1", -cell_deg), ("CRPIX1", crop / 2 + 1), ("CUNIT1", "deg"),
        ("CTYPE2", "DEC--SIN"), ("CRVAL2", dec), ("CDELT2", cell_deg), ("CRPIX2", crop / 2 + 1), ("CUNIT2", "deg"),
        ("CTYPE3", "FREQ"), ("CRVAL3", nu_first), ("CDELT3", dnu), ("CRPIX3", 1.0), ("CUNIT3", "Hz"),
        ("SPECSYS", "LSRK"), ("RESTFRQ", nu0), ("RADESYS", "ICRS"), ("EQUINOX", 2000.0),
    ])
    where = "header's real sky position" if real_sky else "--direction (cube carries no position)"
    print(f"sky model: input channels {lo}..{lo + n - 1} of {cube.shape[0]} ({ctype3}), binned x{bin_n} -> "
          f"{sub.shape[0]} ch of {dv:.3f} km/s, {crop}x{crop} px of {cell_deg * 3.6e6:.2f} mas, at {where}")
    return sub.shape[0], cell_deg * 3.6e6, crop, f"J2000 {ra}deg {dec}deg"


def simulate(skymodel, workdir, config, minutes, cell_mas, imsize, direction_str, pwv):
    from casatasks import simobserve, tclean, exportfits, imsmooth, imhead, importfits, imregrid

    project = f"sim_{minutes:05d}min"
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        if os.path.exists(project):
            shutil.rmtree(project)
        simobserve(project=project, skymodel=os.path.abspath(skymodel) if not os.path.isabs(skymodel) else skymodel,
                   setpointings=True, direction=direction_str, mapsize="", obsmode="int",
                   antennalist=config, totaltime=f"{minutes}min", integration="60s",
                   hourangle="transit", thermalnoise="tsys-atm", user_pwv=pwv,
                   graphics="none", overwrite=True)
        import glob
        ms = glob.glob(os.path.join(project, "*.noisy.ms"))
        truth = [t for t in glob.glob(os.path.join(project, "*.skymodel")) if os.path.isdir(t)]
        assert len(ms) == 1 and len(truth) == 1, f"simobserve outputs: {sorted(os.listdir(project))}"
        img = os.path.join(project, "dirty")
        # restoringbeam="common": one beam for every channel, so the truth can be convolved to
        # exactly the resolution the dirty cube is quoted at (per-channel beams would need 100 kernels)
        tclean(vis=ms[0], imagename=img, imsize=imsize, cell=f"{cell_mas}mas", specmode="cube",
               niter=0, weighting="briggs", robust=0.5, gridder="standard",
               restfreq="", outframe="LSRK", restoringbeam="common", interactive=False)
        from casatools import image
        ia = image()
        ia.open(img + ".image")
        beam = ia.commonbeam()
        ia.close()
        # the sky model CASA actually observed, regridded to the image grid, then convolved
        # to the dirty image's own restoring beam: the matching Jy/beam "clean" target
        imregrid(imagename=truth[0], template=img + ".image", output=img + ".truth.regrid", overwrite=True)
        imsmooth(imagename=img + ".truth.regrid", kernel="gauss",
                 major=f"{beam['major']['value']}{beam['major']['unit']}",
                 minor=f"{beam['minor']['value']}{beam['minor']['unit']}",
                 pa=f"{beam['pa']['value']}{beam['pa']['unit']}",
                 targetres=True, outfile=img + ".truth.conv", overwrite=True)
        return os.path.abspath(img + ".image"), os.path.abspath(img + ".truth.conv"), beam
    finally:
        os.chdir(cwd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Jy/pixel VELO-LSR cube (e.g. kinematic_data/lines.fits)")
    ap.add_argument("--out", default="alma_sim", help="output root, sg_synth-style run folders go here")
    ap.add_argument("--config", default="alma.cycle10.6.cfg",
                    help="ALMA antenna config. C-6 gives ~0.12\" at 230 GHz, close to the project cubes' "
                         "0.10x0.16\" beam; C-8 (0.05\") is three times finer than anything we trained on")
    ap.add_argument("--totaltime", type=int, nargs="+", default=[60], help="minutes on source, one pair each")
    ap.add_argument("--bin", type=int, default=3, help="average this many channels (3 -> 0.1 km/s, sg_synth's width)")
    ap.add_argument("--pad", type=int, default=6, help="empty channels kept either side of the emission")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply the model by this. lines.fits is labelled Jy/pixel but sits 1e11 above "
                         "clean_sg.fits at the same pixel (6.7e7 vs 6.7e-4), so use --scale 1e-11 for it")
    ap.add_argument("--crop", type=int, default=600, help="central square of the model to keep, in pixels")
    ap.add_argument("--direction", default="J2000 16h00m00.0 -30d00m00.0",
                    help="phase centre; the cubes carry RA/Dec 0,0, which ALMA cannot observe")
    ap.add_argument("--pwv", type=float, default=1.0, help="precipitable water vapour, mm (band 6 median-ish)")
    ap.add_argument("--run-id", type=int, default=9901, help="RunID for the output folders (99xx = ALMA sims)")
    a = ap.parse_args()

    from casatools import quanta
    qa = quanta()
    _, ra_s, dec_s = a.direction.split()
    ra = qa.convert(qa.quantity(ra_s), "deg")["value"] % 360
    dec = qa.convert(qa.quantity(dec_s), "deg")["value"]

    os.makedirs(a.out, exist_ok=True)
    work = os.path.abspath(os.path.join(a.out, "_casa"))
    os.makedirs(work, exist_ok=True)
    sky = os.path.join(work, "skymodel.fits")
    _, cell_mas, imsize, direction = build_skymodel(a.model, sky, (ra, dec), a.bin, a.pad, a.crop, a.scale)

    for minutes in a.totaltime:
        dirty, clean, beam = simulate(sky, work, a.config, minutes, cell_mas, imsize, direction, a.pwv)
        run = f"run_{a.run_id}_{minutes:05d}_rt_00"
        d = os.path.join(a.out, run)
        os.makedirs(d, exist_ok=True)
        from casatasks import exportfits
        exportfits(imagename=dirty, fitsimage=os.path.join(d, f"{run}_dirty.fits"),
                   velocity=True, dropstokes=True, dropdeg=True, overwrite=True)
        exportfits(imagename=clean, fitsimage=os.path.join(d, f"{run}_clean.fits"),
                   velocity=True, dropstokes=True, dropdeg=True, overwrite=True)
        print(f"{run}: {minutes} min, beam {beam['major']['value']:.3f}x{beam['minor']['value']:.3f} "
              f"{beam['major']['unit']} -> {d}")


if __name__ == "__main__":
    main()
