"""Is V4046 Sgr 13CO (exoALMA, delivered continuum-subtracted) really carrying continuum? Far blue / far red line-free means, their correlation and peak."""
import sys, json, urllib.request, subprocess, os, numpy as np
from scipy import ndimage
sys.path.insert(0, "experiments")
import alma_unified as au
S = "/private/tmp/claude-501/-Users-Krishan-Projects-EXXA/bfdf2b3f-fb27-4ca4-bfeb-56fb5c0abb96/scratchpad"
name = "V4046_Sgr_13CO_fiducial.image.fits"; raw = f"{S}/dl/{name}"
req = urllib.request.Request("https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/CFHWNH", headers={"User-Agent": "Mozilla/5.0 (EXXA GSoC research)"})
fd = {f["dataFile"]["filename"]: f["dataFile"] for f in json.load(urllib.request.urlopen(req, timeout=60))["data"]["latestVersion"]["files"]}[name]
if not (os.path.exists(raw) and os.path.getsize(raw) == fd["filesize"]):
    subprocess.run(["curl", "-sL", "--retry", "5", "-C", "-", "-A", "Mozilla/5.0 (EXXA GSoC research)", "-o", raw, f"https://dataverse.harvard.edu/api/access/datafile/{fd['id']}"], check=True)
cube, velax, hdr, pix, beam = au.read_cube(raw, 512)
lf, _ = au.line_free(cube, beam[0] / pix)
chk = au.continuum_check(cube, lf)
print({k: v for k, v in chk.items() if k != "map"})
line = np.setdiff1d(np.arange(cube.shape[0]), lf); lo, hi = line.min(), line.max()
blue = [k for k in lf if k < lo - 20]; red = [k for k in lf if k > hi + 20]
sig = float(cube[lf].std())
b, r = cube[blue].mean(0), cube[red].mean(0)
sb, sr = sig / np.sqrt(len(blue)), sig / np.sqrt(len(red))
bs, rs = ndimage.gaussian_filter(b, beam[0] / pix / 2.355), ndimage.gaussian_filter(r, beam[0] / pix / 2.355)
print(f"far blue {len(blue)} planes, far red {len(red)}; smoothed peaks blue {bs.max() / bs.std():.1f}, red {rs.max() / rs.std():.1f} (in units of their smoothed rms)")
c = cube.shape[-1] // 2; rr = int(3 * beam[0] / pix)
print(f"centre box mean: blue {b[c-rr:c+rr, c-rr:c+rr].mean() / sb:+.1f} sigma_mean, red {r[c-rr:c+rr, c-rr:c+rr].mean() / sr:+.1f}")
print(f"corr(blue, red) smoothed, whole field {np.corrcoef(bs.ravel(), rs.ravel())[0,1]:.3f}; inner 3 beams {np.corrcoef(bs[c-rr:c+rr, c-rr:c+rr].ravel(), rs[c-rr:c+rr, c-rr:c+rr].ravel())[0,1]:.3f}")
by, bx = np.unravel_index(np.argmax(bs), bs.shape); ry, rx = np.unravel_index(np.argmax(rs), rs.shape)
print(f"peak positions blue ({bx},{by}), red ({rx},{ry}), centre {c}")
