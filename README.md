# Clear face detection API

FastAPI service that decides whether a photo is usable for face enrollment.

A face passes only when one face is in frame, the face is large enough
(80x80 px), sharp enough, well lit, and its five landmarks sit inside the box at a
plausible spacing and roll angle. This is a quality gate, not a recogniser: it does
no identity matching and ships no recognition weights.

## Requirements

- Python 3.10 or newer
- A face detector. Default: **YuNet** (OpenCV `cv2.FaceDetectorYN`,
  Apache-2.0), fetched by `python -m app.fetch_models` and sha256-checked.
- Internet access on first run only, to install packages and fetch the blob.

No InsightFace dependency, and no recognition runtime: `insightface`'s model
zoo is non-commercial research only, so it is not imported anywhere in this
project. If you swap the detector, check the licence of what you add — the
code here is commercially usable, a dropped-in blob may not be.

## Setup

Run these from the repository folder.

On macOS or Linux:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m app.fetch_models
```

On Windows (PowerShell):

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m app.fetch_models
```

In cmd.exe the activate line is `.\.venv\Scripts\activate.bat`; the rest is
identical. If `py` is not on PATH, use `python` and check with `python -V`
that it is 3.10 or newer.

`start.bat` and `start.ps1` do all of the above on first run and then start
the server, which is the shortest path if you only want to look at it.

## Run

```sh
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8011
```

Open <http://127.0.0.1:8011/> in a browser. The same process serves the upload
and webcam page, the overlay that draws the detected box and landmarks, and the
API: `app.mount("/", StaticFiles(directory=<repo>/static, ...))` is absolute, so
the working directory does not matter.

## API

One endpoint: **POST** `/api/verify`. JSON body with an `image` field holding
base64 bytes; a data URL (`data:image/jpeg;base64,...`) is accepted too. 10 MB
limit after decoding.

```json
{ "image": "/9j/4AAQSkZJRgABAQ..." }
```

Every response carries the same keys, accepted or not, so a rejection can be
inspected and not just read:

- `accepted` — bool
- `flaws` — string, human-readable reason. Never empty on a rejection.
- `face_detected`, `face_count`, `image_size`
- `facial_area` — `{x, y, width, height}` in source pixels
- `landmarks` — five `[x, y]` points, source pixels
- `detection_confidence` — YuNet score
- `quality` — `blur_variance`, `brightness`, `face_size_px`
- `checks` — one `{name, passed, detail}` per gate
- `faces` — the same box/landmarks/confidence for every face in frame

```json
{
  "accepted": false,
  "flaws": "Face is too blurry (sharpness 33.2; need at least 50). Hold still.",
  "facial_area": {"x": 654, "y": 564, "width": 443, "height": 644},
  "landmarks": [[690.1, 620.2], [905.4, 618.9], [801.2, 745.1], [712.4, 870.7], [878.2, 868.9]],
  "detection_confidence": 0.912,
  "quality": {"blur_variance": 33.2, "brightness": 142.7, "acceptable": false}
}
```

Example with `curl`:

```sh
curl -X POST http://127.0.0.1:8011/api/verify \
  -H "Content-Type: application/json" \
  -d "{\"image\":\"$(base64 -w0 photo.jpg)\"}"
```

## Sharpness is scale-normalised

Laplacian variance grows with pixel count: the same sharp face scores 506 in a
1600 px box and 12 in a 512 px box. A fixed cutoff would therefore judge the
camera's resolution rather than its focus, and reject every large, sharp photo
while passing small crops. The crop is resampled to `BLUR_REFERENCE_WIDTH` (112 px)
before measuring, so `MIN_BLUR_VARIANCE = 50` means the same thing at any input size.

## Tests

On Windows (PowerShell):

```powershell
python -m pytest tests -q
```

On macOS or Linux:

```sh
python -m pytest tests -q
```

pytest ships in `requirements.txt`, so the install above is enough.

Expected: 22 tests, 19 pass, 3 fail in `test_align.py`:

- `test_projected_landmarks_are_closer_than_insightface` compares a float64
  fit with `target_template()` against `face_align.estimate_norm`, which
  returns **float32**. On these five points the float64 fit is the smaller
  minimax solution (2.283 px) and the reference the larger (2.286 px), so the
  test measures rounding, not geometry. It only ever passed on a float32 fit.
- `test_similarity_matrix_keeps_scale` pins the scale drawn from that float32
  matrix, so it moves in the sixth digit for the same reason.
- `test_square_crop_keeps_true_origin_inside` expects the crop origin to be
  `(37, 51)`; the padding in `square_crop` gives `(31, 31)`. **Unchecked:** I
  have not established which is right, so it is left red rather than bent.

The other 19 pass, including the detector and API-contract suites and the
sharpness-invariant-across-sizes tests. If `insightface` or `scikit-image`
is not installed, the reference-comparison tests skip by design; neither is
a dependency of the app.

Windows will ask for a firewall exception the first time. Allow it for
private networks only; the server binds to 127.0.0.1, so nothing on the LAN
needs it.

## Licence

This project: MIT. The models in `models/` ship under their own terms, and
each one has its licence text next to it:

- `face_detection_yunet_2023mar.onnx` -- MIT (OpenCV Zoo), in
  `models/LICENSE-yunet.txt`.
- `face_landmarker.task` -- Apache-2.0 (Google MediaPipe), in
  `models/LICENSE-face_landmarker.txt`. It is redistributed unmodified and
  unretrained; `models/README.md` records the upstream URL, the byte count,
  and the SHA-256 of the exact artefact. `app/_eyes.py` refuses to load a file
  whose digest differs, so neither licence nor behaviour can drift silently.

## Hosting

GitHub stores the source. GitHub Pages cannot run this Python API, so deploy
the FastAPI service to a Python-capable host.
