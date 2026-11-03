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

On Windows, the same three lines with `.\.venv\Scripts\activate`, or use
`start.bat` / `start.ps1`, which create `.venv` and install when needed.

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

```sh
python -m pip install pytest
python -m pytest tests -q
```

The suite runs the real detector on a fixture in `tests/assets/`, and pins the
choices that were expensive to learn: absolute-pixel landmarks inside a box
that does not start at the origin, sharpness invariant across input sizes, and
a rejection that always explains itself.

## Licence

This project: MIT. The face detection model in `models/` ships under its own
terms; that file is next to it, out at `models/LICENSE-yunet.txt`.

## Hosting

GitHub stores the source. GitHub Pages cannot run this Python API, so deploy
the FastAPI service to a Python-capable host.
