# Clear face detection API

FastAPI service that uses InsightFace SCRFD-10GF to detect faces. An image passes when exactly one face is detected with sufficient confidence, size, sharpness, and lighting; both eyes are visible and classified open; OpenCV's frontal-face cascade confirms the internal facial features; and mouth detail is visible at SCRFD's mouth landmarks. Partially covered faces are rejected.

## Requirements

- Python 3.10 or newer
- Internet access on first run to install packages and download the InsightFace `buffalo_l` model pack

InsightFace's published pretrained model weights are licensed for non-commercial research use. For commercial deployment, use a separately licensed model.

## Setup

Clone the repository and run the commands from its folder.

On Windows:

```powershell
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS or Linux:

```sh
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

On Windows, run `start.bat` (or `start.ps1` in PowerShell) from the project folder. These scripts create `.venv` and install dependencies when needed.

On macOS or Linux, activate the virtual environment and run:

```sh
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The first request downloads the InsightFace `buffalo_l` model pack, which includes SCRFD-10GF face detection weights.

## Hosting

GitHub stores the source code. GitHub Pages can host static files, but it cannot run this Python API. To let other people use the API over the internet, deploy the FastAPI service to a Python-capable hosting service.

## API

There is one endpoint: **POST** `/api/verify`. Send JSON with an `image` field containing base64-encoded image bytes. A data URL such as `data:image/jpeg;base64,...` is also accepted. Images are limited to 10 MB after decoding.

```json
{
  "image": "/9j/4AAQSkZJRgABAQ..."
}
```

The response always has an `accepted` boolean and a `flaws` string. `flaws` is empty when the face is accepted; otherwise it describes the detected issue(s).

```json
{
  "accepted": false,
  "flaws": "Face is too blurry (sharpness 12.4; need at least 50); Both eyes must be open and clearly visible."
}
```

Example using `curl` with a base64 string:

```sh
curl -X POST http://127.0.0.1:8000/api/verify \
  -H "Content-Type: application/json" \
  -d '{"image":"BASE64_IMAGE_DATA"}'
```
