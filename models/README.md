This directory holds model weights fetched by `python -m app.fetch_models`,
plus the licences their upstreams require. The weights are not in git.

## face_detection_yunet_2023mar.onnx  --  MIT

Copyright (c) 2020 Shiqi Yu <shiqi.yu@gmail.com>, and contributors.
Licence text: `LICENSE-yunet.txt`
Upstream:   https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
Paper:      Shiqi Yu et al., "YuNet: A Tiny Millisecond-level Face Detector",
            Machine Intelligence Research, 2023.

## face_landmarker.task  --  Apache-2.0

Copyright The MediaPipe Authors (Google LLC).
Licence text: `LICENSE-face_landmarker.txt` (Apache License 2.0,
             http://www.apache.org/licenses/LICENSE-2.0)
Upstream:   https://storage.googleapis.com/mediapipe-models/face_landmarker/
            face_landmarker/float16/latest/face_landmarker.task
Bytes:      3,758,596
SHA-256:    64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff
Channel:    float16, "latest" tag as of 2026-11-03

Changes made: none. The file here is the upstream artefact, byte for byte;
`app/_eyes.py` refuses a file whose SHA-256 differs.
NOTICE: this project redistributes it unmodified and unretrained, and
states so here, as Apache-2.0 section 4 asks.

What the file is not: it carries no eye-open/closed label and was not
trained on one. The closed-eye check in `app/_eyes.py` is arithmetic over
the 478 landmark points it returns (the Eye Aspect Ratio), not a third
classifier output.
