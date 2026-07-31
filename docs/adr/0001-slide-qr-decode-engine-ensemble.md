# Slide QR decode uses zxing-cpp primary with cv2 fallback

**Status:** revised (2026-06-23)

The archived `id_parsing.py` decoded slide QRs with `pyzbar`, and `requirements.txt`
declares `pyzbar` as the QR engine. Measured on the 47 real `pi_images/` slides,
`pyzbar` alone decoded **0/47** — the QR is a small, soft speck in a 4056×3040
frame. The first implementation decoded with a localize → crop → upscale →
multi-engine ensemble (`cv2.QRCodeDetector` → `zxing-cpp` → `pyzbar`), which
reached **~94% (44/47)**.

Follow-up ablation showed that `zxing-cpp` alone reaches the same **44/47** and
is a strict superset of `cv2` on the current slide set; `pyzbar` contributes no
unique decodes. The production decoder therefore uses `zxing-cpp` first and
keeps `cv2.detectAndDecode` only as a fallback when zxing misses. OpenCV remains
load-bearing for QR localization and image preprocessing.

## Considered options

- **pyzbar only** — the documented/archived approach. Rejected: 0/47 on real data.
- **cv2-first ensemble** — rejected: same 44/47 but misleading engine-credit
  accounting, because cv2 claimed slides that zxing also decoded.
- **zxing-cpp only, no cv2 decode fallback** — measured at 44/47 on current data.
  Rejected for now only because cv2 fallback is cheap insurance when zxing misses
  on future captures.
- **WeChat QR detector** — purpose-built for small/blurry QRs. Rejected for now:
  requires swapping `opencv-python` → `opencv-contrib-python` plus four downloaded
  Caffe model files to manage, and `zxing-cpp` already reaches ~94% with a single
  pip dependency and no model files. Revisit only if recapture is impossible and
  the last ~6% must be recovered in software.

## Consequences

- A future reader must not "simplify" the decoder back to pyzbar-only — that path
  is measured at 0% on this data.
- Do not re-add `pyzbar` without fresh evidence that it contributes a unique
  decode on current Pi captures.
- The remaining ~6% (3 blur-limited lung/lungs captures) are a recapture problem,
  not an engine problem. The decoder fails closed on them.
