# Cell 27 of arc-agi-3-duck-18-1gc-submit.ipynb (section: Patch cell - v14.74 M74 + M75 (prompt room + field hints))
# ===== v14.74 = M74: COUNT BOARD IMAGES AT THEIR REAL PROMPT COST (club A3 / E4) =====
"""The harness estimates every message by rendered length / 3. A board image is a multi-kilobyte base64 string but
costs the model only a few dozen tokens, so each image is over-counted ~8x and steals prompt room from history.

Qwen3-VL geometry (preprocessor_config.json of the served checkpoint): patch 16, merge 2 -> one visual token per
32x32 px, plus <|vision_start|> and <|vision_end|>. Our boards render 64x64 cells at MULTIMODAL_UPSCALE=4 = 256x256 px
-> (256/32)^2 = 64 tokens + 2 markers = 66, against an estimate of ~505 for the same part.
Nothing is trimmed differently by this patch itself: the trimmer simply stops evicting turns it did not need to evict.
Same fix the rank-3 fork shipped (their `image_part_prompt_tokens`).

READ: m74_summary() -> images_seen, tokens_charged, tokens_saved_vs_old, calls, errors.
KILL: images_seen == 0 -> not installed or no images in the prompt.
"""
from __future__ import annotations

import base64 as _b64
import struct as _struct

_STAT74 = {"calls": 0, "images_seen": 0, "tokens_charged": 0, "tokens_saved_vs_old": 0, "errors": 0, "last_error": "",
           "sizes": {}}

_M74_PATCH, _M74_MERGE = 16, 2
_M74_MIN_PIXELS, _M74_MAX_PIXELS = 65536, 16777216
_M74_MARKERS = 2
_M74_FALLBACK = 64 + _M74_MARKERS


def _m74_png_size(data_url: str):
    """(width, height) from a base64 PNG data URL, without decoding the pixels."""
    head = data_url.split(",", 1)[1][:64]
    raw = _b64.b64decode(head + "=" * (-len(head) % 4))
    if raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        return None
    w, h = _struct.unpack(">II", raw[16:24])
    return int(w), int(h)


def m74_image_tokens(part) -> int | None:
    """Prompt tokens for one image part, or None if the part is not an image."""
    if not isinstance(part, dict) or part.get("type") != "image_url":
        return None
    url = ((part.get("image_url") or {}).get("url")) if isinstance(part.get("image_url"), dict) else None
    size = None
    try:
        size = _m74_png_size(url) if isinstance(url, str) and url.startswith("data:image") else None
    except Exception as exc:
        _STAT74["errors"] += 1
        _STAT74["last_error"] = f"{type(exc).__name__}: {exc}"[:120]
    if size is None:
        return _M74_FALLBACK
    factor = _M74_PATCH * _M74_MERGE
    h_bar = max(factor, int(round(size[1] / factor)) * factor)
    w_bar = max(factor, int(round(size[0] / factor)) * factor)
    if h_bar * w_bar > _M74_MAX_PIXELS:                      # same clamping as the processor's smart_resize
        import math
        beta = ((h_bar * w_bar) / _M74_MAX_PIXELS) ** 0.5
        h_bar = max(factor, int(size[1] / beta // factor) * factor)
        w_bar = max(factor, int(size[0] / beta // factor) * factor)
    elif h_bar * w_bar < _M74_MIN_PIXELS:
        import math
        beta = (_M74_MIN_PIXELS / (h_bar * w_bar)) ** 0.5
        h_bar = math.ceil(size[1] * beta / factor) * factor
        w_bar = math.ceil(size[0] * beta / factor) * factor
    _STAT74["sizes"][f"{w_bar}x{h_bar}"] = _STAT74["sizes"].get(f"{w_bar}x{h_bar}", 0) + 1
    return (h_bar // factor) * (w_bar // factor) + _M74_MARKERS


def install74():
    import inference.agent.tool_agent as T
    orig = T._estimate_tokens
    if getattr(orig, "_m74_wrapped", False):
        raise RuntimeError("M74 installed twice")

    def _strip(node, acc):
        tokens = m74_image_tokens(node)
        if tokens is not None:
            old = orig(node)                                  # what the old estimator charged for this part
            acc[0] += tokens
            acc[1] += max(0, old - tokens)
            _STAT74["images_seen"] += 1
            return ""
        if isinstance(node, dict):
            return {k: _strip(v, acc) for k, v in node.items()}
        if isinstance(node, list):
            return [_strip(v, acc) for v in node]
        return node

    def _estimate_tokens(value):
        _STAT74["calls"] += 1
        try:
            acc = [0, 0]
            payload = _strip(value, acc)
            _STAT74["tokens_charged"] += acc[0]
            _STAT74["tokens_saved_vs_old"] += acc[1]
            return orig(payload) + acc[0]
        except Exception as exc:                              # never block a request on an estimate
            _STAT74["errors"] += 1
            _STAT74["last_error"] = f"{type(exc).__name__}: {exc}"[:120]
            return orig(value)

    _estimate_tokens._m74_wrapped = True
    T._estimate_tokens = _estimate_tokens
    return True


def m74_summary():
    return dict(_STAT74)

install74()
print('V1474_M74 installed', flush=True)
