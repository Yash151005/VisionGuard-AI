"""
VisionGuard AI — Autonomous Visual Inspection Agent
Single-file Streamlit prototype for the OpenCV 5 + AWS competition.

Core workflow:
1) Upload an image or video.
2) OpenCV analyzes image/video evidence (ROI selection, grayscale/blur, edges,
   morphology, contours, temporal frame sampling).
3) A deterministic agent chooses a follow-up action based on visual signals.
4) The user can review evidence and export a report.
5) Optional AWS S3 upload stores selected artifacts when configured.

Important: this prototype detects visual anomalies, not certified industrial defects.
Tune thresholds and validate on labeled data before operational use.
"""
from __future__ import annotations

import io
import json
import os
import time
import hashlib
import tempfile
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image

APP_TITLE = "VisionGuard AI"
APP_SUBTITLE = "Autonomous Visual Inspection Agent"
MAX_UPLOAD_MB = 200

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Styling ---------------------------------------
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp {
    background: #07121d;
    background-image: linear-gradient(rgba(31, 72, 93, .09) 1px, transparent 1px), linear-gradient(90deg, rgba(31, 72, 93, .09) 1px, transparent 1px);
    background-size: 48px 48px;
    color: #edf7f5;
    font-family: "Manrope", sans-serif;
}
@import url("https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap");
[data-testid="stHeader"] { background: rgba(7,18,29,.88); }
[data-testid="stSidebar"] { background: #091a26; border-right: 1px solid #1d3d49; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1480px; }
h1, h2, h3 { letter-spacing: -.035em; }
.hero {
    position: relative; overflow: hidden; padding: 1.7rem 1.8rem; border-radius: 10px;
    background: linear-gradient(115deg, #0c2630 0%, #0d3540 58%, #0b202c 100%);
    border: 1px solid #28616a; margin-bottom: 1.25rem; box-shadow: 0 16px 40px rgba(0, 0, 0, .18);
}
.hero:after { content: ""; position: absolute; right: -8%; top: -80%; width: 46%; height: 260%; transform: rotate(24deg); border-left: 1px solid rgba(111, 224, 204, .22); border-right: 1px solid rgba(111, 224, 204, .12); }
.hero h1 { margin: 0; font-size: 2.25rem; font-weight: 800; letter-spacing: -.055em; }
.hero p { margin: .45rem 0 0; color: #b8d0d2; font-size: .96rem; }
.eyebrow, .workspace-kicker, .sidebar-label { color: #72e6d1; font-family: "DM Mono", monospace; font-size: .7rem; letter-spacing: .1em; text-transform: uppercase; }
.kpi {
    border: 1px solid #24505a; border-radius: 8px; padding: 1rem 1.05rem;
    background: rgba(12, 35, 45, .88); min-height: 106px; box-shadow: 0 8px 22px rgba(0, 0, 0, .12);
}
.kpi .label { color: #9bb9bc; font-family: "DM Mono", monospace; font-size: .72rem; letter-spacing: .04em; text-transform: uppercase; }
.kpi .value { color: #72e6d1; font-size: 1.65rem; font-weight: 800; margin-top: .35rem; }
.section {
    border: 1px solid #214852; border-radius: 8px; padding: 1.15rem 1.2rem;
    background: rgba(10, 30, 40, .86); margin: .5rem 0 1rem;
}
.small-muted { color: #9bb9bc; font-size: .84rem; }
.status-note { margin: .5rem 0 1.25rem; padding: .7rem .9rem; border-radius: 6px; font-size: .84rem; border: 1px solid #6b5628; background: #241e12; color: #f0d99b; }
.empty-title { margin: .25rem 0 .45rem; color: #edf7f5; font-size: 1.2rem; font-weight: 700; }
.step { display: flex; gap: .7rem; align-items: flex-start; margin: .7rem 0; color: #c5d9d8; font-size: .9rem; }
.step-number { display: inline-flex; width: 1.45rem; height: 1.45rem; flex: 0 0 1.45rem; align-items: center; justify-content: center; border: 1px solid #4b9f9a; border-radius: 50%; color: #72e6d1; font-family: "DM Mono", monospace; font-size: .7rem; }
div.stButton > button[kind="primary"] {
    background: #72e6d1; color: #062029; border: 0; font-weight: 800;
}
div[data-testid="stFileUploader"] {
    border: 1px dashed #3b7b7e; border-radius: 8px; background: rgba(11, 37, 46, .72); padding: .35rem;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------- Utilities -------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def cv_version() -> str:
    return getattr(cv2, "__version__", "unknown")

def cv5_status() -> tuple[bool, str]:
    v = cv_version()
    try:
        major = int(v.split(".")[0])
    except Exception:
        major = -1
    if major == 5:
        return True, f"OpenCV {v} detected."
    return False, f"OpenCV {v} detected. Competition submissions must run OpenCV 5; install a verified OpenCV 5 build before submission."

def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode this image. Try PNG or JPEG.")
    return img

def resize_for_processing(frame: np.ndarray, max_width: int = 1600) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame.copy(), 1.0
    scale = max_width / float(w)
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale

def analyze_frame(
    frame_bgr: np.ndarray,
    sensitivity: int = 55,
    min_area: int = 120,
    roi: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Classical CV anomaly proposal: local contrast + edges + morphology + contours."""
    image, scale = resize_for_processing(frame_bgr)
    h, w = image.shape[:2]
    x0, y0, x1, y1 = 0, 0, w, h
    if roi:
        rx0, ry0, rx1, ry1 = roi
        x0, y0 = max(0, int(rx0 * w)), max(0, int(ry0 * h))
        x1, y1 = min(w, int(rx1 * w)), min(h, int(ry1 * h))
        if x1 <= x0 or y1 <= y0:
            x0, y0, x1, y1 = 0, 0, w, h
    crop = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    # Sensitivity controls edge thresholds. These are inspectable heuristic signals,
    # not a trained defect classifier.
    low = max(15, int(90 - sensitivity * 0.75))
    high = max(low + 25, int(190 - sensitivity * 0.55))
    edges = cv2.Canny(denoised, low, high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    annotated = image.copy()
    mask_full = np.zeros((h, w), dtype=np.uint8)
    roi_area = max(1, crop.shape[0] * crop.shape[1])
    min_area_scaled = max(10, int(min_area * (crop.shape[0] * crop.shape[1]) / (1280 * 720)))
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_scaled:
            continue
        cx, cy, cw, ch = cv2.boundingRect(contour)
        # Approximate shape features. High perimeter-to-area shapes often include
        # thin edges; this is a heuristic candidate ranking, not defect truth.
        perimeter = cv2.arcLength(contour, True)
        compactness = (4 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0
        aspect = max(cw, ch) / max(1, min(cw, ch))
        score = float(np.clip(0.45 * min(1.0, area / (roi_area * 0.035)) +
                              0.35 * min(1.0, aspect / 6.0) +
                              0.20 * (1.0 - min(1.0, compactness)), 0, 1))
        gx0, gy0 = x0 + cx, y0 + cy
        candidates.append({
            "x": int(gx0), "y": int(gy0), "width": int(cw), "height": int(ch),
            "area_px": int(area), "shape_score": round(score, 3),
            "aspect_ratio": round(float(aspect), 2),
        })
        cv2.rectangle(annotated, (gx0, gy0), (gx0 + cw, gy0 + ch), (0, 70, 255), 2)
        cv2.putText(annotated, f"candidate {score:.2f}", (gx0, max(18, gy0 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 70, 255), 1, cv2.LINE_AA)
        cv2.drawContours(mask_full[y0:y1, x0:x1], [contour], -1, 255, thickness=cv2.FILLED)

    # Visual indicators are scene complexity proxies, not defect probabilities.
    edge_density = float(np.count_nonzero(edges) / max(1, edges.size))
    candidate_count = len(candidates)
    strongest = max([c["shape_score"] for c in candidates], default=0.0)
    signal = float(np.clip(0.55 * min(1.0, edge_density * 7) +
                           0.45 * min(1.0, strongest + candidate_count * 0.04), 0, 1))
    if candidate_count == 0:
        state = "No prominent contour candidates"
    elif strongest >= 0.62 or candidate_count >= 8:
        state = "Review recommended"
    else:
        state = "Candidate regions found"

    return {
        "annotated_bgr": annotated,
        "gray": gray,
        "edges": edges,
        "mask": mask_full,
        "candidates": sorted(candidates, key=lambda c: c["shape_score"], reverse=True),
        "edge_density": edge_density,
        "visual_signal": signal,
        "state": state,
        "processed_width": w,
        "processed_height": h,
        "roi": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
    }

def agent_next_action(result: dict[str, Any], sensitivity: int) -> dict[str, str]:
    """Deterministic, explainable agent policy; visual output changes the next action."""
    count = len(result["candidates"])
    edge_density = result["edge_density"]
    top_score = result["candidates"][0]["shape_score"] if count else 0.0
    if count == 0:
        return {
            "decision": "Broaden inspection",
            "action": "Re-run with higher sensitivity or inspect a different region/frame.",
            "reason": "No contour candidates passed the current area threshold; absence of candidates is not proof that the surface is defect-free.",
            "status": "NEEDS_MORE_EVIDENCE",
        }
    if count >= 8 or top_score >= 0.62 or edge_density > 0.22:
        return {
            "decision": "Request verification",
            "action": "Inspect the top candidate crop and compare with a known-good reference; escalate if evidence remains ambiguous.",
            "reason": f"The visual pass found {count} candidate region(s), strongest shape score {top_score:.2f}, and edge density {edge_density:.3f}.",
            "status": "HUMAN_REVIEW",
        }
    return {
        "decision": "Inspect top candidate",
        "action": "Zoom into the highest-ranked region and inspect additional frames before closing the case.",
        "reason": f"{count} candidate region(s) found; current evidence supports follow-up inspection, not a confirmed defect label.",
        "status": "FOLLOW_UP",
    }

def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def frame_to_jpeg_bytes(frame: np.ndarray, quality: int = 90) -> bytes:
    ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Failed to encode image.")
    return enc.tobytes()

def sample_video(data: bytes, max_frames: int = 12) -> list[tuple[float, np.ndarray]]:
    """Sample evenly spaced frames from an uploaded video using OpenCV."""
    suffix = ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        path = f.name
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError("OpenCV could not open this video. Try MP4 (H.264) format.")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        if total <= 0:
            total = max_frames
        indices = np.linspace(0, max(0, total - 1), min(max_frames, total), dtype=int)
        out = []
        for idx in sorted(set(int(i) for i in indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                seconds = idx / fps if fps > 0 else float(idx)
                out.append((seconds, frame))
        if not out:
            raise ValueError("No frames could be extracted from this video.")
        return out
    finally:
        if cap is not None:
            cap.release()
        try:
            os.unlink(path)
        except OSError:
            pass

def maybe_upload_s3(bucket: str, key: str, payload: bytes) -> str:
    """Upload an artifact to S3 using the normal AWS credential chain / IAM role."""
    import boto3
    client = boto3.client("s3")
    client.put_object(Bucket=bucket, Key=key, Body=payload, ServerSideEncryption="AES256")
    return f"s3://{bucket}/{key}"

# ----------------------------- Header -----------------------------------------
st.markdown(f"""
<div class="hero">
    <div class="eyebrow">VISUAL INSPECTION / CONTROL ROOM</div>
    <h1><span style="color:#72e6d1">VisionGuard</span> AI</h1>
  <p>{APP_SUBTITLE} &nbsp;·&nbsp; <b>See → Investigate → Decide → Act</b></p>
  <p class="small-muted">OpenCV visual analysis · Explainable follow-up policy · Optional AWS evidence storage</p>
</div>
""", unsafe_allow_html=True)

cv5_ok, cv_message = cv5_status()
if cv5_ok:
    st.success(f"Runtime check passed: {cv_message}")
else:
    st.markdown(f'<div class="status-note"><b>Runtime notice</b><br>{cv_message}</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-label">Inspection controls</div>', unsafe_allow_html=True)
    st.header("Configure a run")
    mode = st.radio("Input type", ["Image inspection", "Video inspection"], index=0)
    sensitivity = st.slider("Edge sensitivity", 10, 90, 55, help="Higher values lower the Canny thresholds and usually produce more edge candidates.")
    min_area = st.slider("Minimum candidate area (px)", 20, 1200, 120, step=20)
    max_video_frames = st.slider("Video frames to sample", 4, 24, 12, step=2)
    st.divider()
    st.subheader("Optional AWS S3")
    bucket = st.text_input("S3 bucket name", value=os.getenv("VISIONGUARD_S3_BUCKET", ""))
    prefix = st.text_input("S3 key prefix", value=os.getenv("VISIONGUARD_S3_PREFIX", "visionguard/"))
    st.caption("Configure AWS credentials or an EC2 IAM role outside the app. Never paste access keys here.")
    st.divider()
    st.subheader("Scope & safety")
    st.caption("This is an explainable prototype using classical CV heuristics. It does not certify equipment safety or replace a trained, validated defect classifier.")

upload_types = ["png", "jpg", "jpeg", "bmp", "webp"] if mode == "Image inspection" else ["mp4", "mov", "avi", "mkv"]
upload = st.file_uploader(
    "Upload an inspection image" if mode == "Image inspection" else "Upload an inspection video",
    type=upload_types,
    help=f"Maximum recommended upload size: {MAX_UPLOAD_MB} MB.",
)
if upload and upload.size > MAX_UPLOAD_MB * 1024 * 1024:
    st.error(f"File exceeds the {MAX_UPLOAD_MB} MB application limit.")
    st.stop()

if not upload:
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown('<div class="workspace-kicker">Ready for input</div><div class="empty-title">Start an inspection run</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="section">
        <div class="step"><span class="step-number">01</span><span>Upload a close-up photo or short equipment video.</span></div>
        <div class="step"><span class="step-number">02</span><span>Tune sensitivity and candidate area for the scene.</span></div>
        <div class="step"><span class="step-number">03</span><span>Review visual evidence and ranked regions.</span></div>
        <div class="step"><span class="step-number">04</span><span>Follow the explainable next-step recommendation.</span></div>
        <div class="step"><span class="step-number">05</span><span>Export the evidence report or store it in S3.</span></div>
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown('<div class="workspace-kicker">System status</div>', unsafe_allow_html=True)
        st.markdown("### Runtime snapshot")
        st.metric("OpenCV runtime", cv_version())
        st.metric("AWS evidence storage", "Configured" if bucket else "Optional / not configured")
        st.metric("Analysis mode", "Classical CV + deterministic agent")
    st.stop()

file_bytes = upload.getvalue()
file_hash = sha256_bytes(file_bytes)
run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + file_hash[:8]
st.markdown(f"<div class='small-muted'>Run ID: <code>{run_id}</code> · SHA-256: <code>{file_hash[:16]}…</code></div>", unsafe_allow_html=True)

try:
    frame_results = []
    if mode == "Image inspection":
        original = decode_image(file_bytes)
        result = analyze_frame(original, sensitivity=sensitivity, min_area=min_area)
        frame_results = [(0.0, result)]
    else:
        with st.spinner("Sampling video frames and running OpenCV analysis…"):
            frames = sample_video(file_bytes, max_frames=max_video_frames)
            frame_results = [(ts, analyze_frame(frame, sensitivity=sensitivity, min_area=min_area)) for ts, frame in frames]
        # The representative frame is the one with the strongest visible signal.
        result = max((r for _, r in frame_results), key=lambda r: r["visual_signal"])
        original = None
except Exception as exc:
    st.error(f"Could not process this file: {exc}")
    st.stop()

agent = agent_next_action(result, sensitivity)
candidate_count = len(result["candidates"])
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"<div class='kpi'><div class='label'>Candidate regions</div><div class='value'>{candidate_count}</div></div>", unsafe_allow_html=True)
with m2:
    st.markdown(f"<div class='kpi'><div class='label'>Edge density</div><div class='value'>{result['edge_density']:.1%}</div></div>", unsafe_allow_html=True)
with m3:
    st.markdown(f"<div class='kpi'><div class='label'>Visual signal</div><div class='value'>{result['visual_signal']:.2f}</div></div>", unsafe_allow_html=True)
with m4:
    st.markdown(f"<div class='kpi'><div class='label'>Agent status</div><div class='value' style='font-size:1.05rem'>{agent['status']}</div></div>", unsafe_allow_html=True)

st.markdown("## Visual evidence")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Annotated inspection frame**")
    st.image(bgr_to_rgb(result["annotated_bgr"]), use_container_width=True)
with c2:
    st.markdown("**OpenCV edge map**")
    st.image(result["edges"], clamp=True, use_container_width=True)

with st.expander("Inspect additional OpenCV outputs"):
    a, b = st.columns(2)
    with a:
        st.markdown("**Grayscale / denoised input**")
        st.image(result["gray"], clamp=True, use_container_width=True)
    with b:
        st.markdown("**Candidate mask**")
        st.image(result["mask"], clamp=True, use_container_width=True)

st.markdown("## Agent decision trace")
st.markdown(f"""
<div class="section">
  <div style="color:#00d3ee;font-weight:700;font-size:1.05rem">{agent['decision']}</div>
  <p><b>Selected next action:</b> {agent['action']}</p>
  <p><b>Why this action:</b> {agent['reason']}</p>
  <p class="small-muted">The action is selected from measured OpenCV outputs. This is a deterministic agent policy, not a claim of autonomous physical control.</p>
</div>
""", unsafe_allow_html=True)

if mode == "Video inspection":
    st.markdown("## Temporal inspection summary")
    rows = []
    for ts, r in frame_results:
        a = agent_next_action(r, sensitivity)
        rows.append({
            "timestamp_s": round(ts, 2),
            "candidate_regions": len(r["candidates"]),
            "edge_density": round(r["edge_density"], 4),
            "visual_signal": round(r["visual_signal"], 3),
            "agent_status": a["status"],
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

st.markdown("## Candidate regions")
if result["candidates"]:
    st.dataframe(result["candidates"][:30], use_container_width=True, hide_index=True)
else:
    st.info("No regions passed the current contour threshold. This does not establish that the asset is defect-free; try another frame, region, or sensitivity.")

# Compose report with no raw uploaded image embedded.
report = {
    "application": APP_TITLE,
    "run_id": run_id,
    "created_at_utc": utc_now(),
    "input": {
        "filename": upload.name,
        "mode": mode,
        "bytes": upload.size,
        "sha256": file_hash,
    },
    "runtime": {
        "opencv_version": cv_version(),
        "opencv5_requirement_satisfied": cv5_ok,
        "numpy_version": np.__version__,
    },
    "analysis": {
        "method": "OpenCV grayscale, Gaussian blur, Canny edges, morphological close, external contours",
        "sensitivity": sensitivity,
        "minimum_candidate_area_px": min_area,
        "candidate_count": candidate_count,
        "edge_density": round(result["edge_density"], 6),
        "visual_signal_heuristic": round(result["visual_signal"], 4),
        "state": result["state"],
        "candidates": result["candidates"][:100],
    },
    "agent": agent,
    "limitations": [
        "Contour and edge heuristics are not a trained or validated defect classifier.",
        "Visual signal and shape scores are not calibrated probabilities of defect.",
        "Results depend on lighting, camera angle, texture, resolution, and selected thresholds.",
        "Human review is required before maintenance or safety decisions.",
    ],
}
report_bytes = json.dumps(report, indent=2).encode("utf-8")
st.download_button(
    "⬇️ Download inspection report (JSON)",
    data=report_bytes,
    file_name=f"visionguard_report_{run_id}.json",
    mime="application/json",
    type="primary",
)

annotated_bytes = frame_to_jpeg_bytes(result["annotated_bgr"])
if bucket:
    st.markdown("## Optional AWS evidence storage")
    if st.button("Upload report and annotated frame to S3"):
        try:
            with st.spinner("Uploading to S3 using the configured AWS credential chain…"):
                safe_name = os.path.basename(upload.name).replace(" ", "_")
                report_uri = maybe_upload_s3(bucket, f"{prefix.rstrip('/')}/{run_id}/report.json", report_bytes)
                image_uri = maybe_upload_s3(bucket, f"{prefix.rstrip('/')}/{run_id}/annotated.jpg", annotated_bytes)
            st.success("Upload complete.")
            st.code(f"{report_uri}\n{image_uri}")
        except Exception as exc:
            st.error(f"S3 upload failed: {exc}")
            st.caption("Check bucket name, region, IAM permissions, and the AWS credential chain.")
else:
    st.caption("S3 is optional. Enter a bucket name in the sidebar to enable uploads; use an IAM role or configured AWS profile.")

with st.expander("Reproducibility and responsible use"):
    st.write({
        "run_id": run_id,
        "input_sha256": file_hash,
        "opencv_version": cv_version(),
        "analysis_method": report["analysis"]["method"],
        "responsible_use": "Treat output as a triage aid only. Validate against labeled test data and require human approval for consequential actions.",
    })
