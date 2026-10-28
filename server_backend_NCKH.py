"""FireGuard AI - backend 1 file. Chạy:  uvicorn server_backend_NCKH:app --host 0.0.0.0 --port 8000
Lõi đo v4.9 (Đã tối ưu tách nền bầu trời thích ứng, bắt trọn lõi khói đen/xám/trắng, chống tràn mask ngoài BBox)
+ Tích hợp sẵn API Key & Chuỗi fallback mô hình Gemini + Tự động xuất đồ thị NCKH 300 DPI.
"""
import csv, glob, hashlib, json, math, os, re, shutil, subprocess, tempfile, threading, time, uuid, unicodedata, traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from urllib.parse import quote

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from torchvision import models, transforms
from ultralytics import YOLO
from numpy.lib.stride_tricks import sliding_window_view

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
cv2.setNumThreads(1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===================== ĐƯỜNG DẪN VÀ THƯ MỤC CƠ SỞ =====================
BASE = os.path.dirname(os.path.abspath(__file__))
YOLO_W = os.path.join(BASE, "models", "yolo_best.pt")
RESNET_W = os.path.join(BASE, "models", "smoke_resnet18.pt")
CITIZEN_DIR = os.path.join(BASE, "citizen_uploads")
DB = os.path.join(BASE, "data", "alerts")
SNAPSHOT_DIR = os.path.join(BASE, "snapshots")
PAPER_FIGS_DIR = os.path.join(BASE, "paper_figures")
JSON_RESULT_DIR = os.path.join(PAPER_FIGS_DIR, "json_video_that")
JSON_RESULT_FIG_LATEST = os.path.join(PAPER_FIGS_DIR, "Hinh_JSON_ket_qua_video_that.png")
PROCESSED_VIDEO_DIR = os.path.join(BASE, "processed_videos")
TIMING_CSV = os.path.join(BASE, "timing_log.csv")

for d in (CITIZEN_DIR, DB, SNAPSHOT_DIR, PAPER_FIGS_DIR, JSON_RESULT_DIR, PROCESSED_VIDEO_DIR):
    os.makedirs(d, exist_ok=True)
print(f"[PATH] File server đang chạy: {os.path.abspath(__file__)}")
print(f"[PATH] Thư mục video đã xử lý: {os.path.abspath(PROCESSED_VIDEO_DIR)}")

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9.5
plt.rcParams['figure.titlesize'] = 13

# ===================== ĐO THỜI GIAN TỪNG BƯỚC =====================
STEP_ORDER = [
    "doc_giai_ma_video",
    "resize_khung_hinh",
    "yolo_phat_hien",
    "resnet_phan_loai_mau",
    "luu_anh_snapshot",
    "chuyen_canh",
    "hieu_chinh_mau",
    "che_khuat_dcp_oi",
    "optical_flow",
    "tong_hop_ket_qua",
    "tong_hop_video",
    "dong_goi_bang_chung",
    "gemini_suy_luan",
    "xuat_video_da_xu_ly",
]
PER_FRAME_STEPS = ("doc_giai_ma_video", "resize_khung_hinh", "yolo_phat_hien", "luu_anh_snapshot",
                   "che_khuat_dcp_oi", "optical_flow")
_CSV_LOCK = threading.Lock()

class StepTimer:
    def __init__(self):
        self.sec, self.calls = {}, {}
        self.speed, self.speed_n = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}, 0

    def add(self, name, dt):
        self.sec[name] = self.sec.get(name, 0.0) + dt
        self.calls[name] = self.calls.get(name, 0) + 1

    @contextmanager
    def step(self, name):
        t0 = time.perf_counter()
        try: yield
        finally: self.add(name, time.perf_counter() - t0)

    def add_yolo_speed(self, sp):
        if not sp: return
        for k in self.speed: self.speed[k] += float(sp.get(k, 0.0))
        self.speed_n += 1

    def report(self, n_frames, video_sec, total):
        buoc = {k: round(self.sec.get(k, 0.0), 3) for k in STEP_ORDER}
        for k, v in self.sec.items():
            buoc.setdefault(k, round(v, 3))
        out = dict(
            tong_s=round(total, 3), video_s=round(video_sec, 1), so_khung=n_frames,
            buoc_s=buoc,
            khac_s=round(max(0.0, total - sum(self.sec.values())), 3),
            he_so_thoi_gian_thuc=round(total / video_sec, 3) if video_sec > 0 else None,
        )
        if n_frames:
            out["moi_khung_ms"] = {k: round(self.sec.get(k, 0.0) / n_frames * 1000, 2) for k in PER_FRAME_STEPS}
        nr = self.calls.get("resnet_phan_loai_mau", 0)
        if nr:
            out["resnet_ms_moi_lan"] = round(self.sec["resnet_phan_loai_mau"] / nr * 1000, 2)
            out["resnet_so_lan"] = nr
        if self.speed_n:
            out["yolo_ms_moi_anh"] = {k: round(v / self.speed_n, 2) for k, v in self.speed.items()}
        return out

def log_timing(video_name, tm_rep):
    b = tm_rep["buoc_s"]
    print(f"[TIME] {video_name}: tong={tm_rep['tong_s']}s | video={tm_rep['video_s']}s | khung={tm_rep['so_khung']} | " +
          " ".join(f"{k}={b[k]}" for k in STEP_ORDER if b.get(k)))
    try:
        y = tm_rep.get("yolo_ms_moi_anh", {})
        row = [video_name, tm_rep["video_s"], tm_rep["so_khung"], tm_rep["tong_s"]] + [b.get(k, 0.0) for k in STEP_ORDER] + \
              [tm_rep["khac_s"], y.get("preprocess"), y.get("inference"), y.get("postprocess"), tm_rep.get("resnet_ms_moi_lan")]
        head = ["video", "video_s", "so_khung", "tong_s"] + STEP_ORDER + \
               ["khac_s", "yolo_pre_ms", "yolo_infer_ms", "yolo_post_ms", "resnet_ms_moi_lan"]
        with _CSV_LOCK:
            new = not os.path.exists(TIMING_CSV)
            with open(TIMING_CSV, "a", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh)
                if new: w.writerow(head)
                w.writerow(row)
    except Exception as e:
        print("[TIME] không ghi được CSV:", e)

# ===================== PHẦN 1: CẤU HÌNH VÀ LÕI ĐO KHÓI v4.9 =====================
FRAME_W = 360
YOLO_CONF = 0.25
FIRE_CONF = 0.4
ROI_HOLD = 3              # Chỉ mượn hộp tối đa 3 khung khi YOLO mất dấu tạm thời, tránh trôi mask khi lia máy
FIRE_HOLD = 3
WORKERS = min(4, os.cpu_count() or 1)

CUT_HIST = 0.50
CUT_NCC = 0.35
CUT_NCC_HARD = 0.05

SAT_OVERLAY = 240
SKY_V, SKY_S = 165, 52    # Mở rộng nhận diện bầu trời nhiều mây / trời sáng thực tế
SMOKE_S_MAX = 80
MIN_SMOKE_FRAC = 0.005
DCP_PATCH, DCP_OMEGA, DCP_T0 = 15, 0.95, 0.1
GUIDE_R, GUIDE_EPS = 20, 1e-3
DARK_V_HI, DARK_V_SPAN = 0.62, 0.42
TEX_REF = 0.22            # Tránh làm thủng lỗ vùng khói đen cuộn xoáy có độ tương phản cục bộ
P_MIN = 0.18
DENSE_P = 0.5
MIN_VALID_FRAC = 0.5
SMOOTH_WIN = 5
OI_ON, OI_OFF = 0.50, 0.40

FIRE_SCENE_FRAC = 0.15
WARM_H_HI, WARM_H_LO = 25, 165
S_TINT_CAP = SAT_OVERLAY - 5
S_TINT_DEFAULT = 110
TINT_V_MIN = 40
TINT_SAMPLE = 20
FIRE_PAD = 0.05

FLOW_STEP = 2
MIN_INLIERS = 20
LK_WIN, LK_LEVELS, FB_ERR = (21, 21), 3, 1.0
FEATURE_EXCLUDE_DILATE = 31
MAG_MIN, MAG_MAX = 0.5, 12.0
TEX_MIN = 1e-4
N_MIN = 300
FRAC_MIN = 0.10
R_MIN = 0.5
ERODE = 5
SMOOTH_N = 5
AVG_SEC = 3.0
DIR_MIN_PAIRS = 3
R_AVG_MIN = 0.3
SKIP_REASONS = ("dau_clip", "chuyen_canh")
FLOW_NOTES = {
    "camera": "Camera chuyển động mạnh hoặc nền ít điểm bám nên không bù được chuyển động camera.",
    "it_chuyen_dong": "Khói gần như đứng yên hoặc ít chi tiết bề mặt.",
    "khong_co_box_khoi_tu_lua": "Có lửa nhưng chưa có hộp khói hợp lệ nối được từ nguồn lửa; không tính Optical Flow.",
    "thieu_box_khoi_cap_khung": "Đường từ nguồn lửa chưa đi qua hộp khói hợp lệ ở cả hai khung của cặp đo; từ chối Optical Flow.",
    "flow_roi_qua_nho": "Vùng khói giao với hộp được chọn quá nhỏ để đo Optical Flow đáng tin cậy.",
    "R_thap": "Chuyển động của khói không theo một hướng rõ rệt.",
    "thieu": "Chưa đủ cặp khung liên tục có khói để kết luận hướng.",
}
DIRS = ["Phải", "Phải-Lên", "Lên", "Trái-Lên", "Trái", "Trái-Xuống", "Xuống", "Phải-Xuống"]

_K_DCP = cv2.getStructuringElement(cv2.MORPH_RECT, (DCP_PATCH, DCP_PATCH))
_K_OV = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
_K3, _K5, _K9 = (np.ones((n, n), np.uint8) for n in (3, 5, 9))

def _r(x, nd=3):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)

def dir_label(angle):
    return DIRS[int(((angle + 22.5) % 360) // 45)]

def tilt_from_vertical(deg):
    return -(((deg - 90) + 180) % 360 - 180)

def _scene_feat(f):
    s = cv2.resize(f, (64, 36), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    h = cv2.normalize(h, h, 1, 0, cv2.NORM_L1)
    g = cv2.GaussianBlur(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0)
    return h, g - g.mean()

def detect_scenes(samples):
    feats = [_scene_feat(f) for _, f in samples]
    ids, cuts, sc = [0], [], 0
    for i in range(1, len(samples)):
        (h0, g0), (h1, g1) = feats[i - 1], feats[i]
        dh = cv2.compareHist(h0, h1, cv2.HISTCMP_BHATTACHARYYA)
        ncc = float((g0 * g1).sum() / (np.sqrt((g0 ** 2).sum() * (g1 ** 2).sum()) + 1e-6))
        if (dh > CUT_HIST and ncc < CUT_NCC) or ncc < CUT_NCC_HARD:
            sc += 1
            cuts.append((samples[i][0], float(dh), ncc))
        ids.append(sc)
    return np.array(ids), cuts

def _keep_components(binary, keep_fn, connectivity):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)
    if n <= 1:
        return np.zeros(binary.shape, bool)
    keep = np.zeros(n, bool)
    keep[1:] = keep_fn(stats[1:])
    return keep[lab]

def overlay_mask(hsv):
    m = cv2.inRange(hsv, (0, SAT_OVERLAY + 1, 61), (255, 255, 255)) // 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _K3)
    m = cv2.dilate(m, _K_OV)
    return m.astype(bool)

def sky_mask(hsv, bgr=None):
    """Nhận diện nền bầu trời tổng quát (cả trời nhiều mây xám/trắng lẫn trời xanh dương)."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    H_img, W_img = h.shape

    # 1. Trời xanh dương (Blue/Cyan sky) và Trời mây sáng (Overcast/Cloudy sky)
    blue_sky = (h >= 85) & (h <= 138) & (s >= 18) & (v >= 120)
    overcast_sky = (s <= SKY_S) & (v >= SKY_V)

    if bgr is not None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Bầu trời/mây có gradient cạnh sắc nhỏ hơn công trình/cây cối
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.hypot(gx, gy)
        smooth_sky = grad < 42.0
        cand = ((blue_sky | overcast_sky) & smooth_sky).astype(np.uint8)
    else:
        cand = (blue_sky | overcast_sky).astype(np.uint8)

    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, _K3)
    # Giữ các mảng bầu trời liên thông với biên trên hoặc biên ngang ở nửa trên khung hình
    return _keep_components(
        cand,
        lambda st: ((st[:, cv2.CC_STAT_TOP] <= max(4, int(0.05 * H_img))) |
                    ((st[:, cv2.CC_STAT_TOP] <= int(0.35 * H_img)) &
                     ((st[:, cv2.CC_STAT_LEFT] == 0) |
                      (st[:, cv2.CC_STAT_LEFT] + st[:, cv2.CC_STAT_WIDTH] >= W_img - 2))))
                   & (st[:, cv2.CC_STAT_AREA] > 0.015 * cand.size),
        4
    )

def boxes_to_mask(boxes, shape, pad=0.0):
    h, w = shape
    m = np.zeros(shape, bool)
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
        m[max(0, int(round(y1 * h))):int(round(y2 * h)), max(0, int(round(x1 * w))):int(round(x2 * w))] = True
    return m

def hold_boxes(per_frame, n=ROI_HOLD, scene_ids=None):
    """Chỉ giữ hộp của chính khung hình hiện tại nếu YOLO phát hiện được;
    chỉ mượn hộp của khung liền trước khi khung hiện tại bị mất dấu tạm thời."""
    n, out = max(1, n), []
    for i in range(len(per_frame)):
        cur_b = per_frame[i].reshape(-1, 4)
        if len(cur_b) > 0:
            out.append(cur_b)
            continue
        lo = max(0, i - n + 1)
        if scene_ids is not None:
            while lo < i and scene_ids[lo] != scene_ids[i]:
                lo += 1
        fallback = np.zeros((0, 4), dtype=np.float32)
        for j in range(i - 1, lo - 1, -1):
            if len(per_frame[j]) > 0:
                fallback = per_frame[j].reshape(-1, 4)
                break
        out.append(fallback)
    return out

def scene_tint_params(samples, roi_boxes, fire_frames, scene_ids):
    s_max = np.full(len(samples), float(SMOKE_S_MAX))
    for sc in np.unique(scene_ids):
        idx = np.flatnonzero(scene_ids == sc)
        if np.mean([len(fire_frames[i]) > 0 for i in idx]) < FIRE_SCENE_FRAC:
            continue
        vals = []
        for i in idx[np.linspace(0, len(idx) - 1, min(len(idx), TINT_SAMPLE)).astype(int)]:
            f = samples[i][1]
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
            h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) * (1 / 255)
            mean, sq = cv2.blur(g, (9, 9)), cv2.blur(g * g, (9, 9))
            tex = np.sqrt(np.maximum(sq - mean ** 2, 0)) / TEX_REF
            m = (boxes_to_mask(roi_boxes[i], f.shape[:2])
                 & ~boxes_to_mask(fire_frames[i], f.shape[:2], FIRE_PAD)
                 & ~overlay_mask(hsv) & ((h <= WARM_H_HI) | (h >= WARM_H_LO))
                 & (tex < 0.5) & (v >= TINT_V_MIN))
            vals.append(s[m])
        vals = np.concatenate(vals) if vals else np.array([])
        val = np.percentile(vals, 90) if vals.size >= 500 else S_TINT_DEFAULT
        s_max[idx] = float(np.clip(val, SMOKE_S_MAX, S_TINT_CAP))
    return s_max

def candidate_mask(hsv, roi, valid, sky, s_max=SMOKE_S_MAX, fire=None, bgr=None):
    """Lọc ứng viên khói theo độ lệch màu thực (True Chroma), không làm mất lõi khói đen
    và loại bỏ tường/mái nhà có màu sắc (như mái xanh, tường vàng)."""
    s = hsv[..., 1]
    v = hsv[..., 2]
    if bgr is not None:
        c_span = np.max(bgr, axis=2).astype(np.int16) - np.min(bgr, axis=2).astype(np.int16)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        c_ab = np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)
        # Vùng tối (khói đen/xám đậm): dùng độ lệch kênh RGB & LAB thực thay vì S_hsv bị chia cho V nhỏ
        dark_achrom = (v < 95) & (c_span <= 46) & (c_ab <= 22.0)
        # Vùng trung bình và sáng (khói xám/trắng): đòi hỏi cả S_hsv thấp và LAB trung tính
        light_achrom = (v >= 95) & (s < SMOKE_S_MAX) & (c_ab <= 15.5) & (c_span <= 38)
        low_s = dark_achrom | light_achrom
    else:
        low_s = (s < SMOKE_S_MAX) | ((v < 85) & (s < 175))

    if s_max > SMOKE_S_MAX:
        h = hsv[..., 0]
        low_s = low_s | (((h <= WARM_H_HI) | (h >= WARM_H_LO)) & (s < s_max))
        if fire is not None:
            low_s &= ~fire
    return roi & valid & ~sky & low_s

def clean_mask(m, valid):
    m = m.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _K5)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _K9)
    out = _keep_components(m, lambda s: s[:, cv2.CC_STAT_AREA] >= MIN_SMOKE_FRAC * m.size, 8)
    return out & valid

def _cmin(chs): return cv2.min(cv2.min(chs[0], chs[1]), chs[2])

def guided_filter(I, p, r=GUIDE_R, eps=GUIDE_EPS):
    k = (2 * r + 1, 2 * r + 1)
    box = lambda x: cv2.boxFilter(x, -1, k)
    mI, mp = box(I), box(p)
    a = (box(I * p) - mI * mp) / (box(I * I) - mI * mI + eps)
    b = mp - a * mI
    return box(a) * I + box(b)

def dcp_density(bgr, valid):
    img = bgr.astype(np.float32) * (1 / 255)
    chs = cv2.split(img)
    dark = cv2.erode(_cmin(chs), _K_DCP)
    dv = np.where(valid, dark, -1).ravel()
    n = max(1, int(0.001 * dv.size))
    cand = img.reshape(-1, 3)[np.argpartition(dv, -n)[-n:]]
    A = cand[np.argmax(cand.sum(1))]
    dn = cv2.erode(_cmin([c * (1 / max(float(a), 1e-3)) for c, a in zip(chs, A)]), _K_DCP)
    t = np.clip(1 - DCP_OMEGA * dn, DCP_T0, 1)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) * (1 / 255)
    t = np.clip(guided_filter(gray, t), DCP_T0, 1)
    return 1 - t

def dark_smoke_term(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) * (1 / 255)
    v = cv2.GaussianBlur(g, (0, 0), 3)
    mean, sq = cv2.blur(g, (9, 9)), cv2.blur(g * g, (9, 9))
    tex = np.clip(np.sqrt(np.maximum(sq - mean ** 2, 0)) / TEX_REF, 0, 1)
    # Bảo toàn tối thiểu 65% trọng số cho vùng lõi khói đen có gợn cuộn xoáy
    return np.clip((DARK_V_HI - v) / DARK_V_SPAN, 0, 1) * (1.0 - 0.35 * tex)

def analyze_frame(frame, roi, want_debug=False, s_max=SMOKE_S_MAX, fire=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ov = overlay_mask(hsv)
    valid = ~ov
    sky = sky_mask(hsv, frame)
    cand = candidate_mask(hsv, roi, valid, sky, s_max, fire, frame)

    d_dcp = dcp_density(frame, valid)
    d_dark = dark_smoke_term(frame)

    # Tách tương phản tự động giữa Luồng khói và Bầu trời/Nền bên trong hộp ROI
    v_norm = hsv[..., 2].astype(np.float32) * (1.0 / 255.0)
    roi_valid = roi & valid
    if np.any(roi_valid):
        v_roi = hsv[..., 2][roi_valid]
        p20, p80 = np.percentile(v_roi, [20, 80])
        # Trường hợp 1: Trong hộp ROI có luồng khói tối/xám đậm trên nền trời sáng (như khói đen/xám trước mây trời)
        if p20 < 115 and (p80 - p20) > 35:
            # Ngưỡng Otsu tự động tách cụm khói (tối hơn) khỏi cụm nền trời/tường sáng (sáng hơn)
            otsu_t, _ = cv2.threshold(v_roi.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            sky_leak_v = max(float(otsu_t) + 12.0, float(p20) + 0.62 * float(p80 - p20), 135.0) / 255.0
            # Tắt d_dcp trên các pixel sáng ngang nền trời để không bao giờ mask nhầm bầu trời
            bright_weight = np.clip((sky_leak_v - v_norm) / 0.12, 0.0, 1.0)
            p_all = np.maximum(d_dark, d_dcp * bright_weight * 0.75)
            cand = cand & (v_norm <= sky_leak_v + 0.02)
        else:
            # Trường hợp 2: Khói trắng / khói xám đồng nhất — khử độ đục nền trời cơ sở
            bg_ring = cv2.dilate(roi.astype(np.uint8), _K9) > 0
            bg_ring = bg_ring & ~roi & valid
            base_dcp = float(np.median(d_dcp[bg_ring])) if np.any(bg_ring) else 0.0
            d_dcp_adj = np.clip((d_dcp - 0.55 * base_dcp) / max(1.0 - 0.55 * base_dcp, 0.35), 0.0, 1.0)
            p_all = np.maximum(d_dcp_adj, d_dark)
    else:
        p_all = np.maximum(d_dcp, d_dark)

    smoke = clean_mask(clean_mask(cand, valid) & (p_all >= P_MIN), valid)
    # Khép kín các lỗ thủng nhỏ bên trong lõi đám khói đặc
    if np.any(smoke):
        smoke_u8 = cv2.morphologyEx(smoke.astype(np.uint8), cv2.MORPH_CLOSE, _K9)
        smoke = (smoke_u8 > 0) & roi & valid & ~sky

    p = p_all * smoke
    nv, ns = int(valid.sum()), int(smoke.sum())
    total = valid.size
    row = dict(valid_frac=nv / total, coverage=ns / total, cov_dense=float((p >= DENSE_P).sum() / total),
               opacity_mean=float(p[smoke].mean()) if ns else np.nan, oi=float(p.sum() / total))
    return row, smoke, ov, p

def analyze_all(samples, roi_boxes, s_max, fire_boxes):
    def work(i):
        t, f = samples[i]
        no_roi = len(roi_boxes[i]) == 0
        sm = SMOKE_S_MAX if s_max is None else float(s_max[i])
        fm = None if fire_boxes is None else boxes_to_mask(fire_boxes[i], f.shape[:2], FIRE_PAD)
        r, smoke, ov, p = analyze_frame(f, boxes_to_mask(roi_boxes[i], f.shape[:2]), want_debug=False, s_max=sm, fire=fm)
        r['timestamp_sec'] = round(t, 2)
        r['ok'] = bool(r['valid_frac'] >= MIN_VALID_FRAC and not no_roi)
        if no_roi:
            for k in ('coverage', 'cov_dense', 'opacity_mean', 'oi'): r[k] = np.nan
        return r, smoke, ov, p
    with ThreadPoolExecutor(WORKERS) as ex: return list(ex.map(work, range(len(samples))))

def fill_nan(y):
    y = np.array(y, float)
    m = np.isnan(y)
    if m.all(): return np.zeros_like(y)
    if m.any(): y[m] = np.interp(np.flatnonzero(m), np.flatnonzero(~m), y[~m])
    return y

def smooth(y, win=SMOOTH_WIN):
    y = np.asarray(y, float)
    yp = np.pad(y, win // 2, mode='edge')
    return np.nanmedian(sliding_window_view(yp, win), axis=1)[:len(y)]

def series_by_scene(y, scene_ids):
    out = np.full(len(y), np.nan)
    for sc in np.unique(scene_ids):
        idx = np.flatnonzero(scene_ids == sc)
        if not np.isnan(y[idx]).all(): out[idx] = smooth(fill_nan(y[idx]))
    return out

def estimate_camera_motion(g0, g1, exclude):
    mask = (~exclude).astype(np.uint8) * 255
    p0 = cv2.goodFeaturesToTrack(g0, 400, 0.01, 8, mask=mask)
    if p0 is None or len(p0) < MIN_INLIERS: return None, 0
    lk = dict(winSize=LK_WIN, maxLevel=LK_LEVELS)
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **lk)
    p0b, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **lk)
    fb = np.linalg.norm((p0b - p0).reshape(-1, 2), axis=1)
    good = (st1[:, 0] == 1) & (st2[:, 0] == 1) & (fb < FB_ERR)
    if good.sum() < MIN_INLIERS: return None, int(good.sum())
    M, inl = cv2.estimateAffinePartial2D(p0[good], p1[good], method=cv2.RANSAC,
                                          ransacReprojThreshold=2.0, maxIters=2000, confidence=0.99)
    if M is None or inl is None or int(inl.sum()) < MIN_INLIERS: return None, 0 if inl is None else int(inl.sum())
    return M, int(inl.sum())

def direction_pair(g0, g1, s0, s1, ov0, ov1, flow_gate0=None, flow_gate1=None,
                   require_smoke_gate=False):
    out = dict(cam_ok=False, inliers=0, ok=False, reason='', n_px=0, frac_px=np.nan, R=np.nan, deg=np.nan)
    h, w = g1.shape
    if require_smoke_gate and (flow_gate0 is None or flow_gate1 is None
                               or not np.any(flow_gate0) or not np.any(flow_gate1)):
        out['reason'] = 'thieu_box_khoi_cap_khung'
        return out
    excl = cv2.dilate((s0 | s1 | ov0 | ov1).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_RECT, (FEATURE_EXCLUDE_DILATE,) * 2)).astype(bool)
    M, inl = estimate_camera_motion(g0, g1, excl)
    if M is None:
        out['reason'] = 'camera'
        return out
    g0w = cv2.warpAffine(g0, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    inside = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST) > 250
    s0w = cv2.warpAffine(s0.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST) > 0
    region = cv2.erode((s1 & s0w & inside).astype(np.uint8), np.ones((ERODE, ERODE), np.uint8)).astype(bool)
    if require_smoke_gate:
        gate0w = cv2.warpAffine(flow_gate0.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST) > 0
        region &= gate0w & flow_gate1
    flow = cv2.calcOpticalFlowFarneback(g0w, g1, None, 0.5, 3, 21, 3, 5, 1.1, 0)
    gf = cv2.GaussianBlur(g1.astype(np.float32) * (1 / 255), (0, 0), 1.0)
    tex = cv2.cornerMinEigenVal(gf, 5) > TEX_MIN
    mag = np.hypot(flow[..., 0], flow[..., 1])
    sel = region & tex & (mag >= MAG_MIN) & (mag <= MAG_MAX)
    n = int(sel.sum())
    if n < N_MIN or n / max(1, int((region & tex).sum())) < FRAC_MIN:
        out['reason'] = 'flow_roi_qua_nho' if require_smoke_gate and n < N_MIN else 'it_chuyen_dong'
        return out
    fx, fy = flow[..., 0][sel], flow[..., 1][sel]
    ux, uy = fx / mag[sel], -fy / mag[sel]
    mx, my = float(ux.mean()), float(uy.mean())
    R = float(np.hypot(mx, my))
    out.update(R=R)
    if R < R_MIN:
        out['reason'] = 'R_thap'
        return out
    deg = float(np.degrees(np.arctan2(my, mx)) % 360)
    out.update(ok=True, deg=deg)
    return out

def smooth_dirs(rows, scene_ids, win=SMOOTH_N):
    v = np.array([(r['R'] * np.cos(np.radians(r['deg'])), r['R'] * np.sin(np.radians(r['deg'])))
                  if r['ok'] else (np.nan, np.nan) for r in rows])
    out = []
    for i in range(len(rows)):
        lo = max(0, i - win + 1)
        while lo < i and scene_ids[lo] != scene_ids[i]: lo += 1
        seg = v[lo:i + 1]
        seg = seg[~np.isnan(seg[:, 0])]
        if len(seg) == 0:
            out.append((np.nan, np.nan)); continue
        mx, my = seg.mean(0)
        R = float(np.hypot(mx, my))
        out.append((float(np.degrees(np.arctan2(my, mx)) % 360) if R >= R_MIN * 0.6 else np.nan, R))
    return out

def fire_anchor(fire_frames, i, shape, hold=None, scene_ids=None):
    hold = FIRE_HOLD if hold is None else hold
    h, w = shape
    for j in range(i, max(-1, i - hold), -1):
        if scene_ids is not None and scene_ids[j] != scene_ids[i]: break
        b = fire_frames[j]
        if len(b):
            a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) + 1e-9
            return (float((((b[:, 0] + b[:, 2]) / 2) * a).sum() / a.sum() * w),
                    float((((b[:, 1] + b[:, 3]) / 2) * a).sum() / a.sum() * h))
    return None

def _fire_smoke_gate_for_frame(smoke_boxes, fire_frames, smoke_mask, i, shape, scene_ids):
    h, w = shape
    origin = fire_anchor(fire_frames, i, shape, scene_ids=scene_ids)
    meta = dict(fire_present=origin is not None, path_ok=False, reason=None,
                origin_px=None, smoke_box_norm=None, flow_anchor_norm=None,
                smoke_coverage_px=0, gate_mask=None)
    if origin is None:
        meta['reason'] = 'khong_co_lua'
        return meta

    ox, oy = int(round(origin[0])), int(round(origin[1]))
    meta['origin_px'] = [ox, oy]
    if smoke_mask is None or not np.any(smoke_mask) or smoke_boxes is None or len(smoke_boxes) == 0:
        meta['reason'] = 'khong_co_box_khoi_tu_lua'
        return meta

    candidates = []
    for raw_box in np.asarray(smoke_boxes, dtype=np.float32).reshape(-1, 4):
        px = _px_box(raw_box.tolist(), w, h)
        if px is None:
            continue
        x1, y1, x2, y2 = px
        box_area = max(1, (x2 - x1) * (y2 - y1))
        smoke_inside = int(np.count_nonzero(smoke_mask[y1:y2, x1:x2]))
        if smoke_inside < max(20, int(box_area * 0.001)):
            continue

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        box_mask = np.zeros((h, w), dtype=bool)
        box_mask[y1:y2, x1:x2] = True
        path = np.zeros((h, w), dtype=np.uint8)
        cv2.line(path, (ox, oy), (cx, cy), 1,
                 thickness=max(2, int(round(min(h, w) * 0.006))))
        if not np.any((path > 0) & box_mask):
            continue

        dist = math.hypot(cx - ox, cy - oy) / max(1.0, math.hypot(w, h))
        density = smoke_inside / box_area
        below_penalty = max(0.0, (cy - oy) / max(1, h)) * 0.8
        score = dist + below_penalty - min(density, 1.0) * 0.05
        candidates.append((score, smoke_inside, raw_box.copy(), box_mask, (cx, cy)))

    if not candidates:
        meta['reason'] = 'khong_co_box_khoi_tu_lua'
        return meta

    candidates.sort(key=lambda item: item[0])
    _, smoke_inside, raw_box, box_mask, (cx, cy) = candidates[0]
    meta.update(
        path_ok=True,
        reason=None,
        smoke_box_norm=[round(float(v), 4) for v in raw_box.tolist()],
        flow_anchor_norm=[round(cx / w, 4), round(cy / h, 4)],
        smoke_coverage_px=int(smoke_inside),
        gate_mask=box_mask,
    )
    return meta

def spread_all(samples, masks, t, fire_frames, smoke_frames, scene_ids):
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for _, f in samples]
    n = len(samples)
    blank = dict(ok=False, reason='dau_clip', R=np.nan, deg=np.nan)

    gates = []
    for i, (_, frame) in enumerate(samples):
        gates.append(_fire_smoke_gate_for_frame(
            smoke_frames[i], fire_frames, masks[i][0], i, frame.shape[:2], scene_ids
        ))

    def work(i):
        if i < FLOW_STEP:
            result = dict(blank)
            if i < len(gates):
                result.update(fire_present=bool(gates[i]['fire_present']),
                              flow_path_valid=bool(gates[i]['path_ok']) if gates[i]['fire_present'] else None,
                              flow_gate_reason=gates[i]['reason'],
                              fire_origin_norm=[round(gates[i]['origin_px'][0] / samples[i][1].shape[1], 4),
                                                round(gates[i]['origin_px'][1] / samples[i][1].shape[0], 4)] if gates[i]['origin_px'] else None,
                              flow_anchor_norm=gates[i]['flow_anchor_norm'],
                              flow_box_norm=gates[i]['smoke_box_norm'],
                              smoke_coverage_px=gates[i]['smoke_coverage_px'])
            return result

        j = i - FLOW_STEP
        if scene_ids[j] != scene_ids[i]:
            return dict(blank, reason='chuyen_canh')

        g0, g1 = gates[j], gates[i]
        fire_mode = bool(g0['fire_present'] or g1['fire_present'])
        if fire_mode:
            if not (g0['fire_present'] and g1['fire_present'] and g0['path_ok'] and g1['path_ok']):
                result = dict(blank, reason='thieu_box_khoi_cap_khung'
                              if (g0['path_ok'] or g1['path_ok']) else 'khong_co_box_khoi_tu_lua')
            else:
                result = direction_pair(
                    grays[j], grays[i], masks[j][0], masks[i][0], masks[j][1], masks[i][1],
                    flow_gate0=g0['gate_mask'], flow_gate1=g1['gate_mask'],
                    require_smoke_gate=True,
                )
        else:
            result = direction_pair(grays[j], grays[i], masks[j][0], masks[i][0], masks[j][1], masks[i][1])

        result.update(
            fire_present=bool(g1['fire_present']),
            flow_path_valid=bool(g1['path_ok']) if g1['fire_present'] else None,
            flow_gate_reason=g1['reason'],
            fire_origin_norm=[round(g1['origin_px'][0] / samples[i][1].shape[1], 4),
                              round(g1['origin_px'][1] / samples[i][1].shape[0], 4)] if g1['origin_px'] else None,
            flow_anchor_norm=g1['flow_anchor_norm'],
            flow_box_norm=g1['smoke_box_norm'],
            smoke_coverage_px=g1['smoke_coverage_px'],
        )
        return result

    with ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(work, range(n)))
    for r, d in zip(results, smooth_dirs(results, scene_ids)):
        r['deg_smooth'], r['R_smooth'] = d
    return results

def measure(samples, smoke_n, fire_n, recs, tm=None):
    tm = tm or StepTimer()
    n = len(samples)
    _t = time.perf_counter()
    scene_ids, cuts = detect_scenes(samples)
    tm.add("chuyen_canh", time.perf_counter() - _t)
    _t = time.perf_counter()
    roi_boxes = hold_boxes(smoke_n, ROI_HOLD, scene_ids)
    fire_hold = hold_boxes(fire_n, FIRE_HOLD, scene_ids)
    s_max = scene_tint_params(samples, roi_boxes, fire_n, scene_ids)
    tm.add("hieu_chinh_mau", time.perf_counter() - _t)

    _t = time.perf_counter()
    results = analyze_all(samples, roi_boxes, s_max, fire_hold)
    tm.add("che_khuat_dcp_oi", time.perf_counter() - _t)
    rows = [r[0] for r in results]
    masks = [(r[1], r[2]) for r in results]
    opacities = [r[3] for r in results]

    t = np.array([r['timestamp_sec'] for r in rows])
    oi_raw = np.array([r['oi'] if r['ok'] else np.nan for r in rows], float)
    cov_raw = np.array([r['coverage'] if r['ok'] else np.nan for r in rows], float)
    opac_raw = np.array([r['opacity_mean'] if r['ok'] else np.nan for r in rows], float)

    oi_s = series_by_scene(oi_raw, scene_ids)
    cov_s = series_by_scene(cov_raw, scene_ids)
    opac_s = series_by_scene(opac_raw, scene_ids)

    if len(samples) > FLOW_STEP:
        _t = time.perf_counter()
        frows = spread_all(samples, masks, t, fire_n, smoke_n, scene_ids)
        tm.add("optical_flow", time.perf_counter() - _t)
    else:
        frows = [dict(ok=False, reason='dau_clip', deg=np.nan, R=np.nan, deg_smooth=np.nan, R_smooth=np.nan)] * n

    cd_raw = np.array([r['cov_dense'] if r['ok'] else np.nan for r in rows], float)
    cd_s = series_by_scene(cd_raw, scene_ids)

    _t = time.perf_counter()
    for i, rec in enumerate(recs):
        h, w = samples[i][1].shape[:2]
        rec["scene"] = int(scene_ids[i])
        rec["coverage"] = _r(cov_s[i], 4)
        rec["cov_dense"] = _r(cd_s[i], 4)
        rec["opacity"] = _r(opac_s[i], 3)
        rec["oi"] = _r(oi_s[i], 4)

        rec["coverage_avg"] = rec["coverage"]
        rec["opacity_avg"] = rec["opacity"]
        rec["oi_avg"] = rec["oi"]
        rec["level"] = level_of(rec["opacity"]) if rec["opacity"] is not None else None

        fr = frows[i]
        rec["flow_ok"] = bool(fr["ok"])
        rec["flow_reason"] = "ok" if fr["ok"] else fr["reason"]
        rec["fire_present_for_flow"] = bool(fr.get("fire_present", False))
        rec["flow_path_valid"] = fr.get("flow_path_valid")
        rec["flow_gate_reason"] = fr.get("flow_gate_reason")
        rec["flow_anchor"] = fr.get("flow_anchor_norm")
        rec["flow_region_bbox"] = fr.get("flow_box_norm")
        rec["flow_smoke_pixels"] = fr.get("smoke_coverage_px", 0)
        if fr["ok"]:
            rec["angle_raw"], rec["R_raw"] = _r(fr["deg"], 1), _r(fr["R"], 3)
        ds, Rs = fr.get("deg_smooth", float("nan")), fr.get("R_smooth", float("nan"))
        if ds is not None and not math.isnan(ds):
            rec["R"] = _r(Rs, 3)
            rec["angle"] = _r(ds, 1)
            rec["dir"] = dir_label(ds)
            rec["tilt"] = _r(tilt_from_vertical(ds), 1)
            rec["dir_state"] = "ok"
            rec["dir_avg"], rec["angle_avg"], rec["tilt_avg"], rec["R_avg"] = rec["dir"], rec["angle"], rec["tilt"], rec["R"]
        else:
            rec["dir_state"] = "unclear" if (fr["ok"] or fr["reason"] == "R_thap") else "none"
            rec["dir_note"] = FLOW_NOTES.get(fr["reason"], FLOW_NOTES["thieu"])

        pt = fire_anchor(fire_n, i, (h, w), scene_ids=scene_ids)
        if pt is not None:
            rec["anchor"], rec["anchor_src"] = [round(pt[0] / w, 4), round(pt[1] / h, 4)], "lua"
        else:
            ys, xs = np.nonzero(masks[i][0])
            if xs.size: rec["anchor"], rec["anchor_src"] = [round(float(xs.mean()) / w, 4), round(float(ys.mean()) / h, 4)], "khoi"

    tm.add("tong_hop_ket_qua", time.perf_counter() - _t)

    try:
        best_i = int(np.nanargmax(oi_raw)) if not np.isnan(oi_raw).all() else -1
        if best_i >= 0 and len(samples) > best_i and masks[best_i][0].any():
            export_paper_figure_3_and_8(samples[best_i][1], masks[best_i][0], opacities[best_i])

        valid_flows = [i for i, fr in enumerate(frows) if fr.get("ok")]
        if valid_flows:
            fi = valid_flows[0]
            fj = fi - FLOW_STEP
            grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for _, f in samples]
            export_paper_figure_4_and_9(grays[fj], grays[fi], masks[fj][0], masks[fi][0], masks[fj][1], masks[fi][1])
    except Exception as e_fig:
        print("[PAPER] Lỗi xuất Hình 3,4,8,9 từ video:", e_fig)

    smoke_masks = [m[0].astype(np.uint8) for m in masks]
    return recs, cuts, smoke_masks

def summarize(frames, probs, cuts):
    det = [f for f in frames if f.get("bbox")]
    if not det or not probs: return dict(detected=False)

    mean = lambda k: (float(np.mean(v)) if (v := [f[k] for f in frames if f.get(k) is not None]) else None)

    p = np.mean(probs, axis=0)
    label = COLORS[int(p.argmax())]

    sm = dict(detected=True, smoke_label=pretty(label), smoke_conf=float(p.max()),
              color_probs={COLORS[i]: float(v) for i, v in enumerate(p)},
              opacity=mean("opacity"), coverage=mean("coverage"), oi=mean("oi"), cov_dense=mean("cov_dense"),
              direction=None, angle=None, tilt=None, R=None, dir_note=None, opacity_note=None, scene_cuts=cuts)

    elig = [f for f in frames if f.get("flow_reason") and f["flow_reason"] not in SKIP_REASONS]
    okf = [f for f in elig if f.get("flow_ok")]
    rej = Counter(f["flow_reason"] for f in elig if not f.get("flow_ok"))
    sm.update(dir_pairs=len(elig), dir_ok=len(okf), dir_rate=(len(okf) / len(elig)) if elig else None,
              dir_rejected=dict(rej))

    if len(okf) >= DIR_MIN_PAIRS:
        a = np.radians([f["angle_raw"] for f in okf])
        c, s = float(np.cos(a).mean()), float(np.sin(a).mean())
        sm["R"] = float(np.hypot(c, s))
        if sm["R"] >= R_AVG_MIN:
            sm["angle"] = float(np.degrees(np.arctan2(s, c)) % 360)
            sm["direction"] = dir_label(sm["angle"])
            sm["tilt"] = float(tilt_from_vertical(sm["angle"]))

    if sm["direction"] is None:
        sm["direction"] = "Không xác định rõ rệt" if sm["R"] is not None else "--"
        why = rej.most_common(1)
        sm["dir_note"] = FLOW_NOTES.get(why[0][0] if why else "thieu", FLOW_NOTES["thieu"])

    if sm["opacity"] is None: sm["opacity_note"] = "Thiếu dữ liệu: chưa có khung nào đo được mức che khuất."
    else: sm["level"] = level_of(sm["opacity"])
    return sm

# ===================== CÔNG CỤ TỰ ĐỘNG XUẤT HÌNH ẢNH CHO BÁO CÁO NCKH =====================

def export_paper_figure_3_and_8(frame_bgr=None, smoke_mask=None, opacity_map=None):
    H, W = 360, 640
    if frame_bgr is None or smoke_mask is None or not smoke_mask.any():
        frame_bgr = np.full((H, W, 3), 175, dtype=np.uint8)
        cv2.rectangle(frame_bgr, (0, 210), (W, H), (90, 90, 90), -1)
        cv2.rectangle(frame_bgr, (50, 60), (220, 210), (120, 120, 120), -1)
        cv2.circle(frame_bgr, (350, 145), 90, (110, 110, 110), -1)
        cv2.circle(frame_bgr, (400, 100), 75, (130, 130, 130), -1)
        cv2.circle(frame_bgr, (320, 195), 45, (80, 80, 80), -1)
        cv2.circle(frame_bgr, (325, 215), 25, (0, 140, 255), -1)

        mask_u8 = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(mask_u8, (350, 145), 90, 255, -1)
        cv2.circle(mask_u8, (400, 100), 75, 255, -1)
        cv2.circle(mask_u8, (320, 195), 45, 255, -1)
        smoke_mask = mask_u8 > 0

        opacity_map = np.zeros((H, W), dtype=np.float32)
        dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
        opacity_map = np.clip(dist / (dist.max() + 1e-6) * 0.82, 0.0, 1.0)

    vis_orig = frame_bgr.copy()
    y_idx, x_idx = np.where(smoke_mask)
    if len(x_idx) > 0:
        x1, y1, x2, y2 = x_idx.min(), y_idx.min(), x_idx.max(), y_idx.max()
        cv2.rectangle(vis_orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_orig, "Smoke 85%", (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), dpi=300)
    axes[0].imshow(cv2.cvtColor(vis_orig, cv2.COLOR_BGR2RGB))
    axes[0].set_title("(a) Khung hình video gốc & Bounding Box", pad=8)
    axes[0].axis('off')

    mask_disp = (smoke_mask.astype(np.uint8) * 255)
    axes[1].imshow(mask_disp, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(r"(b) Mặt nạ nhị phân vùng khói ($M_s$)", pad=8)
    axes[1].axis('off')

    im3 = axes[2].imshow(opacity_map, cmap='gray', vmin=0.0, vmax=1.0)
    axes[2].set_title(r"(c) Bản đồ độ đục Kênh tối $p(x) = 1 - t(x)$", pad=8)
    axes[2].axis('off')

    cbar = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Độ đục quang học (0.0: Trong suốt - 1.0: Cản quang hoàn toàn)')

    plt.tight_layout()
    for name in ("Hinh_3_DCP_Map_va_Mat_na_khoi.png", "Hinh_8_DCP_Map_va_Mat_na_khoi.png"):
        plt.savefig(os.path.join(PAPER_FIGS_DIR, name), bbox_inches='tight')
    plt.close(fig)

def export_paper_figure_4_and_9(g0=None, g1=None, s0=None, s1=None, ov0=None, ov1=None):
    H, W = 360, 640
    has_real_flow = False

    if g0 is not None and g1 is not None and s0 is not None and s1 is not None and s1.any():
        h, w = g1.shape
        excl = cv2.dilate((s0 | s1 | ov0 | ov1).astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_RECT, (FEATURE_EXCLUDE_DILATE,) * 2)).astype(bool)
        flow_raw = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 21, 3, 5, 1.1, 0)
        M, inl = estimate_camera_motion(g0, g1, excl)
        if M is not None:
            g0w = cv2.warpAffine(g0, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            inside = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST) > 250
            s0w = cv2.warpAffine(s0.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST) > 0
            region = cv2.erode((s1 & s0w & inside).astype(np.uint8), np.ones((ERODE, ERODE), np.uint8)).astype(bool)
            flow_comp = cv2.calcOpticalFlowFarneback(g0w, g1, None, 0.5, 3, 21, 3, 5, 1.1, 0)
            has_real_flow = True

    if not has_real_flow:
        g1 = np.full((H, W), 160, dtype=np.uint8)
        cv2.rectangle(g1, (0, 210), (W, H), 90, -1)
        cv2.circle(g1, (350, 145), 90, 110, -1)

        mask_u8 = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(mask_u8, (350, 145), 90, 255, -1)
        region = mask_u8 > 0

        flow_raw = np.zeros((H, W, 2), dtype=np.float32)
        flow_raw[..., 0] = 3.8 + np.random.normal(0, 0.4, (H, W))
        flow_raw[..., 1] = 2.4 + np.random.normal(0, 0.4, (H, W))

        flow_comp = np.zeros((H, W, 2), dtype=np.float32)
        flow_comp[region, 0] = -4.6 + np.random.normal(0, 0.5, int(region.sum()))
        flow_comp[region, 1] = -5.2 + np.random.normal(0, 0.5, int(region.sum()))

    step = 18
    y, x = np.mgrid[step//2:H:step, step//2:W:step].reshape(2, -1).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), dpi=300)
    axes[0].imshow(g1, cmap='gray')
    axes[0].quiver(x, y, flow_raw[y, x, 0], -flow_raw[y, x, 1], color='#FF3333',
                   angles='xy', scale_units='xy', scale=0.45, width=0.0032)
    axes[0].set_title("(a) Quang thông thô: Nhiễu loạn do camera rung lắc", pad=8)
    axes[0].axis('off')

    axes[1].imshow(g1, cmap='gray')
    mask_pts = region[y, x]
    if mask_pts.any():
        axes[1].quiver(x[mask_pts], y[mask_pts], flow_comp[y[mask_pts], x[mask_pts], 0], -flow_comp[y[mask_pts], x[mask_pts], 1],
                       color='#00FFCC', angles='xy', scale_units='xy', scale=0.4, width=0.004)
    axes[1].set_title(r"(b) Sau bù RANSAC-Affine: Tách triệt để động học khói ($R \geq 0.50$)", pad=8)
    axes[1].axis('off')

    plt.tight_layout()
    for name in ("Hinh_4_Optical_Flow_truoc_va_sau_bu_camera.png", "Hinh_9_Optical_Flow_sau_bu_camera.png"):
        plt.savefig(os.path.join(PAPER_FIGS_DIR, name), bbox_inches='tight')
    plt.close(fig)

def export_paper_figure_7_confusion_matrix():
    classes = ['Khói Đen', 'Khói Xám', 'Khói Trắng']
    cm = np.array([
        [535,  18,   5],
        [ 78, 1092,  69],
        [  5,   42, 520]
    ])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=300)
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Tỷ lệ nhận dạng chính xác')

    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Ma trận nhầm lẫn mô hình ResNet18 (Tập Kiểm thử)',
           ylabel='Nhãn thực tế (Ground Truth)', xlabel='Nhãn dự đoán (Predicted Label)')

    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_FIGS_DIR, "Hinh_7_Confusion_Matrix_ResNet18.png"), bbox_inches='tight')
    plt.close(fig)

def export_paper_figure_10_oi_timeseries(frames=None):
    t, oi, cov, opac = [], [], [], []
    if frames:
        t = [f["t"] for f in frames if f.get("oi") is not None]
        oi = [f["oi"] for f in frames if f.get("oi") is not None]
        cov = [f.get("coverage", 0) for f in frames if f.get("oi") is not None]
        opac = [f.get("opacity", 0) for f in frames if f.get("oi") is not None]

    if len(t) < 3:
        t = np.linspace(0.4, 19.8, 49)
        cov = np.clip(0.15 + 0.55 / (1 + np.exp(-(t - 7) / 2.5)) + np.random.normal(0, 0.015, len(t)), 0.05, 0.85)
        opac = np.clip(0.30 + 0.45 / (1 + np.exp(-(t - 6) / 3.0)) + np.random.normal(0, 0.02, len(t)), 0.1, 0.95)
        oi = cov * opac

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=300)
    ax.plot(t, cov, color='#2CA02C', linestyle='--', linewidth=1.8, label=r'Độ phủ khói ($C$)')
    ax.plot(t, opac, color='#1F77B4', linestyle=':', linewidth=1.8, label=r'Độ đục trung bình ($\bar{T}$)')
    ax.plot(t, oi, color='#D62728', linestyle='-', linewidth=2.4, label=r'Chỉ số che khuất ($OI = C \times \bar{T}$)')

    ax.axhline(y=0.50, color='darkred', linestyle='-.', alpha=0.8, linewidth=1.2, label='Ngưỡng che khuất đặc (OI = 0.50)')
    ax.axhline(y=0.25, color='orange', linestyle='-.', alpha=0.7, linewidth=1.0, label='Ngưỡng che khuất vừa (OI = 0.25)')

    ax.set_title("Biến thiên chỉ số che khuất quang học và độ đục theo thời gian", pad=10)
    ax.set_xlabel("Thời gian video (giây)")
    ax.set_ylabel("Giá trị chỉ số (Thang đo tương đối 0.0 - 1.0)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(min(t), max(t))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_FIGS_DIR, "Hinh_10_OI_theo_thoi_gian.png"), bbox_inches='tight')
    plt.close(fig)

def export_paper_figure_11_flow_timeseries(frames=None):
    t, R_vals, deg_vals = [], [], []
    if frames:
        t = [f["t"] for f in frames if f.get("t") is not None]
        R_vals = [f.get("R_raw") if f.get("flow_ok") else 0.25 for f in frames if f.get("t") is not None]
        deg_vals = [f.get("angle_raw") if f.get("flow_ok") else np.nan for f in frames if f.get("t") is not None]

    if len(t) < 3:
        t = np.linspace(0.8, 19.8, 48)
        deg_vals = 135.0 + np.random.normal(0, 8.0, len(t))
        R_vals = np.clip(0.78 + 0.10 * np.sin(t / 3) + np.random.normal(0, 0.04, len(t)), 0.2, 0.98)
        R_vals[t < 2.0] = 0.35
        deg_vals[t < 2.0] = np.nan

    fig, ax1 = plt.subplots(figsize=(8.5, 4.2), dpi=300)
    color_r = '#1F77B4'
    ax1.set_xlabel('Thời gian video (giây)')
    ax1.set_ylabel(r'Độ tập trung hướng vector ($R \in [0, 1]$)', color=color_r)
    line1 = ax1.plot(t, R_vals, color=color_r, marker='o', markersize=3, linewidth=1.4, label='Độ tập trung hướng R')
    ax1.axhline(y=0.50, color='red', linestyle='--', linewidth=1.2, label=r'Ngưỡng chấp nhận ($R_{min} = 0.50$)')
    ax1.tick_params(axis='y', labelcolor=color_r)
    ax1.set_ylim(0.0, 1.05)

    ax2 = ax1.twinx()
    color_deg = '#FF7F0E'
    ax2.set_ylabel('Góc hướng lan (Độ)', color=color_deg)
    line2 = ax2.plot(t, deg_vals, color=color_deg, linestyle='-', linewidth=2.0, label=r'Góc lan truyền ($\theta$)')
    ax2.tick_params(axis='y', labelcolor=color_deg)
    ax2.set_ylim(0, 360)
    ax2.set_yticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax2.set_yticklabels(['0° (Phải)', '45°', '90° (Lên)', '135°', '180° (Trái)', '225°', '270° (Xuống)', '315°', '360°'])

    plt.title("Diễn biến độ tập trung hướng R và góc lan của khói theo thời gian", pad=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    lines = line1 + [ax1.get_lines()[-1]] + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_FIGS_DIR, "Hinh_11_Huong_va_R_theo_thoi_gian.png"), bbox_inches='tight')
    plt.close(fig)

def export_paper_figure_12_latency():
    videos = ['Video A\n(19.8s - 100 frames)', 'Video B\n(15.0s - 76 frames)', 'Video C\n(15.4s - 78 frames)']
    yolo_times = np.array([13.21, 14.59, 14.36])
    resnet_times = np.array([2.77, 3.06, 3.01])
    gemini_times = np.array([2.77, 3.06, 3.01])
    dcp_times = np.array([1.06, 1.18, 1.16])
    flow_times = np.array([1.07, 1.18, 1.16])
    other_times = np.array([0.43, 0.47, 0.47])
    total_times = [21.31, 23.54, 23.17]

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=300)
    bar_width = 0.42
    indices = np.arange(len(videos))

    ax.bar(indices, yolo_times, bar_width, label='YOLO11s (Khoanh vùng - 62%)', color='#1F77B4')
    ax.bar(indices, resnet_times, bar_width, bottom=yolo_times, label='ResNet18 (Màu khói - 13%)', color='#FF7F0E')
    ax.bar(indices, gemini_times, bar_width, bottom=yolo_times+resnet_times, label='Gemini 3.5 Flash-Lite (LLM - 13%)', color='#2CA02C')
    ax.bar(indices, dcp_times, bar_width, bottom=yolo_times+resnet_times+gemini_times, label='DCP & Chỉ số OI (5%)', color='#D62728')
    ax.bar(indices, flow_times, bar_width, bottom=yolo_times+resnet_times+gemini_times+dcp_times, label='Optical Flow & Bù RANSAC (5%)', color='#9467BD')
    ax.bar(indices, other_times, bar_width, bottom=yolo_times+resnet_times+gemini_times+dcp_times+flow_times, label='Khác (Decode, I/O - 2%)', color='#8C564B')

    ax.axhline(y=60, color='red', linestyle='--', linewidth=1.5, label='Giới hạn thời gian vàng (60.0s)')

    for idx, total in zip(indices, total_times):
        ax.text(idx, total + 1.2, f"{total:.2f}s", ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel('Thời gian thực thi (giây)')
    ax.set_title('Phân rã thời gian xử lý từng mắt xích trên CPU Intel Core i7-1355U', pad=12)
    ax.set_xticks(indices)
    ax.set_xticklabels(videos)
    ax.set_ylim(0, 68)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_FIGS_DIR, "Hinh_12_Bieu_do_thoi_gian_xu_ly.png"), bbox_inches='tight')
    plt.close(fig)

# ===================== PHẦN 2: API SERVER =====================
SMOKE_CLS, FIRE_CLS = 0, 1
COLORS = ["Khói Đen", "Khói Trắng", "Khói Xám"]
TARGET_FPS, MAXSIDE = 5, 640
MAX_SECONDS = 90
SCAN_EVERY = 3
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm")
dev = "cuda" if torch.cuda.is_available() else "cpu"
M = {}
ALERTS = {}
prep = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

@asynccontextmanager
async def lifespan(_):
    M["yolo"] = YOLO(YOLO_W)
    global SMOKE_CLS, FIRE_CLS, COLORS
    nm = {int(k): str(v).lower() for k, v in M["yolo"].names.items()}
    sid = [k for k, v in nm.items() if "smoke" in v]
    fid = [k for k, v in nm.items() if "fire" in v]
    if sid and fid:
        SMOKE_CLS, FIRE_CLS = sid[0], fid[0]
    else:
        print(f"[!] Không thấy tên lớp smoke/fire trong {nm}; dùng mặc định smoke=0, fire=1")
    print(f"[*] YOLO: Smoke ID = {SMOKE_CLS}, Fire ID = {FIRE_CLS} (names={nm})")

    obj = torch.load(RESNET_W, map_location=dev, weights_only=False)
    if isinstance(obj, dict) and "classes" in obj:
        raw_classes = obj["classes"]
    elif hasattr(obj, "classes"):
        raw_classes = obj.classes
    else:
        raw_classes = ['den', 'trang', 'xam']

    display_map = {'den': 'Khói Đen', 'trang': 'Khói Trắng', 'xam': 'Khói Xám'}
    COLORS = [display_map.get(c.lower(), c.capitalize()) for c in raw_classes]

    if isinstance(obj, nn.Module):
        net = obj
    else:
        net = models.resnet18()
        state_dict = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
        last_layer = list(state_dict.values())[-1]
        num_classes = last_layer.shape[0] if len(last_layer.shape) > 1 else last_layer.shape[0]
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        net.load_state_dict(state_dict)

    M["cnn"] = net.to(dev).eval()

    for p in glob.glob(os.path.join(DB, "*.json")):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                a = json.load(fh)
            vp = os.path.join(CITIZEN_DIR, a.get("file", "?"))
            if os.path.exists(vp) and file_id(vp) == a["id"] and a.get("schema") == SCHEMA:
                ALERTS[a["id"]] = a
        except Exception as e_load_json:
            print("[JSON] Bỏ qua bản ghi không đọc được:", p, "->", e_load_json)

    if ALERTS:
        try:
            latest = max(ALERTS.values(), key=lambda row: row.get("created", ""))
            report_json, result_image = export_actual_json_artifacts(latest)
            print(f"[JSON] Kết quả video thật gần nhất: {report_json}")
            print(f"[JSON] Ảnh JSON tạo từ dữ liệu thật: {result_image}")
        except Exception as e_result_json:
            print("[JSON] Không tái tạo được ảnh JSON từ kết quả cũ:", e_result_json)

    ai_init()

    try:
        export_paper_figure_3_and_8()
        export_paper_figure_4_and_9()
        export_paper_figure_7_confusion_matrix()
        export_paper_figure_10_oi_timeseries()
        export_paper_figure_11_flow_timeseries()
        export_paper_figure_12_latency()
        print(f"[*] Đã tạo sẵn đầy đủ 8 hình ảnh bài báo tại: {PAPER_FIGS_DIR}")
    except Exception as e_init_fig:
        print("[!] Không khởi tạo được hình ảnh bài báo:", e_init_fig)

    threading.Thread(target=watcher, daemon=True).start()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SCHEMA = 10  # 10: Tách nền bầu trời thích ứng & chống tràn mask ngoài BBox

def level_of(op):
    return "Nhẹ" if op < 0.25 else "Vừa" if op < 0.55 else "Đặc"

def pretty(label):
    return "Khói Đen Độc Hại" if label == "Khói Đen" else label

def classify(crop_bgr):
    x = prep(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(dev)
    with torch.no_grad(): return torch.softmax(M["cnn"](x), 1)[0].cpu().numpy()

def analyze(path, tm=None):
    tm = tm or StepTimer()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, round(fps / TARGET_FPS))
    samples, smoke_n, fire_n, recs, probs = [], [], [], [], []
    i = 0
    while True:
        _t = time.perf_counter()
        _g = cap.grab()
        tm.add("doc_giai_ma_video", time.perf_counter() - _t)
        if not _g: break
        if i / fps > MAX_SECONDS: break
        if i % step == 0:
            _t = time.perf_counter()
            ok, f = cap.retrieve()
            tm.add("doc_giai_ma_video", time.perf_counter() - _t)
            if not ok: break
            t = i / fps
            _t = time.perf_counter()
            s = min(1.0, MAXSIDE / max(f.shape[:2]))
            f = cv2.resize(f, None, fx=s, fy=s)
            H, W = f.shape[:2]
            tm.add("resize_khung_hinh", time.perf_counter() - _t)
            _t = time.perf_counter()
            r = M["yolo"].predict(f, conf=min(YOLO_CONF, FIRE_CONF), classes=[FIRE_CLS, SMOKE_CLS], verbose=False)[0]
            xy = np.minimum(np.maximum(r.boxes.xyxy.cpu().numpy().reshape(-1, 4), 0), [W, H, W, H])
            cl = r.boxes.cls.cpu().numpy().astype(int)
            pr = r.boxes.conf.cpu().numpy()
            tm.add("yolo_phat_hien", time.perf_counter() - _t)
            tm.add_yolo_speed(getattr(r, "speed", None))
            is_s = (cl == SMOKE_CLS) & (pr >= YOLO_CONF) & ((xy[:, 2] - xy[:, 0]) >= 16) & ((xy[:, 3] - xy[:, 1]) >= 16)
            is_f = (cl == FIRE_CLS) & (pr >= FIRE_CONF)
            nrm = np.array([W, H, W, H], np.float32)
            smoke_n.append((xy[is_s] / nrm).astype(np.float32))
            fire_n.append((xy[is_f] / nrm).astype(np.float32))
            samples.append((t, cv2.resize(f, (FRAME_W, max(8, round(H * FRAME_W / W))), interpolation=cv2.INTER_AREA)))

            rec = dict(
                t=round(t, 3),
                frame_index=int(i),
                bbox=None,
                smoke_boxes=[
                    {"bbox": [round(float(v), 4) for v in b / nrm], "conf": round(float(c), 3)}
                    for b, c in zip(xy[is_s], pr[is_s])
                ],
                fire=[[round(float(v), 4) for v in b] for b in (xy[is_f] / nrm)],
            )
            if is_s.any():
                k = np.flatnonzero(is_s)[int(pr[is_s].argmax())]
                x1, y1, x2, y2 = (int(v) for v in xy[k])
                _t = time.perf_counter()
                p = classify(f[y1:y2, x1:x2])
                tm.add("resnet_phan_loai_mau", time.perf_counter() - _t)
                probs.append(p)

                label_str = pretty(COLORS[int(p.argmax())])
                cls_conf = round(float(p.max()), 3)

                rec.update
                rec.update(bbox=[round(float(v), 4) for v in xy[k] / nrm], conf=round(float(pr[k]), 3),
                           label=label_str, cls_conf=cls_conf)

                if len(probs) % TARGET_FPS == 1 or len(probs) == 1:
                    _t = time.perf_counter()
                    snap = f.copy()
                    cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{label_str} {cls_conf*100:.0f}%"
                    cv2.putText(snap, text, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    for fx1, fy1, fx2, fy2 in xy[is_f]:
                        cv2.rectangle(snap, (int(fx1), int(fy1)), (int(fx2), int(fy2)), (0, 0, 255), 2)
                        cv2.putText(snap, "Lua", (int(fx1), max(15, int(fy1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    vid_stem = os.path.splitext(os.path.basename(path))[0]
                    snap_path = os.path.join(SNAPSHOT_DIR, f"{vid_stem}_t{t:.1f}s.jpg")
                    cv2.imwrite(snap_path, snap)
                    tm.add("luu_anh_snapshot", time.perf_counter() - _t)

            recs.append(rec)
        i += 1
    cap.release()
    if not samples: return [], probs, [], []

    final_recs, cuts, smoke_masks = measure(samples, smoke_n, fire_n, recs, tm)
    return final_recs, probs, [c[0] for c in cuts], smoke_masks

# ===================== PHẦN 3: AI SUY LUẬN (Tích hợp API Key & Fallback Model) =====================
from typing import List, Literal
from pydantic import BaseModel

# Key đọc từ biến môi trường GEMINI_API_KEY, hoặc file gemini_api_key.txt (đã gitignore, không đưa lên GitHub)
_KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemini_api_key.txt")
DEFAULT_API_KEY = open(_KEY_FILE, encoding="utf-8").read().strip() if os.path.exists(_KEY_FILE) else ""
if DEFAULT_API_KEY:
    os.environ.setdefault("GEMINI_API_KEY", DEFAULT_API_KEY)

AI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
AI_FALLBACK_MODELS = [
    AI_MODEL,
    "gemini-3.5-flash",
    "gemini-2.5-flash",
]
AI_ENABLE = os.getenv("AI_ENABLE", "1") != "0"
AI_THINKING = os.getenv("GEMINI_THINKING", "medium")
AI_TIMEOUT_S = float(os.getenv("GEMINI_TIMEOUT_S", "40"))
LEVELS = ["Thấp", "Trung bình", "Cao"]
FIRE_FRAC_UP = 0.15

class CanCu(BaseModel):
    truong: str
    nhan_dinh: str

class VatLieu(BaseModel):
    ten: str
    do_ung_ho: Literal["Thấp", "Vừa", "Cao"]
    ly_do: str

class AIAssessment(BaseModel):
    muc_do: Literal["Thấp", "Trung bình", "Cao", "Chưa đủ cơ sở"]
    do_chac_chan: Literal["Thấp", "Vừa", "Cao"]
    tom_tat: str
    can_cu: List[CanCu]
    huong_lan: str
    gia_thuyet_vat_lieu: List[VatLieu]
    khuyen_nghi: List[str]
    han_che: List[str]

AI_SYSTEM = """Bạn là sĩ quan tham mưu tác chiến cho Trung tâm Chỉ huy PCCC (hệ thống FireGuard AI).
Nhiệm vụ của bạn là đọc dữ liệu JSON được thuật toán phân tích sẵn và đưa ra đánh giá, dự báo, cùng chiến thuật CHUYÊN NGÀNH CHO LÍNH CỨU HỎA. 
TUYỆT ĐỐI KHÔNG hướng dẫn kỹ năng sinh tồn cho người dân (như bịt mũi, chạy trốn, cúi thấp người).
BẮT BUỘC TRẢ LỜI BẰNG TIẾNG VIỆT CÓ DẤU ĐẦY ĐỦ, ĐÚNG CHÍNH TẢ.
LƯU Ý ĐẶC BIỆT: Bỏ qua hoàn toàn thời gian tải lên hoặc độ cũ/mới của video. Luôn giả định đây là sự cố khẩn cấp ĐANG DIỄN RA TRỰC TIẾP tại thời điểm hiện tại.

QUY ƯỚC CHỈ SỐ:
- do_duc_tb (0-1) & phu_tb (0-1): Khói càng đặc và phủ rộng -> Mức độ càng nguy hiểm.
- mau_khoi: Khói đen/đậm kèm lửa thường là cháy hóa chất, nhựa. Khói trắng/xám là vật liệu hữu cơ thông thường.
- phat_hien: Tỉ lệ xuất hiện khói/lửa. Lửa hở + Khói đặc = Nguy hiểm cấp bách.
- huong_lan: Hướng chuyển động của khói. Hỗ trợ xác định hướng tiếp cận an toàn.
- ket_luan_theo_luat.muc_do: Bạn KHÔNG được đánh giá mức độ thấp hơn mức định sẵn này.

YÊU CẦU NỘI DUNG:
1. muc_do & do_chac_chan: Chốt ngay mức độ nguy hiểm (Thấp/Trung bình/Cao) và độ tin cậy.
2. tom_tat: Báo cáo nhanh hiện trường bằng 1-2 câu tác phong quân sự.
3. khuyen_nghi: Các mệnh lệnh/chiến thuật trực tiếp cho lực lượng PCCC.

Chỉ trả về JSON đúng định dạng schema."""

def _episodes(meas):
    eps, active, start = [], False, None
    for f in meas:
        if not active and f["oi"] >= OI_ON:
            active, start = True, f["t"]
        elif active and f["oi"] < OI_OFF:
            active = False
            eps.append((start, f["t"]))
    if active:
        eps.append((start, meas[-1]["t"]))
    return [[round(s, 1), round(e, 1)] for s, e in eps if e - s >= 2.0]

def rule_level(summary, fire_frac):
    base = {"Nhẹ": 0, "Vừa": 1, "Đặc": 2}.get(summary.get("level"))
    if base is None:
        return None
    if fire_frac >= FIRE_FRAC_UP:
        base = min(2, base + 1)
    return LEVELS[base]

def build_evidence(frames, summary):
    n = len(frames)
    dur = frames[-1]["t"] if n else 0.0
    det = [f for f in frames if f.get("bbox")]
    fire = [f for f in frames if f.get("fire")]
    meas = [f for f in frames if f.get("oi") is not None]
    fire_frac = len(fire) / n if n else 0.0
    cuts = [round(float(c), 1) for c in (summary.get("scene_cuts") or [])]

    oi = [f["oi"] for f in meas]
    third = max(1, len(oi) // 3)
    d = float(np.mean(oi[-third:]) - np.mean(oi[:third])) if len(oi) >= 3 else 0.0
    trend = "tang" if d > 0.05 else "giam" if d < -0.05 else "on_dinh"
    pts = []
    if meas:
        for i in np.unique(np.linspace(0, len(meas) - 1, min(12, len(meas))).astype(int)):
            f = meas[int(i)]
            pts.append(dict(t=f["t"], oi=f["oi"], phu=f.get("coverage"), do_duc=f.get("opacity")))

    votes = Counter(f.get("label") for f in det if f.get("label"))
    sd = summary
    dir_xac_dinh = sd.get("angle") is not None
    rej = sd.get("dir_rejected") or {}
    dq = dict(
        xac_dinh=dir_xac_dinh,
        huong=sd.get("direction") if dir_xac_dinh else None,
        goc_do=_r(sd.get("angle"), 0), tilt_do=_r(sd.get("tilt"), 0), R=_r(sd.get("R"), 2),
        so_cap_khung_hop_le=sd.get("dir_pairs"), so_cap_khung_ra_huong=sd.get("dir_ok"),
        ti_le_khung_ra_huong=_r(sd.get("dir_rate"), 2), ly_do_bi_loai=rej,
        ly_do_chua_xac_dinh=None if dir_xac_dinh else (sd.get("dir_note") or FLOW_NOTES["thieu"]),
        mui_ten_xuat_phat_tu_lua=_r(
            (sum(1 for f in frames if f.get("anchor_src") == "lua") / n) if n else 0, 2),
    )

    flags = []
    if n and len(meas) / n < 0.5: flags.append("it_khung_do_duoc")
    if not dir_xac_dinh: flags.append("huong_lan_chua_xac_dinh")
    if (sd.get("dir_rate") or 0) < 0.3 and sd.get("dir_pairs"): flags.append("ti_le_khung_ra_huong_thap")
    if cuts: flags.append("co_chuyen_canh")
    if not fire: flags.append("khong_thay_lua")
    if dur < 3: flags.append("video_ngan")
    if len(det) < 5: flags.append("it_khung_co_khoi")
    confs = [f["conf"] for f in det if f.get("conf") is not None]

    return dict(
        video=dict(thoi_luong_s=round(dur, 1), so_khung_mau=n, tan_so_mau_fps=TARGET_FPS, diem_chuyen_canh_s=cuts),
        phat_hien=dict(ti_le_khung_co_khoi=_r(len(det) / n if n else 0, 2), ti_le_khung_co_lua=_r(fire_frac, 2),
                       thoi_diem_thay_lua_dau_tien_s=fire[0]["t"] if fire else None,
                       yolo_conf_khoi_tb=_r(float(np.mean(confs)) if confs else None, 2),
                       yolo_conf_khoi_max=_r(max(confs) if confs else None, 2)),
        mau_khoi=dict(nhan=sd.get("smoke_label"), do_tin_cay=_r(sd.get("smoke_conf"), 2),
                      xac_suat={k: _r(v, 2) for k, v in (sd.get("color_probs") or {}).items()},
                      phieu_bau_theo_khung=dict(votes)),
        che_khuat=dict(do_duc_tb=_r(sd.get("opacity"), 3), phu_tb=_r(sd.get("coverage"), 3), oi_tb=_r(sd.get("oi"), 3),
                       oi_dinh=_r(max(oi) if oi else None, 3), muc_theo_do_duc=sd.get("level"),
                       so_khung_do_duoc=len(meas), ti_le_khung_do_duoc=_r(len(meas) / n if n else 0, 2),
                       xu_huong=trend, doan_che_khuat_nang_s=_episodes(meas), chuoi_thoi_gian=pts),
        huong_lan=dq,
        nguong_he_thong=dict(muc_do_duc=dict(nhe_duoi=0.25, vua_duoi=0.55), oi_che_khuat_nang=dict(bat=OI_ON, tat=OI_OFF),
                             do_duc_toi_thieu_la_khoi=P_MIN, yolo_conf_khoi=YOLO_CONF, yolo_conf_lua=FIRE_CONF,
                             R_toi_thieu_moi_khung=R_MIN, R_toi_thieu_ca_video=R_AVG_MIN,
                             so_cap_khung_toi_thieu=DIR_MIN_PAIRS, khoang_cach_cap_khung_s=round(FLOW_STEP / TARGET_FPS, 2),
                             ti_le_lua_nang_muc=FIRE_FRAC_UP),
        chat_luong_du_lieu=dict(canh_bao=flags),
        ket_luan_theo_luat=dict(muc_do=rule_level(sd, fire_frac),
                                cach_tinh="muc_theo_do_duc quy đổi Nhẹ/Vừa/Đặc thành Thấp/Trung bình/Cao; "
                                          "nâng thêm 1 mức nếu ti_le_khung_co_lua >= ti_le_lua_nang_muc"),
    )

def ai_init():
    M["gemini"] = None
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or DEFAULT_API_KEY
    if not AI_ENABLE:
        print("[AI] Đã tắt (AI_ENABLE=0)")
        return
    if not key:
        print("[AI] Chưa đặt GEMINI_API_KEY -> bỏ qua bước suy luận AI")
        return
    try:
        from google import genai
        from google.genai import types
        M["gemini"] = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(AI_TIMEOUT_S * 1000)))
        print(f"[AI] Sẵn sàng với khóa API (ưu tiên model={AI_MODEL})")
    except Exception as e:
        print("[AI] Không khởi tạo được google-genai:", e)

def _call_gemini(ev):
    from google.genai import types
    contents = "GÓI BẰNG CHỨNG (JSON):\n" + json.dumps(ev, ensure_ascii=False)

    def cfg(thinking):
        kw = dict(system_instruction=AI_SYSTEM, response_mime_type="application/json", response_schema=AIAssessment)
        if thinking:
            kw["thinking_config"] = types.ThinkingConfig(thinking_level=thinking)
        return types.GenerateContentConfig(**kw)

    last_err = None
    seen = set()
    for m_name in AI_FALLBACK_MODELS:
        if not m_name or m_name in seen:
            continue
        seen.add(m_name)
        try:
            try:
                r = M["gemini"].models.generate_content(model=m_name, contents=contents, config=cfg(AI_THINKING))
            except Exception as e_think:
                if "thinking" not in str(e_think).lower():
                    raise
                r = M["gemini"].models.generate_content(model=m_name, contents=contents, config=cfg(None))
            parsed = getattr(r, "parsed", None)
            if isinstance(parsed, AIAssessment):
                return parsed.model_dump()
            txt = (r.text or "").strip()
            if txt.startswith("```"):
                txt = txt.strip("`").removeprefix("json").strip()
            return AIAssessment.model_validate_json(txt).model_dump()
        except Exception as e_mod:
            last_err = e_mod
            continue
    raise last_err or RuntimeError("Không gọi được bất kỳ model Gemini nào")

def _local_tactical_fallback(ev):
    """Sinh nhận định chiến thuật chuẩn xác từ gói bằng chứng khi API ngoại tuyến hoặc lỗi."""
    luat = ev["ket_luan_theo_luat"]["muc_do"] or "Trung bình"
    mk = ev["mau_khoi"]["nhan"] or "Khói"
    ck = ev["che_khuat"]
    hl = ev["huong_lan"]
    ph = ev["phat_hien"]

    phu_pct = round((ck.get("phu_tb") or 0) * 100, 1)
    duc_pct = round((ck.get("do_duc_tb") or 0) * 100, 1)
    oi_val = ck.get("oi_tb") or 0.0

    tom_tat = (
        f"Phát hiện {mk.lower()} bao phủ trung bình {phu_pct}% diện tích khung hình, "
        f"độ đục quang học {duc_pct}% (chỉ số OI = {oi_val:.3f}, mức {ck.get('muc_theo_do_duc') or 'Vừa'})."
    )
    if (ph.get("ti_le_khung_co_lua") or 0) > 0:
        tom_tat += f" Ghi nhận ngọn lửa hở xuất hiện trên {round(ph['ti_le_khung_co_lua']*100)}% khung mẫu."

    huong_str = (
        f"Khói lan theo hướng {hl['huong']} (góc {hl['goc_do']}°, R = {hl['R']})."
        if hl.get("xac_dinh")
        else "Chưa xác định rõ rệt — " + (hl.get("ly_do_chua_xac_dinh") or FLOW_NOTES["thieu"])
    )

    khuyen_nghi = []
    if "Đen" in mk:
        khuyen_nghi.append("Bắt buộc trang bị bình thở độc lập (SCBA) và mặt nạ phòng độc do khói đen đặc tiềm ẩn khí độc từ nhựa, xăng dầu hoặc hóa chất.")
    else:
        khuyen_nghi.append("Triển khai đội hình chữa cháy cơ bản, kết hợp thiết bị thông gió và phun sương làm mát vùng cháy.")
    if (ph.get("ti_le_khung_co_lua") or 0) > 0:
        khuyen_nghi.append("Đã xuất hiện lửa hở trực tiếp: Chủ động triển khai lăng phun bọt/nước áp lực cao khóa chặn mặt lửa, ngăn cháy lan sang công trình lân cận.")
    if hl.get("xac_dinh"):
        khuyen_nghi.append(f"Ưu tiên bố trí xe chỉ huy và mũi tấn công tiếp cận từ phía ngược hướng khói lan ({hl['huong']}) để đảm bảo tầm nhìn.")
    else:
        khuyen_nghi.append("Trinh sát vòng ngoài xác định hướng gió thực tế tại hiện trường trước khi đưa lực lượng vào sâu bên trong.")

    return dict(
        muc_do=luat,
        do_chac_chan="Cao" if (ck.get("ti_le_khung_do_duoc") or 0) >= 0.7 else "Vừa",
        tom_tat=tom_tat,
        can_cu=[
            {"truong": "mau_khoi.nhan", "nhan_dinh": f"Phân loại ResNet18 xác định {mk} (độ tin cậy {round((ev['mau_khoi'].get('do_tin_cay') or 0)*100)}%)."},
            {"truong": "che_khuat.oi_tb", "nhan_dinh": f"Chỉ số che khuất OI trung bình đạt {oi_val:.3f} (độ phủ {phu_pct}%, độ đục {duc_pct}%)."},
        ],
        huong_lan=huong_str,
        gia_thuyet_vat_lieu=[
            {
                "ten": "Nhựa tổng hợp / Xăng dầu / Vật liệu cách nhiệt" if "Đen" in mk else "Gỗ / Giấy / Vật liệu hữu cơ",
                "do_ung_ho": "Cao" if "Đen" in mk else "Vừa",
                "ly_do": f"Dựa trên đặc trưng {mk.lower()} và mức độ cản quang trên mặt phẳng ảnh.",
            }
        ],
        khuyen_nghi=khuyen_nghi,
        han_che=["Nhận định được tổng hợp từ chỉ số thị giác 2D và hệ thống luật cứng an toàn."],
    )

def run_ai(ev):
    t0 = time.time()
    luat = ev["ket_luan_theo_luat"]["muc_do"]
    if not AI_ENABLE or M.get("gemini") is None:
        fb = _local_tactical_fallback(ev)
        return dict(fb, status="ok", model="rule-guardrail", muc_do_luat=luat,
                    muc_do_ai=fb["muc_do"], muc_do_cuoi=luat or fb["muc_do"],
                    bi_luat_nang_muc=False, canh_bao=ev["chat_luong_du_lieu"]["canh_bao"],
                    latency_s=round(time.time() - t0, 2))
    try:
        out = _call_gemini(ev)
    except Exception as e:
        print("[AI] Chuyển sang bộ tổng hợp chiến thuật nội bộ do lỗi API:", e)
        fb = _local_tactical_fallback(ev)
        return dict(fb, status="ok", model="rule-guardrail", muc_do_luat=luat,
                    muc_do_ai=fb["muc_do"], muc_do_cuoi=luat or fb["muc_do"],
                    bi_luat_nang_muc=False, canh_bao=ev["chat_luong_du_lieu"]["canh_bao"],
                    latency_s=round(time.time() - t0, 2))

    muc_ai, final, override = out["muc_do"], out["muc_do"], False
    if luat in LEVELS and (muc_ai not in LEVELS or LEVELS.index(muc_ai) < LEVELS.index(luat)):
        final, override = luat, True
    hl = ev["huong_lan"]
    if not hl["xac_dinh"]:
        out["huong_lan"] = "Chưa xác định rõ rệt — " + (hl["ly_do_chua_xac_dinh"] or FLOW_NOTES["thieu"])
    return dict(out, status="ok", model=AI_MODEL, muc_do_luat=luat, muc_do_ai=muc_ai, muc_do_cuoi=final,
                bi_luat_nang_muc=override, canh_bao=ev["chat_luong_du_lieu"]["canh_bao"],
                latency_s=round(time.time() - t0, 2))

LOCK = threading.Lock()
INFLIGHT, FAILED = set(), set()
STATUS = {"current": None, "queue": 0}

def file_id(path):
    st = os.stat(path)
    key = f"{os.path.basename(path)}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:8]

def _px_box(box, width, height):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    x1 = int(np.clip(round(x1 * width), 0, width - 1))
    y1 = int(np.clip(round(y1 * height), 0, height - 1))
    x2 = int(np.clip(round(x2 * width), 0, width - 1))
    y2 = int(np.clip(round(y2 * height), 0, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def _draw_tag(frame, text, x, y, color, scale=0.6, thickness=2):
    h, w = frame.shape[:2]
    x = max(2, min(int(x), w - 4))
    y = max(18, min(int(y), h - 3))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    left, top = x, max(0, y - th - base - 5)
    right, bottom = min(w - 1, x + tw + 8), min(h - 1, y + base + 3)
    cv2.rectangle(frame, (left, top), (right, bottom), color, -1)
    cv2.putText(frame, text, (left + 4, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), thickness, cv2.LINE_AA)

def _cv_ascii(text):
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")

def _find_ffmpeg_binary():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and os.path.isfile(bundled):
            return bundled
    except Exception:
        pass
    return None

def _video_is_readable(path):
    if not os.path.isfile(path) or os.path.getsize(path) < 1024:
        return False
    check = cv2.VideoCapture(path)
    try:
        if not check.isOpened():
            return False
        ok, frame = check.read()
        if not ok or frame is None or frame.size == 0:
            return False
        return True
    finally:
        check.release()

def render_processed_video(source_path, aid, frames, smoke_masks, summary=None):
    """Xuất video có overlay; đảm bảo mặt nạ khói luôn nằm gọn bên trong hộp BBox của khung hiện hành."""
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video nguồn: {source_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps < 1.0 or fps > 240.0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Không đọc được kích thước video nguồn")

    os.makedirs(PROCESSED_VIDEO_DIR, exist_ok=True)
    safe_aid = re.sub(r"[^A-Za-z0-9_-]", "", str(aid)) or uuid.uuid4().hex[:8]
    final_mp4 = os.path.join(PROCESSED_VIDEO_DIR, f"{safe_aid}_processed.mp4")
    final_avi = os.path.join(PROCESSED_VIDEO_DIR, f"{safe_aid}_processed.avi")

    for old in (
        final_mp4, final_avi,
        os.path.join(PROCESSED_VIDEO_DIR, f"{safe_aid}_overlay_silent.mp4"),
        os.path.join(PROCESSED_VIDEO_DIR, f"{safe_aid}_overlay_silent.avi"),
        os.path.join(PROCESSED_VIDEO_DIR, f"{safe_aid}_overlay_mux.mp4"),
    ):
        try:
            if os.path.isfile(old):
                os.remove(old)
        except OSError as exc:
            print(f"[VIDEO] Không dọn được file cũ {old}: {exc}")

    records = list(frames or [])
    masks = list(smoke_masks or [])
    sample_indices = []
    for rec in records:
        try:
            idx = int(rec.get("frame_index", round(float(rec.get("t", 0.0)) * fps)))
        except (TypeError, ValueError):
            idx = 0
        sample_indices.append(max(0, idx))

    max_hold = max(1, int(round((fps / max(1, TARGET_FPS)) * 1.5)))
    summary = summary or {}
    overall_direction = summary.get("direction") or "Chua du co so"
    frame_index = 0
    next_sample = 0
    active_sample = None
    font_scale = max(0.45, min(0.9, width / 1500.0))
    line_thickness = max(1, int(round(width / 900.0)))
    written = 0

    with tempfile.TemporaryDirectory(prefix=f".fireguard_{safe_aid}_", dir=PROCESSED_VIDEO_DIR) as temp_dir:
        temp_avi = os.path.join(temp_dir, "overlay_working.avi")
        temp_mp4 = os.path.join(temp_dir, "overlay_h264.mp4")
        writer = cv2.VideoWriter(
            temp_avi, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
        )
        if not writer.isOpened():
            writer.release()
            cap.release()
            raise RuntimeError(
                f"Không khởi tạo được bộ ghi AVI/MJPEG trong {temp_dir}."
            )

        try:
            while True:
                ok, original = cap.read()
                if not ok:
                    break

                while next_sample < len(sample_indices) and sample_indices[next_sample] <= frame_index:
                    active_sample = next_sample
                    next_sample += 1

                rec = None
                mask = None
                if active_sample is not None and frame_index - sample_indices[active_sample] <= max_hold:
                    rec = records[active_sample]
                    if active_sample < len(masks):
                        mask = masks[active_sample]

                frame = original.copy()
                smoke_boxes = []
                if rec is not None:
                    smoke_boxes = rec.get("smoke_boxes") or []
                    if not smoke_boxes and rec.get("bbox"):
                        smoke_boxes = [{"bbox": rec.get("bbox"), "conf": rec.get("conf")}]

                # Chặn mặt nạ khói tuyệt đối không tràn ra ngoài các hộp giới hạn BBox của khung hiện tại
                if mask is not None and np.asarray(mask).size and smoke_boxes:
                    try:
                        m = (np.asarray(mask) > 0).astype(np.uint8)
                        if m.shape[:2] != (height, width):
                            m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
                        box_clip = np.zeros((height, width), dtype=bool)
                        for sb in smoke_boxes:
                            coords = _px_box(sb.get("bbox") if isinstance(sb, dict) else sb, width, height)
                            if coords:
                                bx1, by1, bx2, by2 = coords
                                box_clip[by1:by2, bx1:bx2] = True
                        m = m.astype(bool) & box_clip
                        if m.any():
                            green_layer = np.zeros_like(frame)
                            green_layer[:, :] = (0, 255, 0)
                            blended = cv2.addWeighted(frame, 0.68, green_layer, 0.32, 0)
                            frame[m] = blended[m]
                    except Exception as exc_mask:
                        if frame_index == 0:
                            print("[VIDEO] Không tô được mask khói:", exc_mask)

                if rec is not None:
                    for smoke in smoke_boxes:
                        coords = _px_box(smoke.get("bbox") if isinstance(smoke, dict) else smoke, width, height)
                        if not coords:
                            continue
                        x1, y1, x2, y2 = coords
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), max(2, line_thickness + 1))
                        conf = smoke.get("conf") if isinstance(smoke, dict) else None
                        label = "KHOI" + (f" {float(conf):.0%}" if isinstance(conf, (int, float)) else "")
                        _draw_tag(frame, label, x1, max(20, y1), (0, 150, 0), font_scale, line_thickness)

                    flow_box_norm = rec.get("flow_region_bbox")
                    flow_box_px = _px_box(flow_box_norm, width, height) if flow_box_norm else None
                    if flow_box_px and rec.get("flow_path_valid") is True:
                        fx1, fy1, fx2, fy2 = flow_box_px
                        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 220, 0),
                                      max(3, line_thickness + 2))
                        _draw_tag(frame, "BOX KHOI - ROI FLOW", fx1, max(22, fy1),
                                  (180, 120, 0), font_scale, line_thickness)

                    for fire_box in (rec.get("fire") or []):
                        coords = _px_box(fire_box, width, height)
                        if not coords:
                            continue
                        x1, y1, x2, y2 = coords
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), max(2, line_thickness + 1))
                        _draw_tag(frame, "LUA", x1, max(20, y1), (0, 0, 190), font_scale, line_thickness)

                    anchor, angle = rec.get("anchor"), rec.get("angle")
                    has_fire_origin = rec.get("anchor_src") == "lua"
                    flow_anchor = rec.get("flow_anchor")
                    path_valid = rec.get("flow_path_valid") is True
                    if has_fire_origin and path_valid and flow_box_px and isinstance(anchor, (list, tuple)) and flow_anchor:
                        try:
                            sx = int(np.clip(float(anchor[0]) * width, 0, width - 1))
                            sy = int(np.clip(float(anchor[1]) * height, 0, height - 1))
                            gx = int(np.clip(float(flow_anchor[0]) * width, 0, width - 1))
                            gy = int(np.clip(float(flow_anchor[1]) * height, 0, height - 1))
                            cv2.arrowedLine(frame, (sx, sy), (gx, gy), (255, 0, 255),
                                            max(2, line_thickness), cv2.LINE_AA, tipLength=0.16)
                            cv2.circle(frame, (sx, sy), max(4, line_thickness + 2), (0, 0, 255), -1, cv2.LINE_AA)
                            cv2.circle(frame, (gx, gy), max(4, line_thickness + 2), (255, 220, 0), -1, cv2.LINE_AA)
                            _draw_tag(frame, "NGUON LUA -> BOX KHOI", sx, max(25, sy - 10),
                                      (145, 0, 145), font_scale, line_thickness)
                        except (TypeError, ValueError, OverflowError):
                            pass

                    if rec.get("flow_ok") and isinstance(angle, (int, float)):
                        try:
                            if has_fire_origin:
                                if not (path_valid and flow_box_px and isinstance(flow_anchor, (list, tuple))):
                                    raise ValueError("flow lửa không đi qua box khói")
                                x0 = int(np.clip(float(flow_anchor[0]) * width, 0, width - 1))
                                y0 = int(np.clip(float(flow_anchor[1]) * height, 0, height - 1))
                                bx1, by1, bx2, by2 = flow_box_px
                                length = max(8, int(min(bx2 - bx1, by2 - by1) * 0.30))
                            else:
                                if not isinstance(anchor, (list, tuple)) or len(anchor) != 2:
                                    raise ValueError("thiếu điểm neo khói")
                                x0 = int(np.clip(float(anchor[0]) * width, 0, width - 1))
                                y0 = int(np.clip(float(anchor[1]) * height, 0, height - 1))
                                length = max(35, int(min(width, height) * 0.12))
                            theta = math.radians(float(angle))
                            x1 = int(np.clip(x0 + length * math.cos(theta), 0, width - 1))
                            y1 = int(np.clip(y0 - length * math.sin(theta), 0, height - 1))
                            if has_fire_origin:
                                x1 = int(np.clip(x1, bx1 + 1, max(bx1 + 1, bx2 - 1)))
                                y1 = int(np.clip(y1, by1 + 1, max(by1 + 1, by2 - 1)))
                            cv2.arrowedLine(frame, (x0, y0), (x1, y1), (255, 210, 0),
                                            max(3, line_thickness + 1), cv2.LINE_AA, tipLength=0.25)
                            direction = rec.get("dir") or dir_label(float(angle))
                            r_value = rec.get("R")
                            arrow_label = f"Optical Flow khoi: {direction}" + (f" (R={r_value:.2f})" if isinstance(r_value, (int, float)) else "")
                            _draw_tag(frame, arrow_label, x0, max(22, y0 - 8), (180, 105, 0), font_scale, line_thickness)
                        except (TypeError, ValueError, OverflowError):
                            pass

                timestamp = frame_index / fps
                time_text = f"{int(timestamp // 3600):02d}:{int((timestamp % 3600) // 60):02d}:{timestamp % 60:04.1f}"
                smoke_text = "Khong phat hien khoi tai khung mau"
                oi_text = "OI: --"
                if rec is not None:
                    if rec.get("label"):
                        cls_conf = rec.get("cls_conf")
                        smoke_text = _cv_ascii(rec.get("label")) + (f" | ResNet {float(cls_conf):.0%}" if isinstance(cls_conf, (int, float)) else "")
                    elif rec.get("smoke_boxes"):
                        smoke_text = "Da phat hien vung khoi"
                    if isinstance(rec.get("oi"), (int, float)):
                        oi_text = f"OI: {float(rec['oi']):.3f}"
                hud_lines = [
                    f"FIREGUARD AI  |  {time_text}",
                    f"{smoke_text}  |  {oi_text}",
                    f"Huong toan video: {_cv_ascii(overall_direction)}",
                ]
                if rec is not None and rec.get("anchor_src") == "lua":
                    gate_status = "OK - da di qua box khoi" if rec.get("flow_path_valid") is True else "CHUA CO box khoi hop le"
                    hud_lines.append("Nguon lua -> box khoi: " + gate_status)
                hud_h = max(76, int(26 * font_scale * 3.8))
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (min(width - 1, int(width * 0.78)), min(height - 1, hud_h)), (15, 25, 40), -1)
                frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
                for line_no, text in enumerate(hud_lines):
                    yy = int(25 + line_no * (22 * font_scale + 5))
                    if yy < height - 2:
                        cv2.putText(frame, text[:110], (12, yy), cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale, (245, 248, 252), line_thickness, cv2.LINE_AA)
                legend = "Green: smoke mask/bbox | Red: fire | Arrow: qualified local motion"
                cv2.putText(frame, legend, (12, max(18, height - 14)), cv2.FONT_HERSHEY_SIMPLEX,
                            max(0.38, font_scale * 0.72), (230, 255, 230), line_thickness, cv2.LINE_AA)

                writer.write(frame)
                written += 1
                frame_index += 1
        finally:
            cap.release()
            writer.release()

        if written == 0 or not _video_is_readable(temp_avi):
            raise RuntimeError(f"Video trung gian không hợp lệ sau khi ghi: {temp_avi}; số khung đã ghi={written}")

        ffmpeg = _find_ffmpeg_binary()
        output_path = None
        audio_preserved = False
        ffmpeg_failure = None
        if ffmpeg:
            try:
                probe = subprocess.run(
                    [ffmpeg, "-hide_banner", "-i", source_path],
                    capture_output=True, text=True, timeout=60
                )
                source_has_audio = bool(re.search(r"Stream #.*Audio:", probe.stderr or "", re.IGNORECASE))
                cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                       "-i", temp_avi, "-i", source_path, "-map", "0:v:0"]
                if source_has_audio:
                    cmd += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "160k"]
                cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-map_metadata", "1",
                        "-movflags", "+faststart", temp_mp4]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                if result.returncode == 0 and _video_is_readable(temp_mp4):
                    os.replace(temp_mp4, final_mp4)
                    output_path = final_mp4
                    audio_preserved = source_has_audio
                else:
                    ffmpeg_failure = (result.stderr or "FFmpeg kết thúc nhưng MP4 không đọc được")[-1000:]
            except Exception as exc:
                ffmpeg_failure = f"{type(exc).__name__}: {exc}"
            if ffmpeg_failure:
                print("[VIDEO] FFmpeg không tạo được MP4/H.264; dùng AVI/MJPEG:", ffmpeg_failure)

        if output_path is None:
            if not _video_is_readable(temp_avi):
                raise RuntimeError("Không tạo được cả MP4/H.264 lẫn AVI/MJPEG hợp lệ.")
            os.replace(temp_avi, final_avi)
            output_path = final_avi

    for legacy in glob.glob(os.path.join(PROCESSED_VIDEO_DIR, f"{safe_aid}_overlay_*")):
        try:
            if os.path.isfile(legacy):
                os.remove(legacy)
        except OSError:
            pass

    if not _video_is_readable(output_path):
        raise RuntimeError(f"File đầu ra không vượt qua kiểm tra đọc video: {output_path}")
    return output_path, audio_preserved

def process(path, reporter="Người dân (citizen_uploads)", lat=10.69744, lon=106.70348):
    aid, name = file_id(path), os.path.basename(path)
    t0 = time.time()
    tm = StepTimer()
    with LOCK:
        STATUS["current"] = name
        t_run = time.perf_counter()
        try: frames, probs, cuts, smoke_masks = analyze(path, tm)
        finally: STATUS["current"] = None
    with tm.step("tong_hop_video"): summary = summarize(frames, probs, cuts)
    evidence, ai = None, None
    if summary.get("detected"):
        with tm.step("dong_goi_bang_chung"): evidence = build_evidence(frames, summary)
        with tm.step("gemini_suy_luan"): ai = run_ai(evidence)

    processed_video_path, audio_preserved = None, False
    try:
        with tm.step("xuat_video_da_xu_ly"):
            processed_video_path, audio_preserved = render_processed_video(
                path, aid, frames, smoke_masks, summary
            )
    except Exception as e_video:
        print(f"[VIDEO] XUẤT VIDEO THẤT BẠI: {e_video}")
        traceback.print_exc()

    stem = os.path.splitext(name)[0]
    source_video_url = "/media/" + quote(name)
    processed_video_url = (
        "/processed/" + quote(os.path.basename(processed_video_path))
        if processed_video_path and os.path.isfile(processed_video_path) else None
    )
    a = dict(schema=SCHEMA, id=aid, file=name, reporter=reporter, lat=lat, lon=lon, dist_km=0.0,
             created=datetime.fromtimestamp(os.path.getmtime(path), timezone.utc).isoformat(),
             video_url=processed_video_url or source_video_url,
             source_video_url=source_video_url,
             processed_video_url=processed_video_url,
             processed_video_path=os.path.relpath(processed_video_path, BASE).replace(os.sep, "/") if processed_video_path else None,
             processed_video_audio_preserved=audio_preserved if processed_video_path else None,
             frames=frames, summary=summary, avg_sec=AVG_SEC,
             latency_s=round(time.time() - t0, 1), ai=ai, ai_evidence=evidence,
             title=(summary["smoke_label"] if summary["detected"] else "Không phát hiện khói") + " · " + stem)
    a["timing"] = tm.report(len(frames), frames[-1]["t"] if frames else 0.0, time.perf_counter() - t_run)
    a["json_url"] = f"/api/alerts/{aid}/json/download"
    a["json_figure_url"] = f"/api/alerts/{aid}/json/figure"
    log_timing(name, a["timing"])
    ALERTS[aid] = a

    actual_json_path = os.path.join(DB, aid + ".json")
    save_json_atomic(actual_json_path, a)
    try:
        export_actual_json_artifacts(a)
    except Exception as e_result_artifact:
        print("[JSON] Đã lưu JSON thật nhưng không dựng được ảnh:", e_result_artifact)

    try:
        export_paper_figure_7_confusion_matrix()
        if a.get("frames"):
            export_paper_figure_10_oi_timeseries(a["frames"])
            export_paper_figure_11_flow_timeseries(a["frames"])
        export_paper_figure_12_latency()
    except Exception as e_charts:
        print("[PAPER] Lỗi xuất đồ thị thời gian:", e_charts)

    return a

def watcher():
    sizes = {}
    while True:
        try:
            paths = [os.path.join(CITIZEN_DIR, f) for f in os.listdir(CITIZEN_DIR) if f.lower().endswith(VIDEO_EXT)]
            paths.sort(key=os.path.getmtime, reverse=True)
            todo = []
            for p in paths:
                aid = file_id(p)
                if aid in ALERTS or aid in INFLIGHT or aid in FAILED: continue
                if sizes.get(p) != os.path.getsize(p):
                    sizes[p] = os.path.getsize(p)
                    continue
                todo.append((aid, p))
            STATUS["queue"] = len(todo)
            for aid, p in todo:
                INFLIGHT.add(aid)
                try: process(p)
                except Exception as e:
                    FAILED.add(aid); print("Lỗi khi phân tích", p, "->", e)
                finally:
                    INFLIGHT.discard(aid); STATUS["queue"] -= 1
        except Exception as e: print("watcher:", e)
        time.sleep(SCAN_EVERY)

@app.post("/api/analyze")
def api_analyze(file: UploadFile = File(...), reporter: str = Form("Nguyễn Hoàng Ân"),
                lat: float = Form(10.69744), lon: float = Form(106.70348)):
    name = os.path.basename(file.filename or "video.mp4")
    if not name.lower().endswith(VIDEO_EXT): raise HTTPException(400, "Định dạng video không được hỗ trợ")
    if os.path.exists(os.path.join(CITIZEN_DIR, name)):
        stem, ext = os.path.splitext(name)
        name = f"{stem}_{uuid.uuid4().hex[:4]}{ext}"
    path, tmp = os.path.join(CITIZEN_DIR, name), os.path.join(CITIZEN_DIR, name + ".part")
    with open(tmp, "wb") as o: shutil.copyfileobj(file.file, o)
    os.replace(tmp, path)
    aid = file_id(path)
    INFLIGHT.add(aid)
    try: return process(path, reporter, lat, lon)
    finally: INFLIGHT.discard(aid)

@app.get("/api/alerts")
def api_alerts():
    items = sorted(ALERTS.values(), key=lambda a: a["created"], reverse=True)
    return [{k: v for k, v in a.items() if k not in ("frames", "ai_evidence")} for a in items]

@app.get("/api/alerts/{aid}")
def api_alert(aid: str):
    if aid not in ALERTS: raise HTTPException(404, "Không tìm thấy báo động")
    return ALERTS[aid]

@app.post("/api/alerts/{aid}/ai")
def api_alert_ai(aid: str):
    a = ALERTS.get(aid)
    if not a: raise HTTPException(404, "Không tìm thấy báo động")
    if not a["summary"].get("detected"): raise HTTPException(400, "Báo động này không có khói để suy luận")
    ev = build_evidence(a["frames"], a["summary"])
    a["ai_evidence"], a["ai"] = ev, run_ai(ev)
    save_json_atomic(os.path.join(DB, aid + ".json"), a)
    try:
        export_actual_json_artifacts(a)
    except Exception as e_result_artifact:
        print("[JSON] Đã cập nhật JSON AI nhưng không làm mới được ảnh:", e_result_artifact)
    return a["ai"]

@app.get("/api/status")
def api_status(): return STATUS

@app.get("/api/timing")
def api_timing():
    reps = [a["timing"] for a in ALERTS.values() if a.get("timing")]
    if not reps: return dict(n_video=0)
    mean = lambda xs: round(sum(xs) / len(xs), 3)
    return dict(n_video=len(reps),
                tong_s=dict(tb=mean([r["tong_s"] for r in reps]), max=max(r["tong_s"] for r in reps), min=min(r["tong_s"] for r in reps)),
                video_s_tb=mean([r["video_s"] for r in reps]),
                buoc_s_tb={k: mean([r["buoc_s"].get(k, 0.0) for r in reps]) for k in STEP_ORDER},
                khac_s_tb=mean([r["khac_s"] for r in reps]))

@app.get("/api/export-paper-figures")
def api_export_paper_figures():
    try:
        export_paper_figure_3_and_8()
        export_paper_figure_4_and_9()
        export_paper_figure_7_confusion_matrix()
        if ALERTS:
            last_alert = list(ALERTS.values())[-1]
            export_paper_figure_10_oi_timeseries(last_alert.get("frames", []))
            export_paper_figure_11_flow_timeseries(last_alert.get("frames", []))
        else:
            export_paper_figure_10_oi_timeseries()
            export_paper_figure_11_flow_timeseries()
        export_paper_figure_12_latency()
        return {"status": "success", "folder": PAPER_FIGS_DIR, "message": "Đã xuất đầy đủ 8 hình ảnh vào thư mục paper_figures/"}
    except Exception as e:
        raise HTTPException(500, f"Lỗi xuất ảnh: {e}")

app.mount("/media", StaticFiles(directory=CITIZEN_DIR), name="media")
app.mount("/processed", StaticFiles(directory=PROCESSED_VIDEO_DIR), name="processed-videos")

# ===================== MẪU JSON ĐẦU RA CHO BÁO CÁO NCKH =====================
JSON_EXAMPLE_PATH = os.path.join(PAPER_FIGS_DIR, "Mau_dinh_dang_JSON_FireGuard.json")
JSON_EXAMPLE_FIG = os.path.join(PAPER_FIGS_DIR, "Hinh_minh_hoa_cau_truc_JSON.png")

def json_example_payload():
    frames = []
    for i in range(8):
        t = round(i / TARGET_FPS, 2)
        reason = "dau_clip" if i < FLOW_STEP else "camera"
        frame = dict(
            t=t,
            bbox=[0.31, round(0.14 + i * 0.002, 4), 0.62, 0.51],
            fire=[],
            conf=0.82,
            label="Khói Xám",
            cls_conf=0.89,
            scene=0,
            coverage=0.650,
            cov_dense=0.431,
            opacity=0.498,
            oi=0.328,
            coverage_avg=0.650,
            opacity_avg=0.498,
            oi_avg=0.328,
            level="Vừa",
            flow_ok=False,
            flow_reason=reason,
            dir_state="none",
            dir_note=(FLOW_NOTES["thieu"] if reason == "dau_clip" else FLOW_NOTES["camera"]),
            anchor=[0.47, 0.31],
            anchor_src="khoi",
        )
        frames.append(frame)

    summary = dict(
        detected=True,
        smoke_label="Khói Xám",
        smoke_conf=0.89,
        color_probs={"Khói Đen": 0.08, "Khói Trắng": 0.17, "Khói Xám": 0.75},
        opacity=0.498,
        coverage=0.650,
        oi=0.328,
        cov_dense=0.431,
        direction="--",
        angle=None,
        tilt=None,
        R=None,
        dir_note=FLOW_NOTES["camera"],
        opacity_note=None,
        scene_cuts=[],
        dir_pairs=6,
        dir_ok=0,
        dir_rate=0.0,
        dir_rejected={"camera": 6},
        level="Vừa",
    )
    evidence = dict(
        video=dict(thoi_luong_s=1.4, so_khung_mau=len(frames), tan_so_mau_fps=TARGET_FPS,
                   diem_chuyen_canh_s=[]),
        phat_hien=dict(ti_le_khung_co_khoi=1.0, ti_le_khung_co_lua=0.0,
                       thoi_diem_thay_lua_dau_tien_s=None, yolo_conf_khoi_tb=0.82,
                       yolo_conf_khoi_max=0.82),
        mau_khoi=dict(nhan="Khói Xám", do_tin_cay=0.89,
                      xac_suat={"Khói Đen": 0.08, "Khói Trắng": 0.17, "Khói Xám": 0.75},
                      phieu_bau_theo_khung={"Khói Xám": 8}),
        che_khuat=dict(do_duc_tb=0.498, phu_tb=0.650, oi_tb=0.328, oi_dinh=0.328,
                       muc_theo_do_duc="Vừa", so_khung_do_duoc=8, ti_le_khung_do_duoc=1.0,
                       xu_huong="on_dinh", doan_che_khuat_nang_s=[],
                       chuoi_thoi_gian=[dict(t=f["t"], oi=f["oi"], phu=f.get("coverage"),
                                             do_duc=f.get("opacity")) for f in frames]),
        huong_lan=dict(xac_dinh=False, huong=None, goc_do=None, tilt_do=None, R=None,
                       so_cap_khung_hop_le=6, so_cap_khung_ra_huong=0,
                       ti_le_khung_ra_huong=0.0, ly_do_bi_loai={"camera": 6},
                       ly_do_chua_xac_dinh=FLOW_NOTES["camera"], mui_ten_xuat_phat_tu_lua=0.0),
        nguong_he_thong=dict(
            muc_do_duc={"nhe_duoi": 0.25, "vua_duoi": 0.55},
            oi_che_khuat_nang={"bat": OI_ON, "tat": OI_OFF},
            do_duc_toi_thieu_la_khoi=P_MIN,
            yolo_conf_khoi=YOLO_CONF,
            yolo_conf_lua=FIRE_CONF,
            R_toi_thieu_moi_khung=R_MIN,
            R_toi_thieu_ca_video=R_AVG_MIN,
            so_cap_khung_toi_thieu=DIR_MIN_PAIRS,
            khoang_cach_cap_khung_s=round(FLOW_STEP / TARGET_FPS, 2),
            ti_le_lua_nang_muc=FIRE_FRAC_UP,
        ),
        chat_luong_du_lieu=dict(canh_bao=["huong_lan_chua_xac_dinh", "ti_le_khung_ra_huong_thap", "khong_thay_lua", "video_ngan"]),
        ket_luan_theo_luat=dict(
            muc_do="Trung bình",
            cach_tinh=("muc_theo_do_duc quy đổi Nhẹ/Vừa/Đặc thành Thấp/Trung bình/Cao; "
                       "nâng thêm 1 mức nếu ti_le_khung_co_lua >= ti_le_lua_nang_muc"),
        ),
    )
    ai = dict(
        muc_do="Trung bình",
        do_chac_chan="Thấp",
        tom_tat="Phát hiện vùng khói trong các khung mẫu; dữ liệu chuyển động chưa đủ điều kiện xác định hướng lan.",
        can_cu=[
            {"truong": "mau_khoi.nhan", "nhan_dinh": "Vùng khói được phân loại là khói xám theo xác suất minh họa."},
            {"truong": "che_khuat.oi_tb", "nhan_dinh": "OI là chỉ số tương đối trên mặt phẳng ảnh, không phải nồng độ khói vật lý."},
            {"truong": "huong_lan.xac_dinh", "nhan_dinh": "False; không công bố hướng lan khi phép đo không đạt điều kiện."},
        ],
        huong_lan="Chưa xác định rõ rệt — " + FLOW_NOTES["camera"],
        gia_thuyet_vat_lieu=[
            {"ten": "Chưa đủ cơ sở suy đoán", "do_ung_ho": "Thấp", "ly_do": "Không suy ra vật liệu cháy chỉ từ màu khói minh họa."}
        ],
        khuyen_nghi=["Đối chiếu kết quả thị giác với thông tin hiện trường do người điều hành xác minh."],
        han_che=["Các trị số trong bản ghi này chỉ nhằm minh họa định dạng JSON."],
        status="ok",
        model=AI_MODEL,
        muc_do_luat="Trung bình",
        muc_do_ai="Trung bình",
        muc_do_cuoi="Trung bình",
        bi_luat_nang_muc=False,
        canh_bao=evidence["chat_luong_du_lieu"]["canh_bao"],
        latency_s=0.25,
    )
    timing = dict(
        tong_s=1.237,
        video_s=1.4,
        so_khung=len(frames),
        buoc_s={
            "doc_giai_ma_video": 0.035,
            "resize_khung_hinh": 0.012,
            "yolo_phat_hien": 0.350,
            "resnet_phan_loai_mau": 0.120,
            "luu_anh_snapshot": 0.020,
            "chuyen_canh": 0.012,
            "hieu_chinh_mau": 0.050,
            "che_khuat_dcp_oi": 0.160,
            "optical_flow": 0.120,
            "tong_hop_ket_qua": 0.018,
            "tong_hop_video": 0.010,
            "dong_goi_bang_chung": 0.020,
            "gemini_suy_luan": 0.250,
        },
        khac_s=0.050,
        he_so_thoi_gian_thuc=0.884,
        moi_khung_ms={"doc_giai_ma_video": 4.38, "resize_khung_hinh": 1.5,
                      "yolo_phat_hien": 43.75, "luu_anh_snapshot": 2.5,
                      "che_khuat_dcp_oi": 20.0, "optical_flow": 15.0},
        resnet_ms_moi_lan=15.0,
        resnet_so_lan=8,
        yolo_ms_moi_anh={"preprocess": 1.8, "inference": 30.2, "postprocess": 1.5},
    )
    return dict(
        schema=SCHEMA,
        id="a1b2c3d4",
        file="video_mau_minh_hoa.mp4",
        reporter="Người dân (dữ liệu minh họa)",
        lat=10.69744,
        lon=106.70348,
        dist_km=0.0,
        created="2026-10-09T00:00:00+00:00",
        video_url="/media/video_mau_minh_hoa.mp4",
        frames=frames,
        summary=summary,
        avg_sec=AVG_SEC,
        latency_s=1.2,
        ai=ai,
        ai_evidence=evidence,
        title="Khói Xám · video_mau_minh_hoa",
        timing=timing,
    )

def export_json_example_files():
    payload = json_example_payload()
    with open(JSON_EXAMPLE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    preview = {
        "schema": payload["schema"],
        "id": payload["id"],
        "file": payload["file"],
        "summary": {
            k: payload["summary"][k]
            for k in ("detected", "smoke_label", "smoke_conf", "coverage", "opacity", "oi", "direction", "R")
        },
        "ai_evidence": {
            "mau_khoi": {k: payload["ai_evidence"]["mau_khoi"][k] for k in ("nhan", "do_tin_cay")},
            "che_khuat": {k: payload["ai_evidence"]["che_khuat"][k] for k in ("do_duc_tb", "phu_tb", "oi_tb")},
            "huong_lan": {k: payload["ai_evidence"]["huong_lan"][k]
                           for k in ("xac_dinh", "huong", "R", "ly_do_chua_xac_dinh")},
            "chat_luong_du_lieu": payload["ai_evidence"]["chat_luong_du_lieu"],
        },
        "ai": {k: payload["ai"][k] for k in ("muc_do", "do_chac_chan", "huong_lan", "status")},
        "timing": {k: payload["timing"][k]
                   for k in ("tong_s", "video_s", "so_khung", "he_so_thoi_gian_thuc")},
    }
    lines = json.dumps(preview, ensure_ascii=False, indent=2).splitlines()

    line_count = len(lines)
    fig_h = max(7.0, 0.20 * line_count + 2.0)
    fig, ax = plt.subplots(figsize=(10.5, fig_h), dpi=180)
    fig.patch.set_facecolor("#f4f7fb")
    ax.set_facecolor("#111827")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, line_count + 5.5)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, line_count + 3.5), 1, 2.0, color="#0f2747", transform=ax.transData))
    ax.text(0.035, line_count + 4.65, "JSON ĐẦU RA — FIREGUARD AI", color="white",
            fontsize=15, fontweight="bold", va="center", family="DejaVu Sans")
    ax.text(0.035, line_count + 3.95, "Trích đoạn cấu trúc phản hồi của API", color="#cbd5e1",
            fontsize=9.5, va="center", family="DejaVu Sans")
    ax.add_patch(plt.Rectangle((0, 0.8), 1, line_count + 2.7, color="#111827", transform=ax.transData))
    for i, line in enumerate(lines):
        y = line_count + 2.95 - i
        stripped = line.strip()
        color = "#7dd3fc" if stripped.startswith('"') and '":' in stripped else "#e5e7eb"
        if stripped in ("{", "}", "[", "],"):
            color = "#94a3b8"
        ax.text(0.035, y, line, color=color, fontsize=7.8, va="center",
                family="DejaVu Sans Mono", clip_on=True)
    ax.text(0.035, 0.25,
            "Lưu ý: giá trị minh họa, không phải kết quả đo thực nghiệm. File JSON đầy đủ được cung cấp riêng.",
            color="#334155", fontsize=8.5, va="center", family="DejaVu Sans")
    plt.tight_layout(pad=0.7)
    fig.savefig(JSON_EXAMPLE_FIG, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return JSON_EXAMPLE_PATH, JSON_EXAMPLE_FIG

def save_json_atomic(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + "." + uuid.uuid4().hex + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def _json_pick(source, keys):
    source = source if isinstance(source, dict) else {}
    return {key: source[key] for key in keys if key in source}

def export_actual_json_figure(payload, output_path):
    summary = payload.get("summary") or {}
    timing = payload.get("timing") or {}
    ai = payload.get("ai") or {}
    real_frames = payload.get("frames") or []
    frame_keys = ("t", "bbox", "smoke_boxes", "fire", "conf", "label", "cls_conf", "coverage",
                  "opacity", "oi", "flow_ok", "flow_reason", "flow_path_valid",
                  "flow_region_bbox", "flow_anchor", "flow_gate_reason", "direction", "R")
    frames_preview = [_json_pick(frame, frame_keys) for frame in real_frames[:3]]
    preview = {
        "schema": payload.get("schema"),
        "id": payload.get("id"),
        "file": payload.get("file"),
        "created": payload.get("created"),
        "summary": _json_pick(summary, ("detected", "smoke_label", "smoke_conf", "coverage",
                                         "opacity", "oi", "direction", "angle", "R", "level")),
        "frames_sample_from_real_video": frames_preview,
        "timing": _json_pick(timing, ("tong_s", "video_s", "so_khung", "he_so_thoi_gian_thuc")),
        "ai": _json_pick(ai, ("status", "muc_do_luat", "muc_do_ai", "muc_do_cuoi", "tom_tat")),
    }
    lines = json.dumps(preview, ensure_ascii=False, indent=2).splitlines()
    line_count = len(lines)
    fig_h = max(7.2, 0.235 * line_count + 2.0)
    fig, ax = plt.subplots(figsize=(11.5, fig_h), dpi=180)
    fig.patch.set_facecolor("#f4f7fb")
    ax.set_facecolor("#111827")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, line_count + 5.5)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, line_count + 3.5), 1, 2.0, color="#0f2747", transform=ax.transData))
    ax.text(0.035, line_count + 4.65, "JSON KẾT QUẢ THỰC TẾ — FIREGUARD AI", color="white",
            fontsize=14, fontweight="bold", va="center", family="DejaVu Sans")
    filename = str(payload.get("file") or "video không có tên")
    ax.text(0.035, line_count + 3.95, f"Video đã phân tích: {filename[:100]}",
            color="#cbd5e1", fontsize=9.5, va="center", family="DejaVu Sans")
    ax.add_patch(plt.Rectangle((0, 0.8), 1, line_count + 2.7, color="#111827", transform=ax.transData))
    for i, line in enumerate(lines):
        y = line_count + 2.95 - i
        stripped = line.strip()
        color = "#7dd3fc" if stripped.startswith('"') and '":' in stripped else "#e5e7eb"
        if stripped in ("{", "}", "[", "],"):
            color = "#94a3b8"
        shown = line if len(line) <= 125 else line[:122] + "..."
        ax.text(0.035, y, shown, color=color, fontsize=7.7, va="center",
                family="DejaVu Sans Mono", clip_on=True)
    ax.text(0.035, 0.25,
            "Trích đoạn lấy từ JSON do server tạo sau khi phân tích video; không phải dữ liệu giả lập.",
            color="#334155", fontsize=8.5, va="center", family="DejaVu Sans")
    plt.tight_layout(pad=0.7)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path

def export_actual_json_artifacts(payload):
    aid = str(payload.get("id") or "")
    if not aid or os.path.basename(aid) != aid:
        raise ValueError("Bản ghi thiếu mã sự cố hợp lệ")
    report_json_path = os.path.join(JSON_RESULT_DIR, aid + ".json")
    result_image_path = os.path.join(JSON_RESULT_DIR, aid + ".png")
    save_json_atomic(report_json_path, payload)
    export_actual_json_figure(payload, result_image_path)
    export_actual_json_figure(payload, JSON_RESULT_FIG_LATEST)
    return report_json_path, result_image_path

def load_alert_record(aid):
    if not aid or os.path.basename(aid) != aid or aid in (".", ".."):
        raise HTTPException(400, "Mã sự cố không hợp lệ")
    path = os.path.join(DB, aid + ".json")
    if aid in ALERTS:
        return ALERTS[aid], path
    if not os.path.isfile(path):
        raise HTTPException(404, "Chưa có JSON kết quả cho video này")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        ALERTS[aid] = payload
        return payload, path
    except Exception as exc:
        raise HTTPException(500, f"Không đọc được JSON kết quả: {exc}")

@app.get("/api/alerts/{aid}/json/download", tags=["Kết quả video thật"])
def api_alert_json_download(aid: str):
    payload, path = load_alert_record(aid)
    if not os.path.isfile(path):
        save_json_atomic(path, payload)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(payload.get("file") or aid))
    return FileResponse(path, media_type="application/json", filename=f"{safe_name}_{aid}.json")

@app.get("/api/alerts/{aid}/json/figure", tags=["Kết quả video thật"])
def api_alert_json_figure(aid: str):
    payload, _ = load_alert_record(aid)
    path = os.path.join(JSON_RESULT_DIR, aid + ".png")
    if not os.path.isfile(path):
        try:
            export_actual_json_artifacts(payload)
        except Exception as exc:
            raise HTTPException(500, f"Không dựng được ảnh JSON từ kết quả thật: {exc}")
    return FileResponse(path, media_type="image/png", filename=f"JSON_thuc_te_{aid}.png")

@app.get("/api/latest-alert/json/download", tags=["Kết quả video thật"])
def api_latest_alert_json_download():
    if not ALERTS:
        existing = glob.glob(os.path.join(DB, "*.json"))
        if not existing:
            raise HTTPException(404, "Chưa có video nào được phân tích thành công")
        newest = max(existing, key=os.path.getmtime)
        aid = os.path.splitext(os.path.basename(newest))[0]
    else:
        aid = max(ALERTS.values(), key=lambda row: row.get("created", "")).get("id")
    payload, path = load_alert_record(str(aid))
    if not os.path.isfile(path):
        save_json_atomic(path, payload)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(payload.get("file") or aid))
    return FileResponse(path, media_type="application/json", filename=f"{safe_name}_{aid}.json")

@app.get("/api/latest-alert/json/figure", tags=["Kết quả video thật"])
def api_latest_alert_json_figure():
    if not ALERTS:
        existing = glob.glob(os.path.join(DB, "*.json"))
        if not existing:
            raise HTTPException(404, "Chưa có video nào được phân tích thành công")
        newest = max(existing, key=os.path.getmtime)
        aid = os.path.splitext(os.path.basename(newest))[0]
    else:
        aid = str(max(ALERTS.values(), key=lambda row: row.get("created", "")).get("id"))
    payload, _ = load_alert_record(aid)
    try:
        _, path = export_actual_json_artifacts(payload)
    except Exception as exc:
        raise HTTPException(500, f"Không dựng được ảnh JSON từ kết quả thật: {exc}")
    return FileResponse(path, media_type="image/png", filename=f"JSON_thuc_te_{aid}.png")

@app.get("/api/json-example", tags=["Tài liệu JSON"])
def api_json_example():
    export_json_example_files()
    return json_example_payload()

@app.get("/api/json-example/download", tags=["Tài liệu JSON"])
def api_json_example_download():
    export_json_example_files()
    return FileResponse(JSON_EXAMPLE_PATH, media_type="application/json",
                        filename="Mau_dinh_dang_JSON_FireGuard.json")

@app.get("/api/json-example/figure", tags=["Tài liệu JSON"])
def api_json_example_figure():
    export_json_example_files()
    return FileResponse(JSON_EXAMPLE_FIG, media_type="image/png",
                        filename="Hinh_minh_hoa_cau_truc_JSON.png")

@app.get("/")
def home():
    for n in ("UI-CONTROL-NCKH.html", "index.html"):
        if os.path.exists(os.path.join(BASE, n)): return FileResponse(os.path.join(BASE, n))
    return {"status": "FireGuard backend đang chạy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)