# python pipeline_mul.py

"""
Rule 프리필터 + YAMNet 임베딩 + MLP 통합 추론 (온라인 YAMNet)
"""

import functools
print = functools.partial(print, flush=True)
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import os
import glob
import unicodedata
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import tensorflow_hub as hub

from prefilter import rule_prefilter
from common_raw_audio import load_wav_fixed

# ---------- 설정 ----------
AUDIO_ROOT = "/Users/a1/종설/validation_copy" #음원파일위치
MLP_BEST_MODEL = "/Users/a1/종설/yamnet_mlp_best.pt" #mlp 모델 경로
OUT_CSV_PATH = "./result/pipe/rule_yamnet_mlp_from_emb.csv"  #결과 엑셀 파일
TARGET_SR = 16000

RULE_MIN_SCORE = 0.15           # rule 프리필터 기준
POSITIVE_PREFIX = {"S1", "S2", "S3", "S8", "S10"}       # 양성 클래스 prefix
YAMNET_MODEL_HANDLE = os.path.expanduser("~/yamnet_local") #다운 방법 read.me에 적어둠

# 폰트 설정 (한글 깨짐 방지)
plt.rcParams['font.family'] = 'AppleGothic'   # 또는 'Noto Sans CJK KR' NanumGothic
plt.rcParams['axes.unicode_minus'] = False


def norm(s: str, form: str = "NFD") -> str:
    return unicodedata.normalize(form, s)


# ================================
#  1) MLP
# ================================
class MLP(nn.Module):
    def __init__(self, in_dim=1024, hidden1=512, hidden2=256, out_dim=6, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_mlp_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MLP_BEST_MODEL, map_location=device)
    raw_label_to_idx = ckpt["label_to_idx"]
    label_to_idx = {unicodedata.normalize("NFC", k): v for k, v in raw_label_to_idx.items()}
    idx_to_label = {v: k for k, v in label_to_idx.items()}


    model = MLP(in_dim=1024, hidden1=512, hidden2=256, out_dim=len(label_to_idx))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    print("MLP 로드 완료:", MLP_BEST_MODEL)
    print("클래스:", list(idx_to_label.values()))
    return model, device, label_to_idx, idx_to_label


# ================================
#  2) YAMNet
# ================================
def load_yamnet_model():
    print("⏳ Loading YAMNet...")
    model = hub.load(YAMNET_MODEL_HANDLE)
    print("YAMNet loaded.")
    return model


def wav_to_yamnet_embedding(wav, sr, yamnet_model):
    waveform = tf.convert_to_tensor(wav, dtype=tf.float32)
    scores, embeddings, _ = yamnet_model(waveform)
    return tf.reduce_mean(embeddings, axis=0).numpy().astype("float32")


# 숫자 추출 (part_01 → 1)
def get_tail_digits(name: str) -> int:
    m = re.search(r"(\d{1,2})$", name)
    return int(m.group(1)) if m else 0


# ================================
#  3) 단일 파일 추론 (Rule → YAMNet → MLP)
# ================================
def infer_one_file(
    wav_path,
    target_sr,
    mlp_model,
    device,
    idx_to_label,
    yamnet_model,
):
    t0 = time.perf_counter()  # 함수 시작 시간

    rel_path = os.path.relpath(wav_path, AUDIO_ROOT)
    true_label = rel_path.split(os.sep)[0]
    true_label = unicodedata.normalize("NFC", true_label)

    wav = load_wav_fixed(wav_path, sr=target_sr)

    # -------------------------
    # Stage 0: rule prefilter
    # -------------------------
    rule_score = rule_prefilter(wav, sr=target_sr, min_db=-35.0)
    if rule_score < RULE_MIN_SCORE:
        elapsed = time.perf_counter() - t0
        return {
            "path": wav_path,
            "stage": "rule_filtered",
            "true_label": true_label,
            "rule_score": rule_score,
            "pred_label": "RULE_FILTERED",
            "pred_prefix": None,
            "pred_prob": None,
            "reason": f"rule_filtered (rule_score={rule_score:.3f} < {RULE_MIN_SCORE})",
            "elapsed": elapsed,
        }

    # -------------------------
    # Stage 1: YAMNet 임베딩
    # -------------------------
    try:
        emb = wav_to_yamnet_embedding(wav, target_sr, yamnet_model)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "path": wav_path,
            "stage": "yamnet_failed",
            "true_label": true_label,
            "rule_score": rule_score,
            "pred_label": "MLP_FAILED",
            "pred_prefix": None,
            "pred_prob": None,
            "reason": f"yamnet_failed ({e})",
            "elapsed": elapsed,
        }

    # -------------------------
    # Stage 2: MLP 멀티클래스
    # -------------------------
    with torch.no_grad():
        x = torch.from_numpy(emb).float().unsqueeze(0).to(device)
        logits = mlp_model(x)
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(prob.argmax())
    pred_label = idx_to_label[pred_idx]
    pred_label = unicodedata.normalize("NFC", pred_label)
    pred_prefix = pred_label.split("-")[0]

    elapsed = time.perf_counter() - t0
    return {
        "path": wav_path,
        "stage": "passed",
        "true_label": true_label,
        "rule_score": rule_score,
        "pred_label": pred_label,
        "pred_prefix": pred_prefix,
        "pred_prob": float(prob[pred_idx]),
        "reason": "",
        "elapsed": elapsed,
    }


# ================================
#  4) Confusion matrix 그리기
# ================================
def plot_confusion_matrix(cm, labels, out_path, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()
    tick = np.arange(len(labels))
    plt.xticks(tick, labels, rotation=45)
    plt.yticks(tick, labels)
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm[i, j] > thresh else "black"
            plt.text(j, i, cm[i, j], ha="center", va="center", color=color)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def print_binary_metrics(y_true, y_pred, name=""):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average="binary",
        pos_label=1
    )
    print(f"[{name}] acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")

# ================================
#  5) Post-processor (오프라인 지표용)
# ================================
def apply_postprocessor(df_valid, positive_prefix=POSITIVE_PREFIX,
                        window_size=3, min_positive=2):

    if "group" not in df_valid.columns or "seq_idx" not in df_valid.columns:
        raise ValueError("df_valid에 group / seq_idx 없음")

    parts = []
    for gid, sub in df_valid.groupby("group"):
        sub = sub.sort_values("seq_idx").reset_index(drop=True)
        true_bin = sub["true_label"].apply(
            lambda t: 1 if t.split("-")[0] in positive_prefix else 0
        ).to_numpy()
        pred_bin = sub["pred_prefix"].apply(
            lambda p: 1 if p in positive_prefix else 0
        ).to_numpy()
        post = np.zeros_like(pred_bin)

        n = len(pred_bin)
        if n >= window_size:
            for start in range(0, n - window_size + 1):
                end = start + window_size
                if pred_bin[start:end].sum() >= min_positive:
                    post[start:end] = 1

        sub["true_bin"] = true_bin
        sub["pred_bin"] = pred_bin
        sub["post_pred_bin"] = post
        parts.append(sub)

    return pd.concat(parts, ignore_index=True)


# ================================
#  6) 메인 (🔥 스트리밍 post 포함)
# ================================
def main():
    mlp_model, device, label_to_idx, idx_to_label = load_mlp_model()
    yamnet_model = load_yamnet_model()

    # 🔹 CSV 저장 경로 디렉토리 미리 생성
    os.makedirs(os.path.dirname(OUT_CSV_PATH), exist_ok=True)

    wav_paths = sorted(glob.glob(os.path.join(AUDIO_ROOT, "**", "*.wav"), recursive=True))
    print("총 wav:", len(wav_paths))

    results = []

    # 그룹별 스트리밍 상태 (seq, pred_bin, post_bin)
    group_state = {}  # group -> {"seq": [], "pred": [], "post": []}

    for path in wav_paths:
        rel = os.path.relpath(path, AUDIO_ROOT)
        group = os.path.dirname(rel)

        base = os.path.basename(path)
        seq_idx = get_tail_digits(os.path.splitext(base)[0])

        r = infer_one_file(
            wav_path=path,
            target_sr=TARGET_SR,
            mlp_model=mlp_model,
            device=device,
            idx_to_label=idx_to_label,
            yamnet_model=yamnet_model,
        )

        r["group"] = group
        r["seq_idx"] = seq_idx

        # ================================
        # 🔥 스트리밍 post-processor (3중2)
        # ================================
        st = group_state.setdefault(group, {"seq": [], "pred": [], "post": []})

        # 이 segment의 이진 예측값 (파이프라인 기준)
        if r["stage"] == "passed" and r["pred_prefix"] in POSITIVE_PREFIX:
            pred_bin_cur = 1
        else:
            pred_bin_cur = 0

        st["seq"].append(seq_idx)
        st["pred"].append(pred_bin_cur)

        # 마지막 3개 창만 보고 현재 segment 기준 post 결정
        if len(st["pred"]) >= 3:
            window = st["pred"][-3:]
            post_cur = 1 if sum(window) >= 2 else 0
            print(f"[POST] group={group} seq={seq_idx:02d}  window={window} "
                  f"sum={sum(window)} → post={post_cur}")
        else:
            post_cur = 0

        st["post"].append(post_cur)
        r["post_pred_bin_stream"] = int(post_cur)  # CSV에도 저장

        results.append(r)

        df_tmp = pd.DataFrame(results)
        df_tmp.to_csv(OUT_CSV_PATH, index=False, encoding="utf-8-sig")

        post_flag = f" | post={post_cur}"

        # ================================
        # 콘솔 출력
        # ================================
        if r["stage"] == "passed":
            flag = "🔥ALARM" if r["pred_prefix"] in POSITIVE_PREFIX else "🟢NON"
            print(
                f"➡️ [OK] {base}  rule={r['rule_score']:.3f}  "
                f"pred={r['pred_label']} (p={r['pred_prob']:.3f}) "
                f"true={r['true_label']} {flag}{post_flag}"
            )
        elif r["stage"] == "rule_filtered":
            print(f"🚫 [RULE] {base}  rule={r['rule_score']:.3f} true={r['true_label']}{post_flag}")
        else:
            print(f"⚠️ {r['stage'].upper()} {base}  reason={r.get('reason','')}{post_flag}")

    # ----------------
    # CSV 저장 (엑셀에서 한글 안 깨지게 utf-8-sig)
    # ----------------
    df = pd.DataFrame(results)
    # -------------------------
    # 🔥 stage_final 계산 추가
    # -------------------------
    def compute_stage_final(row):
        # 1) Rule 단계에서 컷
        if row["stage"] == "rule_filtered":
            return "rule_filtered"

        # 2) YAMNet 에러
        if row["stage"] == "yamnet_failed":
            return "mlp_failed"

        # 3) passed → MLP 결과 분석
        true_is_pos = row["true_label"].split("-")[0] in POSITIVE_PREFIX
        pred_is_pos = (row["pred_prefix"] in POSITIVE_PREFIX)

        if true_is_pos and pred_is_pos:
            return "mlp_tp"   # 양성 → 양성
        if true_is_pos and not pred_is_pos:
            return "mlp_fn"   # 양성 → 음성
        if not true_is_pos and pred_is_pos:
            return "mlp_fp"   # 음성 → 양성
        if not true_is_pos and not pred_is_pos:
            return "mlp_tn"   # 음성 → 음성

    
    df["stage_final"] = df.apply(compute_stage_final, axis=1)
    df.to_csv(OUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print("CSV 저장:", OUT_CSV_PATH)

    # ----------------
    # 🔍 파이프라인 FN 분석 + MLP FN 분석
    # ----------------
    def is_pos(label):
        return label.split("-")[0] in POSITIVE_PREFIX

    # GT / Pred (전체 파이프라인 기준)
    df["true_bin_all"] = df["true_label"].apply(lambda t: 1 if is_pos(t) else 0)
    df["pred_bin_all"] = df.apply(
        lambda row: 1 if (row["stage"] == "passed" and is_pos(row["pred_prefix"] or "")) else 0,
        axis=1,
    )

    # 파이프라인 기준 FN = GT 양성인데 최종 0
    fn = df[(df["true_bin_all"] == 1) & (df["pred_bin_all"] == 0)]

    print("\n====================== 파이프라인 FN 분석 ========================")
    print(fn.groupby("stage")["true_label"].value_counts())
    print("=============================================================\n")

    # 🔹 MLP 단계 FN (rule 통과 + YAMNet 성공 + MLP가 S2를 다른 클래스로 예측)
    fn_mlp = df[
        (df["true_bin_all"] == 1) &
        (df["stage"] == "passed") &
        (df["pred_bin_all"] == 0)  # pred_prefix가 S2가 아닌 경우
    ]
    print("🔹 MLP 단계 FN 개수:", len(fn_mlp))
    if not fn_mlp.empty:
        print(fn_mlp["pred_label"].value_counts())

    # ----------------
    # 🔵 평균 처리 시간 출력 (stage별)
    # ----------------
    if "elapsed" in df.columns:
        overall_mean = df["elapsed"].mean()
        print(f"[TIME] 전체 평균: {overall_mean:.3f} s (N={len(df)})")

        mask_rule = df["stage"] == "rule_filtered"
        if mask_rule.any():
            m = df.loc[mask_rule, "elapsed"].mean()
            print(f"[TIME] rule_filtered (rule만 통과, N={mask_rule.sum()}): {m:.3f} s")

        mask_pass = df["stage"] == "passed"
        if mask_pass.any():
            m = df.loc[mask_pass, "elapsed"].mean()
            print(f"[TIME] passed (rule+YAMNet+MLP, N={mask_pass.sum()}): {m:.3f} s")

        mask_yfail = df["stage"] == "yamnet_failed"
        if mask_yfail.any():
            m = df.loc[mask_yfail, "elapsed"].mean()
            print(f"[TIME] yamnet_failed (N={mask_yfail.sum()}): {m:.3f} s")

    # ----------------
    # 성능 계산 (Stage=passed만 → MLP 성능)
    # ----------------
    df_valid = df[df["stage"] == "passed"]
    if df_valid.empty:
        print("성능 계산 불가 (passed 없음)")
        return

    # 멀티클래스 (MLP 단계)
    true_mc = df_valid["true_label"].tolist()
    pred_mc = df_valid["pred_label"].tolist()
    labels = sorted(set(true_mc) | set(pred_mc))

    cm_mc = confusion_matrix(true_mc, pred_mc, labels=labels)
    plot_confusion_matrix(
        cm_mc,
        labels,
        "./result/pipe/cm_multiclass.png",
        title="Multiclass CM (MLP stage only)"
    )
    print("멀티클래스 CM 저장 완료: cm_multiclass.png")

    # 이진(segment, MLP 단계 기준)
    true_bin = [1 if t.split("-")[0] in POSITIVE_PREFIX else 0 for t in true_mc]
    pred_bin = [1 if p.split("-")[0] in POSITIVE_PREFIX else 0 for p in pred_mc]

    cm_bin = confusion_matrix(true_bin, pred_bin, labels=[0, 1])
    plot_confusion_matrix(
        cm_bin,
        ["Negative", "Positive"],
        "./result/pipe/emb_cm_binary.png",
        title="Binary CM (MLP stage only)"
    )
    print_binary_metrics(true_bin, pred_bin, "MLP-only (segment)")

    # 🔵 파이프라인 전체 기준 이진 평가
    cm_bin_all = confusion_matrix(df["true_bin_all"], df["pred_bin_all"], labels=[0, 1])
    plot_confusion_matrix(
        cm_bin_all,
        ["Negative", "Positive"],
        "./result/pipe/emb_cm_binary_pipeline.png",
        title="Binary CM (Full pipeline)"
    )
    print("파이프라인 전체 기준 이진 CM 저장 완료: emb_cm_binary_pipeline.png")
    print_binary_metrics(df["true_bin_all"], df["pred_bin_all"], "Full pipeline")

    # 🔵 파이프라인 + 스트리밍 post-processor 기준 이진 평가
    if "post_pred_bin_stream" in df.columns:
        cm_bin_all_post = confusion_matrix(
            df["true_bin_all"],
            df["post_pred_bin_stream"],
            labels=[0, 1],
        )
        plot_confusion_matrix(
            cm_bin_all_post,
            ["Negative", "Positive"],
            "./result/pipe/emb_cm_binary_pipeline_post.png",
            title="Binary CM (Pipeline + streaming post)"
        )
        print("파이프라인 + postprocessor 기준 이진 CM 저장 완료: emb_cm_binary_pipeline_post.png")
        print_binary_metrics(df["true_bin_all"], df["post_pred_bin_stream"], "Pipeline + streaming post")

    # post-processor (오프라인 성능용, group 기준)
    df_post = apply_postprocessor(df_valid)
    cm_post = confusion_matrix(df_post["true_bin"], df_post["post_pred_bin"], labels=[0, 1])
    plot_confusion_matrix(
        cm_post,
        ["Negative", "Positive"],
        "./result/pipe/emb_cm_binary_post.png",
        title="Binary CM (Offline postprocessor, group-wise)"
    )


if __name__ == "__main__":
    main()
