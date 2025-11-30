# prefilter.py
import numpy as np
from scipy.signal import welch


# =========================
# Stage 0: 룰 기반 프리필터
# =========================

def compute_rms_db(wav: np.ndarray) -> float:
    """
    RMS를 dBFS-ish 로 변환 (0~약 -무한대)
    """
    eps = 1e-12
    rms = np.sqrt(np.mean(wav ** 2)) + eps
    db = 20 * np.log10(rms)
    return db


def band_energy_ratio(
    wav: np.ndarray,
    sr: int = 16000,
    band: tuple = (600, 2500),
    nperseg: int = 1024,
) -> float:
    """
    특정 주파수 대역 에너지 / 전체 에너지
    """
    f, Pxx = welch(wav, fs=sr, nperseg=min(nperseg, len(wav)))
    total = np.sum(Pxx) + 1e-12
    band_mask = (f >= band[0]) & (f <= band[1])
    band_power = np.sum(Pxx[band_mask])
    return float(band_power / total)


def simple_periodicity_score(
    wav: np.ndarray,
    sr: int = 16000,
    min_period: float = 0.3,
    max_period: float = 3.0,
) -> float:
    """
    아주 싼 자기상관 기반 주기성 힌트 (0~1 정도 값)
    - min_period ~ max_period 사이에서 max normalized autocorr를 스코어로 사용
    """
    wav = wav.astype(np.float32)
    # DC 제거
    x = wav - np.mean(wav)
    if np.allclose(x, 0):
        return 0.0

    # 자기상관 (full)
    ac = np.correlate(x, x, mode="full")
    ac = ac[len(ac) // 2:]  # 양의 lag만

    # 관심 구간 인덱스
    min_lag = int(min_period * sr)
    max_lag = int(max_period * sr)
    max_lag = min(max_lag, len(ac) - 1)
    if max_lag <= min_lag:
        return 0.0

    ac_seg = ac[min_lag:max_lag]
    # 정규화: lag=0일 때 값으로 나누고, 0~1로 클램핑
    ac0 = ac[0] + 1e-12
    norm_ac_seg = ac_seg / ac0
    score = float(np.clip(np.max(norm_ac_seg), 0.0, 1.0))
    return score


def rule_prefilter(
    wav: np.ndarray,
    sr: int = 16000,
    min_db: float = -35.0,
    band: tuple = (600, 2500),
) -> float:
    """
    Stage 0 룰 기반 프리필터 점수 (0~1)
    - loudness (dB)
    - 관심 대역(600~2500Hz) 에너지 비율
    - 간단한 주기성 힌트
    """
    db = compute_rms_db(wav)
    band_ratio = band_energy_ratio(wav, sr=sr, band=band)
    periodicity = simple_periodicity_score(wav, sr=sr)

    # loudness 스코어: min_db 밑이면 0, -10dB면 1 근처, 이런 느낌으로 스케일링
    # (대충 linear, 필요하면 나중에 튜닝)
    loud_score = np.clip((db - min_db) / 10.0, 0.0, 1.0)

    # 가중합 (임시 값, 네가 튜닝)
    w_loud = 0.5
    w_band = 0.3
    w_per = 0.2

    score = w_loud * loud_score + w_band * band_ratio + w_per * periodicity
    return float(np.clip(score, 0.0, 1.0))


# =========================
# Stage 1: RF용 피처 추출
# =========================

def zero_crossing_rate(wav: np.ndarray) -> float:
    """
    간단한 ZCR (0~대략 1)
    """
    wav = wav.astype(np.float32)
    signs = np.sign(wav)
    # 0은 이전 값으로 채워서 너무 많이 깎이지 않게
    signs[signs == 0] = 1
    zc = np.mean(np.abs(np.diff(signs)) / 2.0)
    return float(zc)


def spectral_features(
    wav: np.ndarray,
    sr: int = 16000,
    nperseg: int = 1024,
) -> dict:
    """
    welch 기반 간단 스펙트럴 피처
    - bands: [0-300], [300-600], [600-2500], [2500-8000]
    - spectral centroid
    - spectral rolloff (0.85)
    """
    f, Pxx = welch(wav, fs=sr, nperseg=min(nperseg, len(wav)))
    Pxx = Pxx + 1e-12
    total = np.sum(Pxx)

    def band_ratio_range(f_lo, f_hi):
        m = (f >= f_lo) & (f < f_hi)
        return float(np.sum(Pxx[m]) / total)

    band_0_300 = band_ratio_range(0, 300)
    band_300_600 = band_ratio_range(300, 600)
    band_600_2500 = band_ratio_range(600, 2500)
    band_2500_8000 = band_ratio_range(2500, 8000)

    # centroid
    centroid = float(np.sum(f * Pxx) / total)

    # rolloff (0.85)
    cumsum = np.cumsum(Pxx)
    target = 0.85 * total
    idx = np.searchsorted(cumsum, target)
    idx = np.clip(idx, 0, len(f) - 1)
    rolloff = float(f[idx])

    return {
        "band_0_300": band_0_300,
        "band_300_600": band_300_600,
        "band_600_2500": band_600_2500,
        "band_2500_8000": band_2500_8000,
        "band_600_2500": band_600_2500,
        "centroid": centroid,
        "rolloff": rolloff,
    }


def extract_rf_features(
    wav: np.ndarray,
    sr: int = 16000,
    min_db: float = -35.0,
) -> np.ndarray:
    """
    RF 프리필터용 feature vector 하나 뽑기
    - rms_db
    - rule_prefilter_score
    - zcr
    - 각 band ratio
    - spectral centroid, rolloff
    """
    rms_db = compute_rms_db(wav)
    rule_score = rule_prefilter(wav, sr=sr, min_db=min_db)
    zcr = zero_crossing_rate(wav)
    spec = spectral_features(wav, sr=sr)

    feats = [
        rms_db,
        rule_score,
        zcr,
        spec["band_0_300"],
        spec["band_300_600"],
        spec["band_600_2500"],
        spec["band_2500_8000"],
        spec["centroid"],
        spec["rolloff"],
    ]
    return np.array(feats, dtype=np.float32)
