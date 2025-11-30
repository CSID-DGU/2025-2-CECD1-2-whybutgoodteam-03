import os
# 1. 라이브러리 충돌 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
matplotlib.use('Agg')

import time
import pyaudio
import wave
import requests
import collections
import torch
import numpy as np
import tensorflow_hub as hub
import csv
from datetime import datetime

# 팀원 모듈 임포트
import pipeline_mul
from pipeline_mul import load_mlp_model, infer_one_file, POSITIVE_PREFIX

# --- 설정 ---
SERVER_URL = "http://127.0.0.1:5000/api/events"
MODEL_PATH = "./yamnet_mlp_best.pt"
YAMNET_HANDLE = "./yamnet_local"

# 경로 설정 (절대 경로로 자동 변환)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.path.join(BASE_DIR, "records")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "detection_log.csv")

# ==========================================
# 🔧 [핵심 수정] 녹음용 vs AI용 주파수 분리
# ==========================================
# 44100이 안되면 48000으로 바꿔보세요! (대부분의 USB 마이크는 44100 지원)
MIC_RATE = 48000   
MODEL_RATE = 16000 

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_SECONDS = 4.0 

# 후처리용 큐
prediction_queue = collections.deque(maxlen=3)

def init_system():
    if not os.path.exists(RECORD_DIR): os.makedirs(RECORD_DIR)
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            header = ["timestamp", "filename", "stage", "rule_score", "pred_label", "pred_prob", "is_fire", "reason", "elapsed"]
            writer.writerow(header)

    pipeline_mul.MLP_BEST_MODEL = MODEL_PATH 
    pipeline_mul.YAMNET_MODEL_HANDLE = YAMNET_HANDLE

def save_log_to_csv(result_dict, is_fire):
    try:
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            row = [
                result_dict.get("timestamp", ""),
                os.path.basename(result_dict.get("path", "")),
                result_dict.get("stage", ""),
                f"{result_dict.get('rule_score', 0):.4f}",
                result_dict.get("pred_label", ""),
                f"{result_dict.get('pred_prob', 0):.4f}" if result_dict.get('pred_prob') else "",
                is_fire,
                result_dict.get("reason", ""),
                f"{result_dict.get('elapsed', 0):.4f}"
            ]
            writer.writerow(row)
    except Exception:
        pass 

def send_alert_to_server():
    try:
        requests.post(SERVER_URL, json={"event_type": "fire_alarm_detected"}, timeout=2)
        print("🚨 [서버 전송 완료]")
    except:
        print("❌ [서버 전송 실패]")

def main():
    print("\n=== 🔥 가드이어 순차 감지기 (Fixed Rate Ver.) 시작 ===")
    init_system()
    
    print("⏳ 모델 로딩 중...")
    try:
        mlp_model, device, label_to_idx, idx_to_label = load_mlp_model()
        yamnet_model = hub.load(YAMNET_HANDLE)
        print("✅ 모델 준비 완료!")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return

    p = pyaudio.PyAudio()
    stream = None
    
    try:
        # [수정] MIC_RATE 사용
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=MIC_RATE, input=True, frames_per_buffer=CHUNK, start=False)
        print(f"🎤 마이크 설정 완료: {MIC_RATE}Hz")
        
        while True:
            # 1. 녹음 (MIC_RATE로 녹음)
            print(f"\n🎤 녹음 중... (4초)")
            stream.start_stream()
            
            frames = []
            # MIC_RATE 기준으로 프레임 수 계산
            for _ in range(0, int(MIC_RATE / CHUNK * RECORD_SECONDS)):
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                except IOError:
                    break
            
            stream.stop_stream()
            
            # 2. 저장 (MIC_RATE로 저장)
            now = datetime.now()
            timestamp_str = now.strftime("%Y%m%d_%H%M%S")
            wav_filename = os.path.join(RECORD_DIR, f"{timestamp_str}.wav")
            
            wf = wave.open(wav_filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(MIC_RATE) # 파일 헤더에 44100Hz라고 기록
            wf.writeframes(b''.join(frames))
            wf.close()
            
            # 3. AI 분석 (MODEL_RATE=16000으로 요청)
            print(f"🧠 분석 중...")
            
            # 여기서 중요! infer_one_file 내부의 librosa가 
            # 44100Hz 파일을 읽어서 자동으로 16000Hz로 변환해줍니다.
            result = infer_one_file(
                wav_path=wav_filename,
                target_sr=MODEL_RATE,  # 16000
                mlp_model=mlp_model,
                device=device,
                idx_to_label=idx_to_label,
                yamnet_model=yamnet_model
            )
            
            # 4. 결과 처리
            is_fire = 0
            if result["stage"] == "passed" and result["pred_prefix"] in POSITIVE_PREFIX:
                is_fire = 1
                print(f"⚠️  [화재 감지!] {result['pred_label']} ({result['pred_prob']:.2f})")
            elif result["stage"] == "rule_filtered":
                print(f"💤  [조용함]")
            else:
                print(f"ℹ️  [일반 소음] {result.get('pred_label')}")
            
            result["timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S")
            save_log_to_csv(result, is_fire)
            
            prediction_queue.append(is_fire)
            if len(prediction_queue) == 3 and sum(prediction_queue) >= 2:
                print("\n🔥🔥🔥 [확정] 화재 경보 발송!!! 🔥🔥🔥")
                send_alert_to_server()
                prediction_queue.clear()
                time.sleep(3)

    except KeyboardInterrupt:
        print("\n👋 종료합니다.")
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
    finally:
        if stream: stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
