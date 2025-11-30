import sqlite3
import os
import atexit
from flask import Flask, jsonify, request, render_template, g
from flask_cors import CORS
from notifications import send_notification_task

# --- 상수 정의 ---
DATABASE = 'gard-ear.db'
DEVICE_ID = 'rasp_pi_main' # 이 서버는 이 장치 하나만 관리

# --- Flask 앱 설정 ---
app = Flask(__name__, template_folder='.') # index.html을 같은 폴더에서 찾음
CORS(app)
app.config['DATABASE'] = DATABASE
app.config['DEBUG'] = True

# --- 데이터베이스 연결 ---
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(app.config['DATABASE'])
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# --- 헬퍼 함수 ---
def query_db(query, args=(), one=False):
    try:
        cur = get_db().execute(query, args)
        rv = [dict(row) for row in cur.fetchall()]
        cur.close()
        return (rv[0] if rv else None) if one else rv
    except sqlite3.Error as e:
        print(f"Database query error: {e}")
        return None

def execute_db(query, args=()):
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute(query, args)
        db.commit()
        cur.close()
        return True
    except sqlite3.Error as e:
        print(f"Database execute error: {e}")
        db.rollback()
        return False

# --- API 엔드포인트 ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """
    현재 장치 상태와 시스템 설정을 반환합니다.
    (v4: camera_url 제거)
    """
    status_data = query_db("SELECT status FROM DeviceStatus WHERE device_id = ?", [DEVICE_ID], one=True)
    settings_data = query_db("SELECT location FROM SystemSettings WHERE id = 1", one=True)
    
    if not status_data:
        execute_db("INSERT INTO DeviceStatus (device_id, status) VALUES (?, ?)", [DEVICE_ID, 'normal'])
        status_data = {'status': 'normal'}

    return jsonify({
        'status': status_data.get('status', 'normal'),
        'location': settings_data.get('location', '미설정') if settings_data else '미설정'
    })

@app.route('/api/events', methods=['POST'])
def create_event():
    print("화재 감지 신호 수신 (POST /api/events)")
    execute_db("UPDATE DeviceStatus SET status = 'alert' WHERE device_id = ?", [DEVICE_ID])
    settings = query_db("SELECT location FROM SystemSettings WHERE id = 1", one=True)
    location = settings.get('location', '미설정') if settings else '미설정'
    execute_db("INSERT INTO Events (device_id, event_type, location) VALUES (?, 'fire_alarm_detected', ?)", [DEVICE_ID, location])
    
    print("알림 발송 스레드 시작...")
    import threading
    threading.Thread(target=send_notification_task, args=(location,)).start()
    
    return jsonify({'message': 'Event received and alert triggered.'}), 201

@app.route('/api/acknowledge', methods=['POST'])
def acknowledge_event():
    print("경보 해제 신호 수신 (POST /api/acknowledge)")
    execute_db("UPDATE DeviceStatus SET status = 'normal' WHERE device_id = ?", [DEVICE_ID])
    return jsonify({'message': 'Alert acknowledged and status reset to normal.'}), 200

@app.route('/api/events', methods=['GET'])
def get_events():
    events = query_db("SELECT * FROM Events ORDER BY timestamp DESC")
    return jsonify(events)

@app.route('/api/logs', methods=['GET'])
def get_logs():
    logs = query_db("SELECT * FROM NotificationLogs ORDER BY timestamp DESC")
    return jsonify(logs)

@app.route('/api/system-settings', methods=['GET', 'POST'])
def system_settings():
    """
    (v4: camera_url 제거)
    """
    if request.method == 'POST':
        data = request.json
        location = data.get('location')
        
        execute_db("UPDATE SystemSettings SET location = ? WHERE id = 1", [location])
        print(f"시스템 설정 저장: 위치={location}")
        return jsonify({'message': 'Settings updated successfully.'}), 200
    
    # GET
    settings = query_db("SELECT location FROM SystemSettings WHERE id = 1", one=True)
    return jsonify(settings if settings else {'location': ''})

@app.route('/api/notification-contacts', methods=['GET', 'POST'])
def manage_contacts():
    if request.method == 'POST':
        data = request.json
        name = data.get('name')
        phone = data.get('phone')
        email = data.get('email')
        execute_db("INSERT INTO NotificationContacts (name, phone, email) VALUES (?, ?, ?)", [name, phone, email])
        print(f"새 연락처 추가: {name}")
        return jsonify({'message': 'Contact added.'}), 201
    
    # GET
    contacts = query_db("SELECT * FROM NotificationContacts ORDER BY name")
    return jsonify(contacts)

@app.route('/api/notification-contacts/<int:contact_id>', methods=['PATCH', 'DELETE'])
def manage_contact_detail(contact_id):
    if request.method == 'DELETE':
        execute_db("DELETE FROM NotificationContacts WHERE id = ?", [contact_id])
        print(f"연락처 삭제: ID={contact_id}")
        return jsonify({'message': 'Contact deleted.'}), 200
    
    if request.method == 'PATCH':
        data = request.json
        is_active = data.get('is_active')
        execute_db("UPDATE NotificationContacts SET is_active = ? WHERE id = ?", [is_active, contact_id])
        print(f"연락처 상태 변경: ID={contact_id}, 활성={is_active}")
        return jsonify({'message': 'Contact status updated.'}), 200

# --- 서버 시작 ---
if __name__ == '__main__':
    if not os.path.exists(DATABASE):
        print(f"{DATABASE} 파일을 찾을 수 없습니다. 'database.py'를 실행하여 DB를 생성하세요.")
        print(f"'database.py'를 자동으로 실행하여 DB를 생성합니다...")
        import database
        database.create_tables()
        print("DB 생성 완료.")
        
    print(f"가드이어 서버가 http://127.0.0.1:5000 에서 실행됩니다.")
    print(f"같은 네트워크의 다른 장치에서 접속하려면 http://[라즈베리파이_IP]:5000 으로 접속하세요.")
    app.run(host='0.0.0.0', port=5000)