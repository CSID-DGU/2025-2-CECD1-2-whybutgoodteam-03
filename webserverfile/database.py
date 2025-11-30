import sqlite3
import os

DATABASE = 'gard-ear.db'

def create_tables():
    """
    모든 테이블을 생성합니다.
    v5 변경 사항: NotificationLogs의 status에 'pending' 추가
    """
    # DB 파일이 이미 있으면 삭제 (구조 변경 적용을 위해)
    if os.path.exists(DATABASE):
        print(f"기존 {DATABASE} 파일을 삭제합니다.")
        os.remove(DATABASE)

    print(f"{DATABASE} 파일을 새로 생성합니다...")
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    # 1. 장치 상태 테이블
    c.execute('''
    CREATE TABLE DeviceStatus (
        device_id TEXT PRIMARY KEY,
        status TEXT NOT NULL DEFAULT 'normal' CHECK(status IN ('normal', 'alert'))
    )
    ''')
    c.execute("INSERT INTO DeviceStatus (device_id, status) VALUES ('rasp_pi_main', 'normal')")

    # 2. 화재 이벤트 로그 테이블
    c.execute('''
    CREATE TABLE Events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        device_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        location TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 3. 알림 연락처 테이블
    c.execute('''
    CREATE TABLE NotificationContacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        phone TEXT,
        email TEXT,
        is_active BOOLEAN NOT NULL DEFAULT 1
    )
    ''')

    # 4. 알림 발송 내역 테이블 (v5: pending 추가)
    c.execute('''
    CREATE TABLE NotificationLogs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recipient_name TEXT NOT NULL,
        type TEXT NOT NULL CHECK(type IN ('email', 'sms')),
        status TEXT NOT NULL CHECK(status IN ('success', 'failed', 'pending')), 
        message TEXT,
        error_message TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 5. 시스템 설정 테이블
    c.execute('''
    CREATE TABLE SystemSettings (
        id INTEGER PRIMARY KEY CHECK(id = 1),
        location TEXT DEFAULT '미설정'
    )
    ''')
    c.execute("INSERT INTO SystemSettings (id) VALUES (1)")


    conn.commit()
    conn.close()
    print(f"데이터베이스 테이블 생성이 완료되었습니다. ({DATABASE})")

if __name__ == '__main__':
    create_tables()