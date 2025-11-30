가드이어 (Gard-Ear) 서버 실행 안내서 (OS별)

이 문서는 '가드이어' 백엔드 서버를 라즈베리파이(Linux) 또는 Windows 환경에서 실행하는 방법을 OS별로 구분하여 안내합니다.

실행 순서 (공통)

라이브러리 설치

데이터베이스 생성

환경 변수 설정 (OS별로 방법이 다름)

(선택) 이메일 발송 테스트

메인 서버 실행

대시보드 접속

1. 라이브러리 설치 (공통)

서버 실행에 필요한 Python 라이브러리들을 설치합니다.
(가상환경(venv)을 만들어 그 안에서 설치하는 것을 권장합니다.)

# 터미널에서 실행
pip install -r requirements.txt


2. 데이터베이스 생성 (공통)

서버가 사용할 gard-ear.db 파일을 생성합니다. 이 파일은 SQLite 데이터베이스이며, 모든 설정과 로그를 저장합니다.

# 터미널에서 실행
python database.py


[중요] 이전에 database.py를 실행한 적이 있고, app.py v3 (설치 위치 추가)로 업그레이드했다면, 반드시 기존 gard-ear.db 파일을 삭제한 후 위 명령어를 다시 실행하여 DB 구조를 갱신해야 합니다.

3. 환경 변수 설정 (OS별)

알림 발송에 필요한 민감한 API 키와 계정 정보를 코드에 직접 적는 대신, '환경 변수'로 등록합니다.

[매우 중요!]

환경 변수는 현재 터미널 창(세션)에만 적용됩니다. 터미널을 끄면 다시 설정해야 합니다.

반드시 5단계(서버 실행)와 동일한 터미널 창에서 이 명령어를 실행해야 합니다.

3-A. 🔴 라즈베리파이 (Linux) / macOS

라즈베리파이의 터미널에서는 export 명령어를 사용합니다.

# 1. Gmail 설정 (총 2줄)
export GMAIL_USER="your-email@gmail.com"
export GMAIL_APP_PASSWORD="your-16-digit-app-password"

# 2. NHN Cloud 설정 (총 3줄)
export NHN_APP_KEY="발급받은 AppKey"
export NHN_SECRET_KEY="발급받은 SecretKey"
export NHN_SENDER_NUMBER="인증받은 발신번호 (예: 01012345678)"


3-B. 🔵 윈도우 (Windows)

윈도우는 터미널 종류에 따라 명령어가 다릅니다. 본인의 터미널을 확인하세요.

옵션 1: 명령 프롬프트 (CMD) (프롬프트가 C:\>로 시작)
set 명령어를 사용하고, 큰따옴표(")를 사용하는 것이 안전합니다.

:: 1. Gmail 설정 (총 2줄)
set GMAIL_USER="your-email@gmail.com"
set GMAIL_APP_PASSWORD="your-16-digit-app-password"

:: 2. NHN Cloud 설정 (총 3줄)
set NHN_APP_KEY="발급받은 AppKey"
set NHN_SECRET_KEY="발급받은 SecretKey"
set NHN_SENDER_NUMBER="인증받은 발신번호 (예: 01012345678)"


옵션 2: 파워셸 (PowerShell) (프롬프트가 PS C:\>로 시작)
$env: 명령어를 사용합니다.

# 1. Gmail 설정 (총 2줄)
$env:GMAIL_USER = "your-email@gmail.com"
$env:GMAIL_APP_PASSWORD = "your-16-digit-app-password"

# 2. NHN Cloud 설정 (총 3줄)
$env:NHN_APP_KEY = "발급받은 AppKey"
$env:NHN_SECRET_KEY = "발급받은 SecretKey"
$env:NHN_SENDER_NUMBER = "인증받은 발신번호 (예: 01012345678)"


4. (선택) 이메일 발송 테스트 (공통)

위 3번 단계를 완료한 바로 그 터미널에서 아래 명령어를 실행하여 Gmail 설정이 잘 되었는지 테스트할 수 있습니다.

python test_email.py


스크립트 안내에 따라 테스트 메일을 받을 주소를 입력하고, [성공] 메시지가 뜨는지 확인합니다.

5. 메인 서버 실행 (공통)

위 3번 단계를 완료한 바로 그 터미널에서 메인 서버를 실행합니다.

python app.py


서버가 실행되면 터미널에 http://127.0.0.1:5000 등의 주소가 나타납니다.

6. 대시보드 접속 (공통)

웹 브라우저를 열고 아래 주소로 접속합니다.

서버를 실행한 장치(PC/라즈베리파이)에서 접속할 때:
http://127.0.0.1:5000 또는 http://localhost:5000

같은 네트워크의 다른 장치(스마트폰 등)에서 접속할 때:
http://[서버의_IP주소]:5000
(예: http://192.168.0.27:5000)