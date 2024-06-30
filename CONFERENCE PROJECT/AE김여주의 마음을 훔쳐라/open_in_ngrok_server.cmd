# Ngrok 서버를 이용해 외부로 공개할 수 있는 도메인 부여
# Ngrok pro version 기준
# 먼저 Ngrok을 통해 서버를 연 뒤 도메인 주소를 설정한다.
# (중요) Ngrok을 통해 로컬 호스트를 열 수 있도록 도메인을 부여할 땐, 
  ## 반드시 도메인 주소에 ngrok이 들어가야 에러가 나지 않음
# 앞에 localhost.cmd를 통해 로컬 호스트가 열렸다면, 만든 도메인 주소 아래와 같이 입력하고
  ## 로컬 호스트 옆에 뜬 숫자를 아래 domain=' ' 열에 붙인다.

C:\Users\user>cd C:/Program Files/ngrok #파일 경로 설정
C:\Program Files\ngrok>ngrok http --domain=chatwithkimyeoju.kubig.ngrok.io 8501
