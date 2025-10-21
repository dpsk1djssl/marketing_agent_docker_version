# KAIT 주최 및 신한카드 주관 - 2025 빅콘테스트 AI데이터 활용분야
### "내 가게를 살리는 AI 비밀상담사 - 가맹점별 찰떡 마케팅 전략을 찾아라" 예시 코드

<br>

## 설명

- 본 코드는 대회 참가자 분들의 과제에 대한 이해와 보다 수월한 접근을 돕기 위해 작성되었습니다.
- 모범답안이 아니며 PoC(Proof of Concept) 수준의 샘플 코드입니다.
- 크게 다음 두 부분으로 구성되어 있습니다.
  1) LLM을 활용하여 데이터에 기반한 응답을 만드는 부분
  2) Streamlit을 활용하여 웹페이지 UI를 만드는 부분
- 대회 진행을 위해 본 코드를 자유롭게 사용하실 수 있으며, 본 코드의 사용 여부는 평가와 무관합니다.

<br>

## 실행 결과 (서비스 웹 페이지)

https://shcard2025bigcontest.streamlit.app/

- 위 웹 페이지는 Streamlit Cloud를 이용해 배포되었습니다.
- Streamlit Cloud를 활용하면 별도 서버없이 무료로 웹페이지 배포가 가능하며, 대회 진행을 위해 사용하셔도 좋습니다.
- 참고 : https://streamlit.io/cloud

<br>

## 로컬 개발 환경 구성 방법

```bash
# On macOS and Linux.
# git소스 복사하기
git clone https://github.com/dpsk1djssl/marketing_agent
cd shcard_2025_bigcontest

# venv 환경 설정 (사전에 uv 설치가 필요합니다. 아래 항목 참조)
uv venv
source .venv/bin/activate

# 필요한 python library 설치
uv pip install -r requirements.txt

# streamlit 환경 변수 저장용 폴더 생성 + GOOGLE_API_KEY환경 변수 파일 생성
# (Google API KEY)는 Google AI Studio에서 무료로 생성 가능 (아래 항목 참조)
mkdir .streamlit
echo 'GOOGLE_API_KEY="(Google API KEY)"' > .streamlit/secrets.toml

# 로컬에서 실행
uv run streamlit run streamlit_app.py
```

```bat
:: On Windows
:: git 소스 복사하기
git clone https://github.com/dpsk1djssl/marketing_agent
cd shcard_2025_bigcontest

:: venv 환경 설정 (사전에 uv 설치가 필요합니다. 아래 항목 참조)
uv venv
call .venv\Scripts\activate.bat

:: 필요한 python library 설치
uv pip install -r requirements.txt

:: streamlit 환경 변수 저장용 폴더 생성 + GOOGLE_API_KEY 환경 변수 파일 생성
:: (Google API KEY)는 Google AI Studio에서 무료로 생성 가능 (아래 항목 참조)
mkdir .streamlit
echo GOOGLE_API_KEY="(Google API KEY)" > .streamlit\secrets.toml

:: 로컬에서 실행
uv run streamlit run streamlit_app.py
```

<br>

## uv 설치 방법

https://docs.astral.sh/uv/getting-started/installation/ (공식 사이트. OS에 맞게 설치하면 됩니다.)

<br>

## Google AI Studio API KEY 생성 방법

https://aistudio.google.com/apikey 접속 후 (Google 로그인 필요) Get API KEY 메뉴에서 생성하면 됩니다.
