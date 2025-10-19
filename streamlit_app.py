import streamlit as st
import asyncio

from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from PIL import Image
from pathlib import Path

# 환경변수
ASSETS = Path("assets")
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

system_prompt = """
당신은 신한카드 빅데이터 기반의 전문 마케팅 컨설턴트입니다. 사용자의 요청을 분석하여 아래의 절차에 따라 임무를 수행합니다.

### 절차 1: 가맹점 검색 및 특정
1. 사용자가 가맹점 이름이나 ID를 입력하면, 가장 먼저 `search_merchants` 도구를 사용해 가맹점을 검색합니다.
2. **검색 결과가 1개**이면, 해당 가맹점을 대상으로 바로 아래 '절차 2'를 진행합니다.
3. **검색 결과가 여러 개**이면, 사용자에게 "어떤 가맹점을 분석할까요?"라고 질문하며 번호와 함께 `가맹점명`, `가맹점주소` 목록을 보여줍니다. 사용자가 번호나 가맹점 ID로 특정하면, 그 가맹점을 대상으로 '절차 2'를 진행합니다.
4. **검색 결과가 0개**이면, "해당하는 가맹점을 찾을 수 없습니다."라고 답변합니다.

### 절차 2: 임무 결정 및 분석 수행
성공적으로 하나의 가맹점이 특정되면, 사용자의 최초 질문 의도에 따라 아래 두 임무 중 하나를 수행합니다.

- **임무 1: 간단 채널 추천 (Q1)**
  - **조건**: 사용자가 단순히 '채널 추천'이나 가벼운 마케팅 문의를 했을 경우.
  - **수행**: `recommend_channels` 도구를 사용하여 가맹점의 '슈머유형'과 'A_STAGE'에 맞는 핵심 채널을 추천하는 간단한 보고서를 작성합니다.

- **임무 2: 재방문율 저하 원인 심층 분석 (Q2)**
  - **조건**: 사용자가 '재방문율', '심층 분석', '문제점 진단' 등의 키워드를 사용하여 상세 분석을 요청했을 경우.
  - **수행**: 기획서의 분석 절차에 따라 다음 3단계를 수행합니다.
      1. **데이터 확보 (STEP 1)**: `analyze_low_revisit_store` 도구를 호출하여 해당 가맹점의 7P 분석 데이터(백분위 순위)를 확보합니다. (값은 0~1 사이, 1에 가까울수록 우수)
      2. **원인 분석 (STEP 2)**: 확보된 데이터를 근거로 Product, Price, Place, Process 각 영역에서 강점과 약점을 진단합니다. 특히 백분위 순위가 0.3 미만인 항목을 '핵심 문제점'으로 식별합니다.
      3. **마케팅 제안 (STEP 3)**: 진단된 문제점을 해결하기 위해, 각 P 영역별로 구체적인 마케팅 아이디어를 제시합니다.

### 절차 3: 보고서 생성
- 모든 분석 결과는 아래 보고서 구조를 반드시 따르는 Markdown 형식으로 제공합니다.
- `# 요약 → ## 핵심 인사이트(불릿) → ## 추천 전략 및 채널(표) → ## 실행 가이드(불릿) → ## 데이터 근거(표)`
"""
greeting = "마케팅이 필요한 가맹점을 알려주세요  \n(조회가능 예시: 동대*, 유유*, 똥파*, 본죽*, 본*, 원조*, 희망*, 혁이*, H커*, 케키*)"

# Streamlit App UI
@st.cache_data 
def load_image(name: str):
    return Image.open(ASSETS / name)

st.set_page_config(page_title="2025년 빅콘테스트 AI데이터 활용분야 - 맛집을 수호하는 AI비밀상담사")

def clear_chat_history():
    st.session_state.messages = [SystemMessage(content=system_prompt), AIMessage(content=greeting)]

# 사이드바
with st.sidebar:
    st.image(load_image("shc_ci_basic_00.png"), width='stretch')
    st.markdown("<p style='text-align: center;'>2025 Big Contest</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI DATA 활용분야</p>", unsafe_allow_html=True)
    st.write("")
    col1, col2, col3 = st.columns([1,2,1])  # 비율 조정 가능
    with col2:
        st.button('Clear Chat History', on_click=clear_chat_history)

# 헤더
st.title("신한카드 소상공인 🔑 비밀상담소")
st.subheader("#우리동네 #숨은맛집 #소상공인 #마케팅 #전략 .. 🤤")
st.image(load_image("image_gen3.png"), width='stretch', caption="🌀 머리아픈 마케팅 📊 어떻게 하면 좋을까?")
st.write("")

# 메시지 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=system_prompt),
        AIMessage(content=greeting)
    ]

# 초기 메시지 화면 표시
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

def render_chat_message(role: str, content: str):
    with st.chat_message(role):
        st.markdown(content.replace("<br>", "  \n"))

# LLM 모델 선택
llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # 최신 Gemini 2.5 Flash 모델
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1
    )

# MCP 서버 파라미터(환경에 맞게 명령 수정)
server_params = StdioServerParameters(
    command="uv",
    args=["run","mcp_server.py"],
    env=None
)

# 사용자 입력 처리
async def process_user_input():
    """사용자 입력을 처리하는 async 함수"""
    async with stdio_client(server_params) as (read, write):
        # 스트림으로 ClientSession을 만들고
        async with ClientSession(read, write) as session:
            # 세션을 initialize 한다
            await session.initialize()

            # MCP 툴 로드
            tools = await load_mcp_tools(session)

            # 에이전트 생성
            agent = create_react_agent(llm, tools)

            # 에이전트에 전체 대화 히스토리 전달
            agent_response = await agent.ainvoke({"messages": st.session_state.messages})
            
            # AI 응답을 대화 히스토리에 추가
            ai_message = agent_response["messages"][-1]  # 마지막 메시지가 AI 응답

            return ai_message.content
            

# 사용자 입력 창
if query := st.chat_input("가맹점 이름을 입력하세요"):
    # 사용자 메시지 추가
    st.session_state.messages.append(HumanMessage(content=query))
    render_chat_message("user", query)

    with st.spinner("Thinking..."):
        try:
            # 사용자 입력 처리
            reply = asyncio.run(process_user_input())
            st.session_state.messages.append(AIMessage(content=reply))
            render_chat_message("assistant", reply)
        except* Exception as eg:
            # 오류 처리
            for i, exc in enumerate(eg.exceptions, 1):
                error_msg = f"오류가 발생했습니다 #{i}: {exc!r}"
                st.session_state.messages.append(AIMessage(content=error_msg))
                render_chat_message("assistant", error_msg)
