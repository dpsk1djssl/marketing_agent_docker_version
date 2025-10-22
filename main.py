# main.py
import os
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from langchain_mcp_adapters.tools import load_mcp_tools


# ==============================
# 설정
# ==============================
SYSTEM_PROMPT = """
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
GREETING = "마케팅이 필요한 가맹점을 알려주세요  \n(조회가능 예시: 동대*, 유유*, 똥파*, 본죽*, 본*, 원조*, 희망*, 혁이*, H커*, 케키*)"


# ==============================
# 인메모리 히스토리 저장소
# ==============================
_STORE: Dict[str, ChatMessageHistory] = {}

def get_history(session_id: str) -> ChatMessageHistory:
    """세션별 히스토리 반환. 초기에는 빈 히스토리."""
    hist = _STORE.get(session_id)
    if hist is None:
        hist = ChatMessageHistory()
        # 초기에는 아무것도 추가하지 않음!
        _STORE[session_id] = hist
    return hist

def trim_history(hist: ChatMessageHistory, max_pairs: int = 6) -> None:
    """토큰 절약용: 최근 max_pairs 쌍만 유지."""
    msgs = hist.messages
    if len(msgs) > max_pairs * 2:
        # 최근 N개 대화쌍만 유지
        hist.messages = msgs[-max_pairs * 2:]


# ==============================
# FastAPI 앱
# ==============================
app = FastAPI(title="Merchant Marketing Agent API (Contest Demo)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    user_message: str


class ResetRequest(BaseModel):
    session_id: str


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.post("/reset")
async def reset(req: ResetRequest):
    _STORE.pop(req.session_id, None)
    return {"ok": True}


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    - 인메모리 세션별 히스토리 유지
    - MCP stdio 세션은 요청마다 안전하게 열고 닫음
    - LangGraph ReAct agent + RunnableWithMessageHistory
    """
    if "GOOGLE_API_KEY" not in os.environ:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is not set in environment variables.")

    # LLM 인스턴스
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.1,
    )

    # MCP 서버 파라미터
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
        env=None
    )

    # 요청마다 MCP 세션 안전하게 생성/종료
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as mcp_session:
            try:
                await mcp_session.initialize()
                tools = await load_mcp_tools(mcp_session)
                
                # ReAct agent 생성 (SystemMessage는 state_modifier로 전달!)
                agent = create_react_agent(
                    llm, 
                    tools,
                    state_modifier=SYSTEM_PROMPT
                )

                # RunnableWithMessageHistory 구성
                with_history = RunnableWithMessageHistory(
                    agent,
                    get_session_history=lambda sid: get_history(sid),
                    input_messages_key="messages",
                    history_messages_key="messages",
                )

                # 첫 메시지 확인 (ainvoke 전에!)
                hist = get_history(req.session_id)
                is_first_message = len(hist.messages) == 0

                # Agent 실행
                result = await with_history.ainvoke(
                    {"messages": [HumanMessage(content=req.user_message)]},
                    config={"configurable": {"session_id": req.session_id}},
                )

                # 응답 받은 후 토큰 절약
                trim_history(hist, max_pairs=6)

                # AI 응답 추출
                reply = result["messages"][-1].content
                
                # 첫 메시지인 경우 GREETING 추가
                if is_first_message:
                    reply = GREETING + "\n\n" + reply

                return {"reply": reply}

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Agent/MCP error: {e!r}")