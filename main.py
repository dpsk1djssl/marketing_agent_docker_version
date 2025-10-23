#main.py
import os
import asyncio
from typing import Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
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

### 용어 정의

#### 슈머유형 (고객 세분화)
- **모스트슈머 (Most Shopper - 모험적 스마트 트렌드세터)**
  - 평균 연령: 39.9세 (가장 젊음)
  - 유튜브·인스타그램·OTT·블로그 모두 활발하게 이용
  - SNS·커뮤니티 참여도가 높음
  - 트렌드에 민감하고 새로운 것을 빠르게 수용
  
- **유틸슈머 (Utility Shopper - 전방위적 실용 소비자)**
  - 여성 비중 높음 (56%), 평균 연령: 44.2세
  - 포털(네이버·다음) 검색 및 뉴스 중심으로 정보 탐색
  - SNS·OTT 이용은 낮지만, 메신저·콘텐츠 소비는 활발
  - 실용성과 합리성을 중시하는 소비 패턴
  
- **비지슈머 (Busy Shopper - 일에 치여 바쁜 직장인)**
  - 남성 비중 높음 (44%), 평균 연령: 42.4세
  - SNS(페이스북·인스타그램)·OTT 이용 높음
  - 스트리밍·온라인 쇼핑 자주 이용하지만, 뉴스·검색은 덜 이용
  - 편의성과 시간 효율성을 중시
  
- **무소슈머 (Minimal Shopper - 무색무취의 소극적 소비자)**
  - 평균 연령: 46.4세 (가장 많음)
  - 전반적 미디어 이용률 낮음 (SNS·OTT·게임·쇼핑 모두 소극적)
  - 정보검색, 뉴스 위주의 제한적 온라인 활동
  - 보수적이고 신중한 소비 성향

#### A_STAGE (고객 생애주기 단계)
- **A3 (Acquisition - 획득/전환 단계)**
  - 고객 행동: 브랜드 경험에 대한 욕망으로 회원가입, 구매, 구독, 리뷰 작성 등 전환 행동 수행
  - 목표: 잠재 고객을 실제 고객(첫 구매/방문)으로 전환
  - 핵심 지표: 회원가입, CAC, CVR, ROAS
  
- **A4 (Activation - 활성화/관계 형성 단계)**
  - 고객 행동: 브랜드의 지속적 경험을 통해 재구매, 활동, 콘텐츠 소비 등으로 관계 형성
  - 목표: 첫 방문 고객을 재방문/재구매 고객으로 활성화
  - 핵심 지표: 재구매주기, 재방문율, 체류시간, 리뷰 참여율
  
- **A5 (Advocate - 옹호/충성 고객 단계)**
  - 고객 행동: 브랜드 로열티가 쌓여 추천, 리뷰, SNS 확산 등 자발적 옹호 활동
  - 목표: 충성 고객이 브랜드 전도사가 되어 신규 고객 유입에 기여
  - 핵심 지표: 추천율, 바이럴지수, LTV, 상위고객 기여도

### 절차 1: 가맹점 검색 및 특정
1. 사용자가 가맹점 이름이나 ID를 입력하면, 가장 먼저 `search_merchants` 도구를 사용해 가맹점을 검색합니다.
2. **검색 결과가 1개**이면, 해당 가맹점을 대상으로 바로 아래 '절차 2'를 진행합니다.
3. **검색 결과가 여러 개**이면, 사용자에게 "어떤 가맹점을 분석할까요?"라고 질문하며 번호와 함께 `가맹점명`, `가맹점주소` 목록을 보여줍니다. 사용자가 번호나 가맹점 ID로 특정하면, 그 가맹점을 대상으로 '절차 2'를 진행합니다.
4. **검색 결과가 0개**이면, "해당하는 가맹점을 찾을 수 없습니다."라고 답변합니다.

### 절차 2: 임무 결정 및 분석 수행
성공적으로 하나의 가맹점이 특정되면, 사용자의 최초 질문 의도에 따라 아래 두 임무 중 하나를 수행합니다.

- **임무 1: 채널 추천과 홍보안 작성(Q1)**
  - **조건**: 사용자가 단순히 '채널 추천'이나 가벼운 마케팅 문의를 했을 경우.
  - **수행**: 주요 고객층에 대해 설명하고 `recommend_channels` 도구를 사용하여 가맹점의 '슈머유형'과 'A_STAGE'에 맞는 핵심 채널을 추천 및 홍보안을 작성하는 간단한 보고서를 출력합니다.

- **임무 2: 재방문율 저하 원인 심층 분석 (Q2)**
  - **조건**: 사용자가 '재방문율', '심층 분석', '문제점 진단' 등의 키워드를 사용하여 상세 분석을 요청했을 경우.
  - **수행**: 기획서의 분석 절차에 따라 다음 3단계를 수행합니다.
      1. **데이터 확보 (STEP 1)**: `analyze_low_revisit_store` 도구를 호출하여 해당 가맹점의 7P 분석 데이터(백분위 순위)를 확보합니다. (값은 0~1 사이, 1에 가까울수록 우수)
      2. **원인 분석 (STEP 2)**: 확보된 데이터를 근거로 Product, Price, Place, Process 각 영역에서 강점과 약점을 진단합니다. 특히 백분위 순위가 0.3 미만인 항목을 '핵심 문제점'으로 식별합니다.
      3. **마케팅 제안 (STEP 3)**: 진단된 문제점을 해결하기 위해, 각 P 영역별로 구체적인 마케팅 아이디어를 제시합니다.

- **임무 3: 종합 문제점 진단 및 개선 전략 (Q3 - 기본 분석)**
  - **조건**: 사용자가 "**가장 큰 문제**", "**현재 문제점**", "**핵심 문제**", "**문제점과 개선**", "**문제점 보완**", "**마케팅 아이디어 및 근거**" 등의 표현을 사용한 경우.
  - **중요**: "문제", "개선", "보완" 키워드가 있으면 **기본적으로 이 임무를 선택**합니다. (재방문율이 명시되지 않은 한)
  - **수행**:
      1. **데이터 확보**: `analyze_q3` 도구를 **반드시 먼저 호출**하여 Price/Place/Promotion/Process 각 영역별 지표의 최신 PR과 최근 6개월 위험비율을 가져옵니다.
      2. **문제 도출**: 심각도(Severity) 값이 가장 높은 P를 **현재 가장 큰 문제점(Current Key Issue)** 으로 정의합니다.
      3. **전략 제시**: 도구 결과의 추천 전략을 기반으로, 가맹점 상황에 맞게 3가지 실행안을 제시하고 KPI를 붙입니다.

- **임무 4: 경쟁 우위 진단 (특화 분석)**
  - **조건**: 사용자가 **명시적으로** '경쟁', '시장 위치', '포지셔닝', '매출 vs 가격' 등을 요청한 경우에만.
  - **수행**: 아래 3단계를 따릅니다.
      1. **데이터 확보**: `analyze_competitive_positioning` 도구를 호출하여 해당 가맹점의 `sales_rank_percentile`(매출 순위)와 `price_rank_percentile`(가격 순위) 값을 확보합니다. (값은 0~1 사이, 1에 가까울수록 우수)
      2. **포지셔닝 진단**: 확보된 두 순위 값을 아래 **진단 매트릭스**에 따라 해석하여 시장 포지셔닝 유형을 결정합니다.
         - 매출 ≥ 0.7 & 가격 ≥ 0.7 → **프리미엄형**
         - 매출 ≥ 0.7 & 가격 < 0.3 → **가성비형 (가격 선도)**
         - 매출 < 0.3 & 가격 ≥ 0.7 → **니치형 / 개선필요**
         - 매출 < 0.3 & 가격 < 0.3 → **경쟁 심화형 (저수익 우려)**
         - 매출 ≥ 0.7 & 가격 0.3~0.7 → **가성비-고성과형**
         - 매출 0.3~0.7 & 가격 ≥ 0.7 → **중가-고성과형**
         - 매출 < 0.3 & 가격 0.3~0.7 → **저가-저성과형**
         - 매출 0.3~0.7 & 가격 < 0.3 → **박리다매형**
         - 그 외 (모두 중위) → **평균 경쟁형**
      3. **전략 방향 제안**: 진단된 포지셔닝 유형의 일반적인 강점/약점을 고려하여, 이를 강화하거나 개선하기 위한 **핵심 전략 방향 1가지**를 제안합니다. (예: '가성비형' -> "객단가 상승 유도" 또는 "운영 효율화 통한 원가 절감" 제안)

### 절차 3: 보고서 생성
- 모든 분석 결과는 아래 보고서 구조를 반드시 따르는 Markdown 형식으로 제공합니다.
- `# 요약 → ## 핵심 인사이트(불릿) → ## 추천 전략 및 채널(표) → ## 실행 가이드(불릿) → ## 데이터 근거(표)는 자세하게`
"""


# ==============================
# 전역 변수: MCP 세션, Agent, 히스토리
# ==============================
_MCP_READ = None
_MCP_WRITE = None
_MCP_SESSION: Optional[ClientSession] = None
_AGENT = None
_AGENT_WITH_HISTORY = None

# 세션별 히스토리 저장소
_SESSION_STORES: Dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """세션별 히스토리 반환 (없으면 자동 생성)"""
    if session_id not in _SESSION_STORES:
        _SESSION_STORES[session_id] = ChatMessageHistory()
    return _SESSION_STORES[session_id]

def trim_session_history(session_id: str, max_pairs: int = 4):
    """최근 N개 대화쌍만 유지"""
    if session_id in _SESSION_STORES:
        history = _SESSION_STORES[session_id]
        messages = history.messages
        if len(messages) > max_pairs * 2:
            # 최근 N쌍만 유지
            history.clear()
            for msg in messages[-max_pairs * 2:]:
                history.add_message(msg)


# ==============================
# FastAPI 라이프사이클: MCP 세션과 Agent 초기화
# ==============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 MCP 세션과 Agent를 한 번만 초기화"""
    global _MCP_READ, _MCP_WRITE, _MCP_SESSION, _AGENT, _AGENT_WITH_HISTORY
    
    if "GOOGLE_API_KEY" not in os.environ:
        raise RuntimeError("GOOGLE_API_KEY is not set in environment variables.")
    
    # LLM 초기화
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
    
    # MCP 세션 시작
    async with stdio_client(server_params) as (read, write):
        _MCP_READ, _MCP_WRITE = read, write
        async with ClientSession(read, write) as mcp_session:
            _MCP_SESSION = mcp_session
            await mcp_session.initialize()
            tools = await load_mcp_tools(mcp_session)
            
            # Agent 생성 (SystemMessage 포함)
            _AGENT = create_react_agent(llm, tools, state_modifier=SYSTEM_PROMPT)
            
            # 히스토리 래핑
            _AGENT_WITH_HISTORY = RunnableWithMessageHistory(
                _AGENT,
                get_session_history,
                input_messages_key="messages",
            )
            
            print("MCP 세션과 Agent 초기화 완료")
            yield  # 앱 실행
            print("MCP 세션 종료")


# ==============================
# FastAPI 앱
# ==============================
app = FastAPI(title="Merchant Marketing Agent API", lifespan=lifespan)

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
    """세션 히스토리 삭제"""
    _SESSION_STORES.pop(req.session_id, None)
    return {"ok": True}


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    RunnableWithMessageHistory를 사용한 효율적인 히스토리 관리
    """
    if _AGENT_WITH_HISTORY is None:
        raise HTTPException(status_code=500, detail="Agent가 초기화되지 않았습니다. 서버를 재시작하세요.")

    try:
        # Agent 실행 (히스토리 자동 관리)
        result = await _AGENT_WITH_HISTORY.ainvoke(
            {"messages": [HumanMessage(content=req.user_message)]},
            config={"configurable": {"session_id": req.session_id}}
        )

        # AI 응답 추출
        ai_message = result["messages"][-1]
        reply = ai_message.content

        # 토큰 절약을 위한 히스토리 trim
        trim_session_history(req.session_id, max_pairs=4)

        return {"reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 실행 오류: {e!r}")
