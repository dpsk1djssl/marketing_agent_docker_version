# -*- coding: utf-8 -*-
"""
mcp.server.py (추천 강화 버전)

기능
- 전처리/매핑까지 반영된 CSV(mct_sample_with_persona_3_mapped_final.csv)를 로드
- 사용자 질의에서 브랜드명을 부분일치로 인식
- 해당 브랜드의 '슈머유형'과 'A_STAGE'에 따라 플랫폼 추천을 반환
- 배달매출비율이 높으면(>=50%) 배달 채널 추가

엔드포인트(툴)
- find_brands(query: str) -> 부분일치 후보 리스트
- recommend_channels(brand_query: str, prefer_stage: Optional[str]) -> 추천안(JSON)

사용 예
- recommend_channels("성우**")
- recommend_channels("스타벅", prefer_stage="A4")  # A_STAGE 우선 적용(없으면 기존 A_STAGE 사용)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from fastmcp.server import FastMCP, Context

# =========================
# 설정
# =========================
# 데이터 파일 경로를 스크립트 위치 기준으로 안전하게 계산
ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "final_data_with_q2_kpi.csv"  

# 브랜드 후보 컬럼 우선순위(존재하는 첫 컬럼 사용)
BRAND_COL_CANDIDATES = [
    "브랜드명",
    "가맹점명",
    "상호명",
    "ENCODED_MCT",
    "BRAND",
    "brand",
    "상호",
]

# 배달 비율 컬럼 후보
DELIVERY_RATIO_COLS = [
    "배달매출금액 비율",
    "배달매출 비율",
    "배달 비율",
]

# 슈머유형 & A-Stage 기반 추천 테이블
RECO_TABLE: Dict[str, Dict[str, str]] = {
    "모스트슈머": {
        "A3": "인스타그램 릴스, 틱톡",
        "A4": "유튜브 쇼츠, 인스타그램 게시물",
        "A5": "인스타그램 게시물, 지역 카페",
    },
    "유틸슈머": {
        "A3": "네이버 블로그, 유튜브 영상",
        "A4": "네이버 블로그, 지역 카페",
        "A5": "지역 카페, 당근마켓",
    },
    "비지슈머": {
        "A3": "유튜브 쇼츠, 네이버 블로그",
        "A4": "유튜브 영상, 지역 카페",
        "A5": "네이버 블로그, 지역 카페",
    },
    "무소슈머": {
        "A3": "네이버 블로그, 지역 카페",
        "A4": "당근마켓, 네이버 블로그",
        "A5": "지역 카페, 당근마켓",
    },
}
BASE_CHANNEL = "네이버/카카오/구글맵, 리뷰노트"
DELIVERY_EXTRA = "배달의민족/쿠팡이츠"

# =========================
# 유틸
# =========================
def _to_percent100(x: Any) -> Optional[float]:
    """
    '57%', '57', '0.57' 등 -> 57.0
    변환 불가/결측 -> None
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().replace(",", "")
    if s == "" or s == "-999999.9":
        return None
    if s.endswith("%"):
        s = s[:-1].strip()
    try:
        v = float(s)
    except Exception:
        return None
    if 0.0 <= v <= 1.0:
        v *= 100.0
    return v


def _normalize(text: Any) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return re.sub(r"\s+", "", str(text)).lower()


def _choose_brand_column(df: pd.DataFrame) -> Optional[str]:
    for c in BRAND_COL_CANDIDATES:
        if c in df.columns:
            return c
    # 브랜딩 관련 컬럼 추정(한글/영문 '명' 포함)
    for c in df.columns:
        if any(key in c for key in ["브랜드", "가맹점", "상호", "brand", "name"]):
            return c
    return None


def _choose_delivery_col(df: pd.DataFrame) -> Optional[str]:
    for c in DELIVERY_RATIO_COLS:
        if c in df.columns:
            return c
    # 추정
    for c in df.columns:
        if "배달" in c and "비율" in c:
            return c
    return None


def _stage_key(stage: str) -> str:
    """A_STAGE 값을 A3/A4/A5 키로 정규화"""
    s = (stage or "").upper()
    if s.startswith("A3"):
        return "A3"
    if s.startswith("A4"):
        return "A4"
    if s.startswith("A5"):
        return "A5"
    return ""


def _split_channels(s: str) -> List[str]:
    # "네이버 블로그, 유튜브 영상" -> ["네이버 블로그","유튜브 영상"]
    return [t.strip() for t in re.split(r"[,/]", s) if t.strip()]


# =========================
# 데이터 로드
# =========================
DF: Optional[pd.DataFrame] = None
BRAND_COL: Optional[str] = None
DELIV_COL: Optional[str] = None

def _load_df() -> pd.DataFrame:
    global DF, BRAND_COL, DELIV_COL
    DF = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    BRAND_COL = _choose_brand_column(DF)
    DELIV_COL = _choose_delivery_col(DF)
    return DF

# 초기 로드
_load_df()

# =========================
# MCP 서버
# =========================
mcp = FastMCP(
    "MerchantSearchServer",
    instructions="""
    브랜드(가맹점)를 찾아 '슈머유형'과 'A_STAGE'에 따라 마케팅 채널을 추천합니다.
    - find_brands: 부분일치 검색으로 브랜드 후보를 보여줍니다.
    - get_merchant: 가맹점(브랜드)명을 정확히 일치시켜 상세 레코드를 반환합니다.
    - recommend_channels: 브랜드명을 입력하면 추천 채널을 JSON으로 돌려줍니다.
    """
)

# -------------------------
# 통합검색 도구 
# -------------------------
@mcp.tool
def search_merchants(query: str) -> Dict[str, Any]:
    """
    가맹점명(부분/마스킹 일치) 또는 가맹점구분번호(완전 일치)로 가맹점을 검색합니다.
    검색 결과가 여러 개일 경우, 선택 가능한 목록을 반환합니다.
    
    매개변수:
      - query: 검색어 (예: "성우**", "16184E93D9")
    """
    if DF is None: _load_df()

    normalized_query = _normalize(query).replace("*", "")
    if not normalized_query:
        return {"count": 0, "merchants": [], "reason": "빈 검색어"}

    # 가맹점구분번호로 검색 시도 (숫자와 알파벳으로만 구성된 긴 문자열)
    if re.fullmatch(r'[a-z0-9]{10,}', normalized_query):
        mask = DF['가맹점구분번호'].astype(str).str.lower() == normalized_query
        matched_df = DF[mask]
    # 가맹점명으로 검색
    else:
        ser = DF[BRAND_COL].astype(str)
        mask = ser.map(lambda x: normalized_query in _normalize(x))
        matched_df = DF[mask]

    count = len(matched_df)
    if count == 0:
        return {"count": 0, "merchants": [], "reason": "검색 결과 없음"}

    # 사용자 선택에 필요한 최소한의 정보만 추출하여 반환
    merchants = matched_df[[
        '가맹점구분번호', BRAND_COL, '가맹점주소'
    ]].rename(columns={BRAND_COL: '가맹점명'}).to_dict(orient='records')[:10] # 최대 10개만

    return {
        "count": count,
        "merchants": merchants
    }

# -------------------------
# 검색: 부분일치 후보
# -------------------------
@mcp.tool
def find_brands(query: str) -> Dict[str, Any]:
    """
    부분일치로 브랜드 후보를 찾아 상위 20개 반환
    """
    if DF is None or BRAND_COL is None:
        _load_df()

    q = _normalize(query).replace("*", "")  # '성우**' 같은 마스킹 허용
    if not q:
        return {"ok": False, "reason": "빈 검색어"}

    ser = DF[BRAND_COL].astype(str)
    mask = ser.map(lambda x: _normalize(x).__contains__(q))
    hits = ser[mask].dropna().unique().tolist()[:20]
    return {
        "ok": True,
        "brand_column": BRAND_COL,
        "count": len(hits),
        "candidates": hits,
    }

# -------------------------
# 조회: 정확 일치로 레코드 가져오기
# -------------------------
@mcp.tool
def get_merchant(merchant_name: str) -> Dict[str, Any]:
    """
    브랜드/가맹점명을 정확히 일치시켜 해당 레코드들을 반환합니다.
    - 공백 제거/소문자화/마스킹(*) 제거 후 비교
    - 결과는 최대 50건 제한
    """
    if DF is None or BRAND_COL is None:
        _load_df()

    key = _normalize(merchant_name).replace("*", "")
    if not key:
        return {"ok": False, "reason": "빈 입력"}

    ser = DF[BRAND_COL].astype(str)
    norm = ser.map(lambda x: _normalize(x).replace("*", ""))
    mask = norm.eq(key)
    matched = DF[mask]
    records = matched.to_dict(orient="records")[:50]
    return {
        "ok": bool(records),
        "brand_column": BRAND_COL,
        "count": len(records),
        "records": records,
    }

# -------------------------
# 추천: 채널
# -------------------------
@mcp.tool
def recommend_channels(brand_query: str, prefer_stage: Optional[str] = None) -> Dict[str, Any]:
    """
    brand_query: 부분일치 문자열(예: '성우', '할매순대', '스타벅')
    prefer_stage: 'A3'|'A4'|'A5' 또는 'A3_Acquisition' 등(선택)
    """
    if DF is None or BRAND_COL is None:
        _load_df()

    q = _normalize(brand_query).replace("*", "")
    if not q:
        return {"ok": False, "reason": "빈 브랜드 질의"}

    # 매칭 행 추출: 정확 일치(정규화 & '*' 제거) 우선 -> 없으면 부분 일치
    ser = DF[BRAND_COL].astype(str)
    norm_ser = ser.map(lambda x: _normalize(x).replace("*", ""))
    exact_idx = norm_ser[norm_ser.eq(q)].index.tolist()
    match_mode = "exact"
    if exact_idx:
        i = exact_idx[0]
    else:
        part_idx = [ix for ix, name in ser.items() if q in _normalize(name)]
        if not part_idx:
            return {"ok": False, "reason": f"해당 브랜드를 찾을 수 없음: {brand_query}"}
        i = part_idx[0]
        match_mode = "partial"
    row = DF.loc[i]

    # 필수 값
    cluster = str(row.get("슈머유형", "") or "")
    a_stage_raw = str(row.get("A_STAGE", "") or "")
    stage_key = _stage_key(prefer_stage or a_stage_raw)  # 우선 prefer_stage

    # 추천 표 조회
    reco_by_stage = RECO_TABLE.get(cluster, {})
    stage_channels = reco_by_stage.get(stage_key, "")
    base_channels = BASE_CHANNEL
    delivery_channels = DELIVERY_EXTRA

    # 배달 비율 판단
    delivery_ratio = None
    if DELIV_COL and DELIV_COL in DF.columns:
        delivery_ratio = _to_percent100(row.get(DELIV_COL))
    include_delivery = delivery_ratio is not None and delivery_ratio >= 50.0

    # 응답 구성
    primary = _split_channels(stage_channels) if stage_channels else []
    base = _split_channels(base_channels)
    extra = _split_channels(delivery_channels) if include_delivery else []

    return {
        "ok": True,
        "brand_column": BRAND_COL,
        "brand": row.get(BRAND_COL),
        "match_mode": match_mode,
        "cluster_type": cluster,           # 슈머유형
        "a_stage": a_stage_raw,            # 원본 A_STAGE
        "stage_used": stage_key or None,   # 실제 추천에 사용한 스테이지 키(A3/A4/A5)
        "delivery_ratio_col": DELIV_COL,
        "delivery_ratio": delivery_ratio,
        "include_delivery_channels": include_delivery,
        "recommendations": {
            "primary_by_stage": primary,   # A3/A4/A5별 주요 채널(2개)
            "base_channels": base,         # 기본 채널
            "delivery_additional": extra,  # 배달 높은 경우 추가
        },
    }

# -------------------------
# Q2
# -------------------------
@mcp.tool
def analyze_low_revisit_store(merchant_id: str) -> Dict[str, Any]:
    """
    재방문율이 낮은 특정 가맹점(merchant_id)의 7P 마케팅 믹스 지표를 종합적으로 분석하여 반환합니다. 
    
    매개변수:
      - merchant_id: 분석할 가맹점의 ID (가맹점구분번호)
    
    반환값:
      - 7P 분석 지표가 담긴 딕셔너리
    """
    if DF is None: _load_df()
    
    # 가맹점 구분번호는 문자열 타입으로 비교해야 정확합니다.
    store_data = DF[DF['가맹점구분번호'].astype(str) == str(merchant_id)]
    
    if len(store_data) == 0:
        return {"found": False, "message": f"'{merchant_id}' 가맹점을 찾을 수 없습니다."}
    
    # 동일 가맹점의 여러 월 데이터가 있을 경우, 최신 월을 기준으로 분석합니다.
    result = store_data.sort_values(by='기준년월', ascending=False).iloc[0].to_dict()
    
    # 에이전트가 분석하기 쉽도록 7P 기준으로 데이터를 구조화하여 반환합니다.
    report = {
        "found": True,
        "merchant_name": result.get("가맹점명"),
        "product": {
            "revisit_rank": result.get("PCT_REVISIT"),
            "rtf_rank": result.get("PCT_RTF"),
            "sales_rank": result.get("PCT_SALES"),
            "customer_type": result.get("CUSTOMER_TYPE")
        },
        "price": {
            "price_rank": result.get("PCT_PRICE")
        },
        "place": {
            "tenure_rank": result.get("PCT_TENURE")
        },
        "process": {
            "process_score_rank": result.get("PCT_PROCESS")
        }
    }
    return report



# -------------------------
# 헬스체크 / 리로드
# -------------------------
@mcp.tool
def ping() -> str:
    return "pong"


@mcp.tool
def reload_data() -> Dict[str, Any]:
    df = _load_df()
    return {
        "ok": True,
        "rows": int(len(df)),
        "brand_col": BRAND_COL,
        "delivery_col": DELIV_COL,
        "columns": df.columns.tolist(),
    }


if __name__ == "__main__":
    # 개발 로컬 테스트용
    # uv run python mcp.server.py
    mcp.run()
