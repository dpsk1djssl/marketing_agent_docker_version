
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "data" / "final_2_q1.csv"
OUTPUT_PATH = ROOT / "final_data_.csv"

df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

# convert interval-like string columns to numeric if present
cols_to_convert = [
    '매출건수 구간', '유니크 고객 수 구간', '매출금액 구간',
    '객단가 구간', '가맹점 운영개월수 구간', '취소율 구간'
]
for col in cols_to_convert:
    if col in df.columns:
        df[col] = df[col].astype(str).str.extract(r'(\d+)')
        df[col] = pd.to_numeric(df[col], errors='coerce')

# parse month
def parse_yyyymm(x):
    try:
        s = str(int(float(x)))
    except Exception:
        s = str(x).strip()
    if len(s) == 6 and s.isdigit():
        return pd.to_datetime(s, format='%Y%m', errors='coerce')
    return pd.to_datetime(s, errors='coerce')

if '기준년월' in df.columns:
    df['기준년월'] = df['기준년월'].apply(parse_yyyymm)

GROUP_KEYS = ["상권", "업종"]
for k in GROUP_KEYS:
    if k not in df.columns:
        df[k] = 'ALL'

# Product
df['PCT_REVISIT'] = df.groupby(GROUP_KEYS)['재방문 고객 비중'].transform(lambda s: s.rank(pct=True, ascending=True) if '재방문 고객 비중' in df.columns else np.nan)

if {'매출건수 구간', '유니크 고객 수 구간'}.issubset(df.columns):
    denom = (7 - df['유니크 고객 수 구간'])
    df['RTF'] = np.where(denom == 0, np.nan, (7 - df['매출건수 구간']) / denom)
    df['RTF'] = df['RTF'].replace([np.inf, -np.inf], np.nan)
    df['PCT_RTF'] = df.groupby(GROUP_KEYS)['RTF'].transform(lambda s: s.rank(pct=True, ascending=True))
else:
    df['RTF'] = np.nan
    df['PCT_RTF'] = np.nan

df['CRI'] = np.where(df[['재방문 고객 비중','RTF']].notna().all(axis=1), df['재방문 고객 비중'] * df['RTF'], np.nan)
df['PCT_CRI'] = df.groupby(GROUP_KEYS)['CRI'].transform(lambda s: s.rank(pct=True, ascending=True))

revisit_med = df.groupby(GROUP_KEYS)['PCT_REVISIT'].transform('median')
rtf_med = df.groupby(GROUP_KEYS)['PCT_RTF'].transform('median')
conds = [
    (df['PCT_REVISIT'] >= revisit_med) & (df['PCT_RTF'] >= rtf_med),
    (df['PCT_REVISIT'] >= revisit_med) & (df['PCT_RTF'] < rtf_med),
    (df['PCT_REVISIT'] < revisit_med) & (df['PCT_RTF'] >= rtf_med),
    (df['PCT_REVISIT'] < revisit_med) & (df['PCT_RTF'] < rtf_med)
]
choices = ['A형(충성고객형)','B형(유입형)','C형(집중형)','D형(단발형)']
df['CUSTOMER_TYPE'] = np.select(conds, choices, default='분류불가')

df['PCT_SALES'] = df.groupby(GROUP_KEYS)['매출금액 구간'].transform(lambda s: s.rank(pct=True, ascending=False) if '매출금액 구간' in df.columns else np.nan)

# Price
df['PCT_PRICE'] = df.groupby(GROUP_KEYS)['객단가 구간'].transform(lambda s: s.rank(pct=True, ascending=False) if '객단가 구간' in df.columns else np.nan)

if '객단가 구간' in df.columns:
    price_counts = df.groupby(GROUP_KEYS + ['객단가 구간']).size().unstack(fill_value=0)
    def calc_similar(row):
        key = tuple(row[k] for k in GROUP_KEYS)
        my = row['객단가 구간']
        if pd.isna(my) or key not in price_counts.index:
            return np.nan
        gs = price_counts.loc[key]
        total = gs.sum()
        if total <= 1:
            return np.nan
        similar = 0
        for p in [my-1, my, my+1]:
            if p in gs.index:
                similar += gs.loc[p]
        if my in gs.index and gs.loc[my] > 0:
            similar -= 1
        denom = total - 1
        if denom <= 0:
            return np.nan
        return similar / denom
    df['SIMILAR_PRICE_RATIO'] = df.apply(calc_similar, axis=1)
    df['PCT_SIMILAR_PRICE'] = df.groupby(GROUP_KEYS)['SIMILAR_PRICE_RATIO'].transform(lambda s: s.rank(pct=True, ascending=True))
else:
    df['SIMILAR_PRICE_RATIO'] = np.nan
    df['PCT_SIMILAR_PRICE'] = np.nan

# Place
df['PCT_TENURE'] = df.groupby(GROUP_KEYS)['가맹점 운영개월수 구간'].transform(lambda s: s.rank(pct=True, ascending=False) if '가맹점 운영개월수 구간' in df.columns else np.nan)

# Process
if '취소율 구간' in df.columns:
    df['PROCESS_SCORE'] = 7 - df['취소율 구간']
    df['PCT_PROCESS'] = df.groupby(GROUP_KEYS)['PROCESS_SCORE'].transform(lambda s: s.rank(pct=True, ascending=True))
else:
    df['PROCESS_SCORE'] = np.nan
    df['PCT_PROCESS'] = np.nan

# Promotion (last 12 months)
stage_map = {'A3':3,'A4':4,'A5':5}
df['A_STAGE_STR'] = df['A_STAGE'].astype(str) if 'A_STAGE' in df.columns else np.nan
df['A_STAGE_NUM'] = df['A_STAGE_STR'].map(stage_map)
df = df.sort_values(by=['가맹점구분번호','기준년월'] if '기준년월' in df.columns else ['가맹점구분번호'])
df['PREV_A_STAGE_NUM'] = df.groupby('가맹점구분번호')['A_STAGE_NUM'].shift(1)
df['A3_to_A4_flag'] = np.where((df['PREV_A_STAGE_NUM']==3)&(df['A_STAGE_NUM']==4),1,0)
df['A4_to_A5_flag'] = np.where((df['PREV_A_STAGE_NUM']==4)&(df['A_STAGE_NUM']==5),1,0)
df['PREV_A3_flag'] = np.where(df['PREV_A_STAGE_NUM']==3,1,0)
df['PREV_A4_flag'] = np.where(df['PREV_A_STAGE_NUM']==4,1,0)

promo_list = []
if '기준년월' in df.columns:
    for mct, g in df.groupby('가맹점구분번호'):
        g = g.sort_values('기준년월')
        recent = g['기준년월'].max()
        cutoff = recent - pd.DateOffset(months=12)
        last12 = g[g['기준년월'] > cutoff]
        prev_a3 = int(last12['PREV_A3_flag'].sum()) if not last12.empty else 0
        prev_a4 = int(last12['PREV_A4_flag'].sum()) if not last12.empty else 0
        a3a4 = (int(last12['A3_to_A4_flag'].sum())/prev_a3*100) if prev_a3>0 else np.nan
        a4a5 = (int(last12['A4_to_A5_flag'].sum())/prev_a4*100) if prev_a4>0 else np.nan
        last_stage = g['A_STAGE_STR'].iloc[-1] if 'A_STAGE_STR' in g.columns else np.nan
        promo_list.append({'가맹점구분번호':mct,'A3A4_12':a3a4,'A4A5_12':a4a5,'PROMO_LAST_STAGE':last_stage})
    promo_df = pd.DataFrame(promo_list)
else:
    promo_df = df.groupby('가맹점구분번호').apply(lambda g: pd.Series({'A3A4_12':np.nan,'A4A5_12':np.nan,'PROMO_LAST_STAGE':g['A_STAGE_STR'].iloc[-1] if 'A_STAGE_STR' in g.columns else np.nan})).reset_index()

promo_df['PROMOTION_STATUS'] = np.where(promo_df[['A3A4_12','A4A5_12']].isna().all(axis=1), promo_df['PROMO_LAST_STAGE'].fillna('미확인') + ' 유지단계', '전환발생')
df = df.merge(promo_df[['가맹점구분번호','A3A4_12','A4A5_12','PROMOTION_STATUS']], on='가맹점구분번호', how='left')
df['PCT_A3A4'] = df.groupby(GROUP_KEYS)['A3A4_12'].transform(lambda s: s.rank(pct=True, ascending=True))
df['PCT_A4A5'] = df.groupby(GROUP_KEYS)['A4A5_12'].transform(lambda s: s.rank(pct=True, ascending=True))

# Categorize percentiles
pct_cols = ['PCT_REVISIT','PCT_RTF','PCT_CRI','PCT_SALES','PCT_PRICE','PCT_SIMILAR_PRICE','PCT_TENURE','PCT_PROCESS','PCT_A3A4','PCT_A4A5']
def categorize(v):
    if pd.isna(v):
        return '정보없음'
    if v >= 0.7:
        return '상위'
    if v >= 0.3:
        return '중위'
    return '하위'
for c in pct_cols:
    cat = c.replace('PCT_','') + '_CAT'
    df[cat] = df[c].apply(categorize) if c in df.columns else '정보없음'

# placeholders for external-data-required fields
for col in ['PEOPLE_SCORE','PHYSICAL_EVIDENCE_SCORE']:
    if col not in df.columns:
        df[col] = np.nan

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
