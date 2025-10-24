# 파일명: q2_preprocess.py (최종 수정본)
import pandas as pd
import numpy as np

print("Q2 데이터 전처리 스크립트를 시작합니다.")

# 1. 데이터 불러오기
df = pd.read_csv("data/final_2_q1.csv", encoding="utf-8-sig")
print(f"데이터 로딩 완료. Shape: {df.shape}")

# ------------------------------------------------------------------- #
# ✅ [수정된 부분] 문자열에서 숫자만 추출한 후, 숫자형으로 변환합니다.
# ------------------------------------------------------------------- #
cols_to_convert = [
    '매출건수 구간', '유니크 고객 수 구간', '매출금액 구간',
    '객단가 구간', '가맹점 운영개월수 구간', '취소율 구간'
]
print(f"\n데이터 타입 변환을 시작합니다. 대상 컬럼: {cols_to_convert}")
for col in cols_to_convert:
    # 정규표현식을 사용하여 컬럼 값에서 숫자 부분(\d+)만 추출합니다.
    df[col] = df[col].str.extract(r'(\d+)').astype(float)

print("숫자 변환 후 각 컬럼의 결측치(NaN) 개수:")
print(df[cols_to_convert].isnull().sum())
# ------------------------------------------------------------------- #

# (이하 코드는 동일)
GROUP_KEYS = ["상권", "업종"]
print(f"\n상대위치 계산을 위한 그룹화 기준: {GROUP_KEYS}")

# 3. Product (제품) 관련 KPI 계산
print("\n[STEP 1/5] Product 관련 KPI를 계산합니다...")
df['PCT_REVISIT'] = df.groupby(GROUP_KEYS)['재방문 고객 비중'].rank(pct=True, ascending=True)
df['RTF'] = (7 - df['매출건수 구간']) / (7 - df['유니크 고객 수 구간'])

# ✅ [수정된 부분] FutureWarning 해결을 위해 inplace=True 대신 재할당 방식을 사용합니다.
df['RTF'] = df['RTF'].replace([np.inf, -np.inf], np.nan)

df['PCT_RTF'] = df.groupby(GROUP_KEYS)['RTF'].rank(pct=True, ascending=True)
df['CRI'] = df['재방문 고객 비중'] * df['RTF']
revisit_median = df.groupby(GROUP_KEYS)['PCT_REVISIT'].transform('median')
rtf_median = df.groupby(GROUP_KEYS)['PCT_RTF'].transform('median')
conditions = [
    (df['PCT_REVISIT'] >= revisit_median) & (df['PCT_RTF'] >= rtf_median),
    (df['PCT_REVISIT'] >= revisit_median) & (df['PCT_RTF'] < rtf_median),
    (df['PCT_REVISIT'] < revisit_median) & (df['PCT_RTF'] >= rtf_median),
    (df['PCT_REVISIT'] < revisit_median) & (df['PCT_RTF'] < rtf_median)
]
choices = ['A형(충성고객형)', 'B형(유입형)', 'C형(집중형)', 'D형(단발형)']
df['CUSTOMER_TYPE'] = np.select(conditions, choices, default='분류불가')
df['PCT_SALES'] = df.groupby(GROUP_KEYS)['매출금액 구간'].rank(pct=True, ascending=False)

# 4. Price (가격) 관련 KPI 계산
print("[STEP 2/5] Price 관련 KPI를 계산합니다...")
df['PCT_PRICE'] = df.groupby(GROUP_KEYS)['객단가 구간'].rank(pct=True, ascending=False)

print("유사 가격대 점포 비중 계산 시작...")

# 각 그룹 내에서 객단가 구간별 점포 수를 미리 계산 (효율성 위해)
price_counts = df.groupby(GROUP_KEYS + ['객단가 구간']).size().unstack(fill_value=0)

def calculate_similar_price_ratio(row):
    group_key = tuple(row[k] for k in GROUP_KEYS)
    my_price_class = row['객단가 구간']

    if pd.isna(my_price_class) or group_key not in price_counts.index:
        return np.nan

    group_prices = price_counts.loc[group_key]
    total_stores_in_group = group_prices.sum()

    if total_stores_in_group <= 1:
        return np.nan # 비교 대상 없음

    similar_count = 0
    for price_class in [my_price_class - 1, my_price_class, my_price_class + 1]:
        if price_class in group_prices:
            similar_count += group_prices[price_class]

    # 자기 자신 제외
    if my_price_class in group_prices and group_prices[my_price_class] > 0:
         similar_count -= 1 

    return similar_count / (total_stores_in_group - 1)

df['SIMILAR_PRICE_RATIO'] = df.apply(calculate_similar_price_ratio, axis=1)
df['PCT_SIMILAR_PRICE'] = df.groupby(GROUP_KEYS)['SIMILAR_PRICE_RATIO'].rank(pct=True, ascending=True) # 비중 높을수록 경쟁 밀집 (상위)
print("유사 가격대 점포 비중 및 백분위 계산 완료.")

# 5. Place (입지) 관련 KPI 계산
print("[STEP 3/5] Place 관련 KPI를 계산합니다...")
df['PCT_TENURE'] = df.groupby(GROUP_KEYS)['가맹점 운영개월수 구간'].rank(pct=True, ascending=False)

# 6. Process (운영절차) 관련 KPI 계산
print("[STEP 4/5] Process 관련 KPI를 계산합니다...")
df['PROCESS_SCORE'] = 7 - df['취소율 구간']
df['PCT_PROCESS'] = df.groupby(GROUP_KEYS)['PROCESS_SCORE'].rank(pct=True, ascending=True)

print("\n[STEP 5/5] 백분위 순위 3단계 구간화(_CAT 컬럼 추가)...")

pct_columns = [
    'PCT_REVISIT', 'PCT_RTF', 'PCT_SALES', 'PCT_PRICE', 
    'PCT_TENURE', 'PCT_PROCESS', 'PCT_SIMILAR_PRICE' 
]

def categorize_pct(pct_value):
    if pd.isna(pct_value):
        return '정보없음'
    elif pct_value >= 0.7:
        return '상위'
    elif pct_value >= 0.3:
        return '중위'
    else:
        return '하위'

for col in pct_columns:
    cat_col_name = col.replace('PCT_', '') + '_CAT' # 예: PCT_REVISIT -> REVISIT_CAT
    df[cat_col_name] = df[col].apply(categorize_pct)

print("모든 백분위 순위에 대한 구간화 완료.")

# 추가된 카테고리 컬럼명 확인 (선택 사항)
new_cat_cols = [col.replace('PCT_', '') + '_CAT' for col in pct_columns]
print(f"추가된 카테고리 컬럼: {new_cat_cols}")


# 7. 최종 결과 저장
output_filename = "final_data_with_cats.csv"
df.to_csv(output_filename, index=False, encoding="utf-8-sig")
print(f"\n모든 전처리 완료! 최종 파일 저장: {output_filename}")