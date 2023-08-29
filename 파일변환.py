import pandas as pd

# 엑셀 파일 경로
excel_file_path = 'C:/Training.xlsx'

# 엑셀 파일을 읽어옵니다.
df = pd.read_excel(excel_file_path)

# 필요한 열 추출 (감정_대분류, 사람문장1)
df_filtered = df[['감정_대분류', '사람문장1']]

# CSV 파일로 저장
csv_file_path = 'C:/Training.csv'
df_filtered.to_csv(csv_file_path, index=False, encoding='utf-8')

print('CSV 파일 생성 완료:', csv_file_path)