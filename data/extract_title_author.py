import json

# 입력 파일과 출력 파일 경로
input_file = './library_books.json'
output_file = './library_books_title_author.json'

# 데이터 읽기
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# title과 author만 추출
extracted_data = []
for book in data:
    extracted_data.append({
        'title': book['title'],
        'author': book['author']
    })

# 결과를 새 JSON 파일로 저장
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(extracted_data, f, ensure_ascii=False, indent=2)

print(f'추출 완료! {len(extracted_data)}개의 책 정보가 {output_file}에 저장되었습니다.')