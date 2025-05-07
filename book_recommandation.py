import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import json
import torch

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 디바이스 설정
if torch.cuda.is_available():
    device = 'cuda'
    device_map = {'': 0}
elif torch.backends.mps.is_available():
    device = 'mps'
    device_map = {'': device}
else:
    device = 'cpu'
    device_map = {'': device}

# 임베딩 모델
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': device}
)

# 벡터 DB 저장 경로
BOOKS_DB_DIR = './chroma_books'
HISTORY_DB_DIR = './chroma_history'
PREFS_DB_DIR = './chroma_prefs'

# 도서 정보 벡터 DB
if not os.path.exists(BOOKS_DB_DIR):
    with open('./data/library_books.json', 'r', encoding='utf-8') as f:
        books = json.load(f)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    book_documents = []
    for book in books:
        text = f"카테고리: {book['category_name']}\n제목: {book['title']}\n내용: {book['contents']}\n목차: {book['table_of_contents']}"
        metadata = {
            'title': book['title'],
            'category': book['category_name'],
            'author': book['author'],
            'isbn': book['isbn'],
            'publish_year': book['publish_year'],
        }
        chunks = text_splitter.create_documents(
            texts=[text],
            metadatas=[metadata]
        )
        book_documents.extend(chunks)
    books_vectorstore = Chroma.from_documents(
        documents=book_documents,
        embedding=embeddings,
        collection_name="books_collection",
        persist_directory=BOOKS_DB_DIR
    )
    books_vectorstore.persist()
else:
    books_vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name="books_collection",
        persist_directory=BOOKS_DB_DIR
    )

# 독서 이력 벡터 DB
if not os.path.exists(HISTORY_DB_DIR):
    with open('./data/test-case/user_reading_history.json', 'r', encoding='utf-8') as f:
        reading_history = json.load(f)
    history_documents = []
    for book in reading_history['read_books']:
        text = f"제목: {book['title']}\n저자: {book['author']}\n평점: {book['rating']}\n리뷰: {book['review']}\n장르: {', '.join(book['genre'])}\n읽은 날짜: {book['read_date']}"
        metadata = {
            'title': book['title'],
            'author': book['author'],
            'rating': book['rating'],
            'genre_str': ', '.join(book['genre']),
            'read_date': book['read_date']
        }
        doc = Document(page_content=text, metadata=metadata)
        history_documents.append(doc)
    history_vectorstore = Chroma.from_documents(
        documents=history_documents,
        embedding=embeddings,
        collection_name="reading_history_collection",
        persist_directory=HISTORY_DB_DIR
    )
    history_vectorstore.persist()
else:
    history_vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name="reading_history_collection",
        persist_directory=HISTORY_DB_DIR
    )

# 독서 취향 벡터 DB
if not os.path.exists(PREFS_DB_DIR):
    with open('./data/test-case/user_reading_preferences.json', 'r', encoding='utf-8') as f:
        reading_preferences = json.load(f)
        preferences_text = f"선호 장르: {', '.join([f'{g['name']}({g['weight']})' for g in reading_preferences['preferences']['genres']])}\n"
        preferences_text += f"독서 스타일: 길이({reading_preferences['preferences']['reading_style']['preferred_length']}), "
        preferences_text += f"복잡도({reading_preferences['preferences']['reading_style']['complexity_level']}), "
        preferences_text += f"톤({reading_preferences['preferences']['reading_style']['tone']})\n"
        preferences_text += f"관심 키워드: {', '.join(reading_preferences['preferences']['keywords'])}\n"
        preferences_text += f"기피 주제: {', '.join(reading_preferences['preferences']['avoid_topics'])}"
        preferences_metadata = {
            'genres_str': ', '.join([g['name'] for g in reading_preferences['preferences']['genres']]),
            'keywords_str': ', '.join(reading_preferences['preferences']['keywords']),
            'avoid_topics_str': ', '.join(reading_preferences['preferences']['avoid_topics'])
        }
        preferences_doc = Document(page_content=preferences_text, metadata=preferences_metadata)
        preferences_vectorstore = Chroma.from_documents(
            documents=[preferences_doc],
            embedding=embeddings,
            collection_name="reading_preferences_collection",
            persist_directory=PREFS_DB_DIR
        )
        preferences_vectorstore.persist()
else:
    preferences_vectorstore = Chroma(
        embedding_function=embeddings,
        collection_name="reading_preferences_collection",
        persist_directory=PREFS_DB_DIR
    )

# 프롬프트 템플릿 및 포맷 함수
BASE_RECOMMENDATION_TEMPLATE = """당신은 사용자의 독서 취향과 감정 상태에 맞춰 도서를 추천하는 전문가입니다.
다음 정보를 분석하여 사용자에게 가장 적합한 {num_recommendations}권의 책을 추천해주세요.

## 사용자 정보
- 현재 감정 상태: {user_emotion}
- 원하는 감정적 효과: {desired_emotional_effect}
- 직업: {occupation}
- 독서 상황: {reading_context}
- 선호하는 집중도: {focus_level}

## 사용자 취향 정보
{preferences}

## 사용자가 읽은 관련 책들
{reading_history}

## 후보 도서 목록
{candidate_books}

다음과 같은 척도로 각 후보 도서를 평가하세요:
1. 사용자의 현재 감정 상태와 원하는 감정적 효과와의 적합성
2. 사용자의 취향과의 일치도 (장르, 스타일, 선호 키워드)
3. 사용자의 기피 주제와 겹치지 않는지 여부
4. 사용자의 독서 이력과의 연관성
5. 독서 상황과 선호하는 집중도에 맞는지 여부
6. 사용자의 직업과 관련된 통찰이나 도움이 될 수 있는지 여부

최종 추천은 다음 형식으로 제시해주세요:
1. [첫 번째 추천 도서 제목] - 저자
   - 추천 이유: (사용자의 현재 감정과 원하는 효과를 고려한 구체적인 이유)
   - 이 책이 도움이 될 수 있는 이유: (감정적 효과나 얻을 수 있는 통찰 등)

2. [두 번째 추천 도서 제목] - 저자
   - 추천 이유: (사용자의 현재 감정과 원하는 효과를 고려한 구체적인 이유)
   - 이 책이 도움이 될 수 있는 이유: (감정적 효과나 얻을 수 있는 통찰 등)

3. [세 번째 추천 도서 제목] - 저자
   - 추천 이유: (사용자의 현재 감정과 원하는 효과를 고려한 구체적인 이유)
   - 이 책이 도움이 될 수 있는 이유: (감정적 효과나 얻을 수 있는 통찰 등)
"""
base_prompt_template = ChatPromptTemplate.from_template(BASE_RECOMMENDATION_TEMPLATE)

def format_reading_history(reading_history_data):
    if not reading_history_data:
        return "관련 독서 이력이 없습니다."
    if isinstance(reading_history_data, list) and all(isinstance(item, dict) and 'metadata' in item for item in reading_history_data):
        return "\n\n".join([
            f"- 도서: {item['metadata'].get('title', '제목 없음')}\n"
            f"  저자: {item['metadata'].get('author', '저자 미상')}\n"
            f"  장르: {item['metadata'].get('genre_str', '장르 없음')}\n"
            f"  평점: {item['metadata'].get('rating', 0)}\n"
            f"  내용: {item['content'][:200]}..."
            for item in reading_history_data
        ])
    elif isinstance(reading_history_data, list) and all(isinstance(item, dict) and 'title' in item for item in reading_history_data):
        return "\n".join([
            f"- '{item['title']}' (저자: {item['author']}, 장르: {item['genre']}, 평점: {item['rating']})\n  리뷰: {item['review']}"
            for item in reading_history_data
        ])
    return str(reading_history_data)

def format_book_candidates(candidates_data):
    if not candidates_data:
        return "추천할 만한 후보 도서가 없습니다."
    if isinstance(candidates_data, list) and all(isinstance(item, dict) and 'metadata' in item for item in candidates_data):
        return "\n\n".join([
            f"{i+1}. 제목: {item['metadata'].get('title', '제목 없음')}\n"
            f"   저자: {item['metadata'].get('author', '저자 미상')}\n"
            f"   카테고리: {item['metadata'].get('category', '카테고리 없음')}\n"
            f"   내용: {item['content'][:300]}..."
            for i, item in enumerate(candidates_data)
        ])
    elif isinstance(candidates_data, list) and all(isinstance(item, dict) and 'index' in item for item in candidates_data):
        return "\n".join([
            f"{book['index']}. '{book['title']}' (저자: {book['author']}, 카테고리: {book['category']})\n   내용 요약: {book['content'][:200]}..."
            for book in candidates_data
        ])
    return str(candidates_data)

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    api_key=api_key
)
chain = base_prompt_template | llm

def multi_stage_rag_search(user_query, num_candidates, num_history_results):
    results = {}
    preference_results = preferences_vectorstore.similarity_search_with_score(user_query, k=1)
    if preference_results:
        results['user_preferences'] = preference_results[0]
    history_results = history_vectorstore.similarity_search_with_score(user_query, k=num_history_results)
    if history_results:
        results['reading_history'] = history_results
    enhanced_query = user_query
    if 'user_preferences' in results:
        pref_doc = results['user_preferences'][0]
        keywords = pref_doc.metadata.get('keywords_str', '')
        if keywords:
            enhanced_query += f" 키워드: {keywords}"
    if 'reading_history' in results:
        liked_genres = []
        for doc, _ in results['reading_history']:
            if doc.metadata.get('rating', 0) >= 4.0:
                genres = doc.metadata.get('genre_str', '').split(', ')
                liked_genres.extend(genres)
        if liked_genres:
            unique_genres = list(set(liked_genres))
            enhanced_query += f" 선호 장르: {', '.join(unique_genres[:3])}"
    book_results = books_vectorstore.similarity_search_with_score(enhanced_query, k=num_candidates)
    results['recommended_books'] = book_results
    formatted_results = []
    if 'user_preferences' in results:
        pref_doc, pref_score = results['user_preferences']
        formatted_results.append({
            'type': 'user_preferences',
            'content': pref_doc.page_content,
            'metadata': pref_doc.metadata,
            'score': pref_score
        })
    if 'reading_history' in results:
        for i, (hist_doc, hist_score) in enumerate(results['reading_history']):
            if i < 2:
                formatted_results.append({
                    'type': 'reading_history',
                    'content': hist_doc.page_content,
                    'metadata': hist_doc.metadata,
                    'score': hist_score
                })
    recommended_count = 0
    for book_doc, book_score in results['recommended_books']:
        formatted_results.append({
            'type': 'recommended_book',
            'content': book_doc.page_content,
            'metadata': book_doc.metadata,
            'score': book_score
        })
        recommended_count += 1
        if recommended_count >= num_candidates - 2:
            break
    return formatted_results

def multi_stage_recommendation(user_emotion, desired_emotional_effect, occupation, reading_context, focus_level, num_candidates=8, num_history_results=3, num_recommendations=3):
    search_query = f"{user_emotion}을 느끼는 사람이 {desired_emotional_effect}할 수 있는 책"
    
    rag_results = multi_stage_rag_search(search_query, num_candidates, num_history_results)
    
    user_preferences = None
    reading_history = []
    book_candidates = []
    
    for item in rag_results:
        if item['type'] == 'user_preferences':
            user_preferences = item
        elif item['type'] == 'reading_history':
            reading_history.append(item)
        elif item['type'] == 'recommended_book':
            book_candidates.append(item)
    
    preferences_formatted = user_preferences['content'] if user_preferences else "취향 정보가 없습니다."
    
    history_formatted = format_reading_history(reading_history)
    candidates_formatted = format_book_candidates(book_candidates)
    
    result = chain.invoke({
        "user_emotion": user_emotion,
        "desired_emotional_effect": desired_emotional_effect,
        "occupation": occupation,
        "reading_context": reading_context,
        "focus_level": focus_level,
        "preferences": preferences_formatted,
        "reading_history": history_formatted,
        "candidate_books": candidates_formatted,
        "num_recommendations": num_recommendations
    })
    
    return result.content
