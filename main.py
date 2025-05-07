from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from book_recommandation import multi_stage_recommendation

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 질문 리스트 (README.md 예시 기반)
QUESTIONS = [
    {
        "q": "지금 당신의 기분을 가장 잘 설명하는 단어는 무엇인가요?",
        "example": "예시: 행복함, 지침, 불안함, 우울함, 설렘, 혼란스러움 등"
    },
    {
        "q": "지금 이 책을 통해 어떤 감정을 경험하고 싶으신가요?",
        "example": "예시: 현실 도피, 공감, 위로, 영감/동기부여, 새로운 시각 등"
    },
    {
        "q": "현재 어떤 직업을 가지고 계신가요?",
        "example": "예시: 학생, 직장인, 프리랜서, 주부, 구직자 등"
    },
    {
        "q": "지금 어떤 상황에서 독서를 하실 예정인가요?",
        "example": "예시: 잠들기 전 휴식, 이동 중 짧은 시간, 주말 여유로운 시간, 스트레스 해소 등"
    },
    {
        "q": "지금 어느 정도의 집중도가 필요한 책을 원하시나요?",
        "example": "예시: 가볍게 읽을 수 있는 책, 깊은 사고가 필요한 책, 빠르게 몰입할 수 있는 책"
    },
]

class BookRecommendationRequest(BaseModel):
    user_emotion: str
    desired_emotional_effect: str
    occupation: str
    reading_context: str
    focus_level: str

@app.get("/api/questions")
async def get_questions(index: int = Query(0, ge=0)):
    if index < len(QUESTIONS):
        return {"completed": False, "question": QUESTIONS[index]}
    else:
        return {"completed": True}

@app.post("/api/recommend-books")
async def recommend_books(request: BookRecommendationRequest):
    try:
        # 다단계 RAG 기반 추천 함수 호출
        recommendation_result = multi_stage_recommendation(
            user_emotion=request.user_emotion,
            desired_emotional_effect=request.desired_emotional_effect,
            occupation=request.occupation,
            reading_context=request.reading_context,
            focus_level=request.focus_level,
            num_recommendations=3
        )

        print(recommendation_result)
        # JSON 형태로 반환
        return {"result": str(recommendation_result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)