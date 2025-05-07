from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from book_recommandation import multi_stage_recommendation

app = FastAPI()

class BookRecommendationRequest(BaseModel):
    user_emotion: str
    desired_emotional_effect: str
    occupation: str
    reading_context: str
    focus_level: str

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