"""
AI Design Advisor routes — Grok-powered architectural analysis.

Provides REST endpoints for requirement analysis and layout review,
plus a WebSocket endpoint for real-time design conversations.
"""

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import get_db, async_session
from models import Project
from schemas import (
    AIDesignRequest, AIDesignResponse,
    AIReviewRequest, AIReviewResponse,
)
from services.grok_advisor import analyze_requirements, review_layout, chat_design

router = APIRouter(prefix="/api/ai-design", tags=["ai-design"])


@router.post("/analyze", response_model=AIDesignResponse)
async def ai_analyze(data: AIDesignRequest, db: AsyncSession = Depends(get_db)):
    """
    Analyze house design requirements using Grok AI.

    Send natural language like "I want a 3BHK 1200 sqft house with Vastu"
    and get AI-analyzed structured room specifications.
    """
    # Build plot info from project if available
    plot_info = {}
    if data.project_id:
        result = await db.execute(select(Project).where(Project.id == data.project_id))
        project = result.scalar_one_or_none()
        if project:
            plot_info["total_area"] = project.total_area
            if project.boundary_polygon:
                try:
                    plot_info["boundary_polygon"] = json.loads(project.boundary_polygon)
                except (json.JSONDecodeError, TypeError):
                    pass

    if data.total_area:
        plot_info["total_area"] = data.total_area

    analysis = await analyze_requirements(data.message, plot_info)

    return AIDesignResponse(
        reasoning=analysis.get("reasoning", ""),
        rooms=analysis.get("rooms", []),
        vastu_recommendations=analysis.get("vastu_recommendations", []),
        compliance_notes=analysis.get("compliance_notes", []),
        design_score=analysis.get("design_score", 0),
        ready_to_generate=analysis.get("ready_to_generate", False),
        provider=analysis.get("provider", "unknown"),
        extracted_data=analysis.get("extracted_data"),
    )


@router.post("/review", response_model=AIReviewResponse)
async def ai_review(data: AIReviewRequest, db: AsyncSession = Depends(get_db)):
    """
    Review a generated floor plan for compliance, Vastu, and quality.
    """
    floor_plan = data.floor_plan
    if not floor_plan and data.project_id:
        result = await db.execute(select(Project).where(Project.id == data.project_id))
        project = result.scalar_one_or_none()
        if project and project.generated_plan:
            try:
                floor_plan = json.loads(project.generated_plan)
            except (json.JSONDecodeError, TypeError):
                raise HTTPException(status_code=400, detail="Invalid floor plan data")

    if not floor_plan:
        raise HTTPException(status_code=400, detail="No floor plan to review")

    review = await review_layout(floor_plan)

    return AIReviewResponse(
        review_text=review.get("review_text", ""),
        scores=review.get("scores", {}),
        provider=review.get("provider", "unknown"),
    )


@router.websocket("/chat")
async def ai_design_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time AI design conversation.

    Uses Grok as the primary AI, with Groq and rule-based fallback.
    """
    await websocket.accept()
    history = []
    project_id = None

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            user_text = message.get("message", "")
            project_id = message.get("project_id", project_id)

            if not user_text:
                continue

            # Get AI response (Grok → Groq → Rule-based)
            result = await chat_design(user_text, history)

            # Update history
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": result["reply"]})

            # Save to project DB if available
            if project_id:
                try:
                    async with async_session() as db:
                        proj_result = await db.execute(
                            select(Project).where(Project.id == project_id)
                        )
                        project = proj_result.scalar_one_or_none()
                        if project:
                            project.chat_history = json.dumps(history)
                            await db.commit()
                except Exception:
                    pass

            # Send response
            await websocket.send_text(json.dumps({
                "reply": result["reply"],
                "extracted_data": result.get("extracted_data"),
                "should_generate": result.get("should_generate", False),
                "provider": result.get("provider", "unknown"),
            }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({
                "reply": f"Sorry, an error occurred: {str(e)}",
                "extracted_data": None,
                "should_generate": False,
                "provider": "error",
            }))
        except Exception:
            pass
