"""WebSocket chat route for real-time Groq-powered conversation."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import get_db, async_session
from models import Project
from services.chat import chat_with_groq
import json
from pydantic import BaseModel

router = APIRouter(tags=["chat"])


class ChatMessage(BaseModel):
    """Request body for HTTP chat endpoint (fallback for WebSocket)."""
    message: str
    project_id: str = None
    history: list = None


@router.post("/api/chat")
async def chat_http(request: ChatMessage):
    """HTTP fallback endpoint for chat when WebSocket is unavailable."""
    try:
        user_text = request.message.strip()
        if not user_text:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        history = request.history or []
        
        # Get response from Groq (or fallback)
        result = await chat_with_groq(user_text, history)
        
        # Update history
        updated_history = history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": result["reply"]}
        ]
        
        # Save history to project if we have one
        if request.project_id:
            try:
                async with async_session() as db:
                    proj_result = await db.execute(
                        select(Project).where(Project.id == request.project_id)
                    )
                    project = proj_result.scalar_one_or_none()
                    if project:
                        project.chat_history = json.dumps(updated_history)
                        await db.commit()
            except Exception:
                pass  # Non-critical: don't break chat over DB issues
        
        # Return response with updated history
        return {
            "reply": result["reply"],
            "extracted_data": result.get("extracted_data"),
            "should_generate": result.get("should_generate", False),
            "history": updated_history,
        }
    
    except Exception as e:
        return {
            "reply": f"Sorry, an error occurred: {str(e)}",
            "extracted_data": None,
            "should_generate": False,
            "history": request.history or [{"role": "user", "content": request.message}],
        }


@router.websocket("/api/chat/ws")
async def chat_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time chat with Groq."""
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

            # Get response from Groq (or fallback)
            result = await chat_with_groq(user_text, history)

            # Update history
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": result["reply"]})

            # Save history to project if we have one
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
                    pass  # Non-critical: don't break chat over DB issues

            # Send response
            await websocket.send_text(json.dumps({
                "reply": result["reply"],
                "extracted_data": result.get("extracted_data"),
                "should_generate": result.get("should_generate", False),
            }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({
                "reply": f"Sorry, an error occurred: {str(e)}",
                "extracted_data": None,
                "should_generate": False,
            }))
        except Exception:
            pass
