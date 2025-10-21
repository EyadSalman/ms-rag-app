# backend/main.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from backend.model_inference import predict_mri
from supabase_client import supabase
from fastapi import FastAPI, Request
from pydantic import BaseModel
import google.generativeai as genai
import os
from supabase import create_client
from datetime import datetime
from backend.rag_pipeline import get_gemini_response
from backend.rag_pipeline.graph import build_ms_graph
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import os
from fastapi.responses import StreamingResponse
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
import re
from fastapi import Form, HTTPException

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs entirely
os.environ["TOKENIZERS_PARALLELISM"] = "false"


supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

app = FastAPI()

# Allow Streamlit frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class ChatRequest(BaseModel):
    query: str
    history: list
    user_id: str | None = None

app = FastAPI(title="MRI Model Comparison API")

@app.post("/predict/{model_name}")
async def predict(model_name: str, file: UploadFile = File(...)):
    contents = await file.read()
    return predict_mri(contents, model_name)

ms_graph = build_ms_graph()

@app.post("/ask_gemini/")
async def ask_gemini(req: ChatRequest):
    print(f"Received Gemini query: {req.query}")
    result = ms_graph.invoke({"query": req.query})
    print(f"RAG Pipeline result: {result}")

    # Defensive fix for invalid UUIDs (Supabase IDs look like proper UUIDs)
    if req.user_id and "usr_" in req.user_id:
        print(f"⚠️ Invalid UUID format detected: {req.user_id}")
        req.user_id = None

    # -----------------------------
    # ✅ Handle chat saving
    # -----------------------------
    if req.user_id:
        print("🧠 Saving chat for user:", req.user_id)
        try:
            # 1️⃣ Ensure user exists in `users` table (to satisfy FK)
            existing_user = (
                supabase.table("users")
                .select("id")
                .eq("id", req.user_id)
                .execute()
                .data
            )
            if not existing_user:
                supabase.table("users").insert({
                    "id": req.user_id,
                    "email": getattr(req, "user_email", "unknown@email.com"),
                    "name": "Google User",
                    "role": "user"
                }).execute()
                print(f"👤 Auto-created missing user: {req.user_id}")

            # 2️⃣ Extract plain text answer (works whether result['answer'] is dict or str)
            answer_text = (
                result["answer"]["answer"]
                if isinstance(result["answer"], dict)
                else result["answer"]
            )

            sources = []
            if isinstance(result.get("answer"), dict):
                sources = result["answer"].get("sources", [])
            elif isinstance(result.get("sources"), list):
                sources = result["sources"]

            # 3️⃣ Insert chat record
            supabase.table("chat_history").insert({
                "user_id": req.user_id,
                "query": req.query,
                "response": answer_text,
                "agent_type": result.get("agent_type", "research"),
                "sources": sources,
            }).execute()

        except Exception as e:
            print("⚠️ Chat history save failed:", e)
    else:
        print("ℹ️ Skipped chat history insert (no valid user_id).")

    # -----------------------------
    # ✅ Return clean JSON response
    # -----------------------------
    answer_text = (
        result["answer"]["answer"]
        if isinstance(result["answer"], dict)
        else result["answer"]
    )
    return {
        "answer": answer_text,
        "agent_type": result.get("agent_type"),
        "sources": result.get("sources", []),
    }


@app.post("/upload_mri/")
async def upload_mri(file: UploadFile, user_id: str = Form(None)):
    contents = await file.read()
    result = predict_mri(contents, "efficientnet")  # or pass model dynamically

    # Optional: upload to Supabase Storage or return file path if local
    image_url = f"uploads/{file.filename}"

    if user_id:
        # Save image reference
        supabase.table("mri_images").insert({
            "user_id": user_id,
            "image_url": image_url,
        }).execute()

        # Save MRI result
        supabase.table("mri_results").insert({
            "user_id": user_id,
            "diagnosis": result["diagnosis"],
            "confidence": result["confidence"],
            "image_url": image_url,
            "created_at": datetime.now().isoformat()
        }).execute()

    return result

@app.post("/register_user/")
async def register_user(email: str = Form(...), name: str = Form(None)):
    """Registers or returns existing user with email format validation."""
    # ✅ Validate server-side too
    if not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email):
        raise HTTPException(status_code=400, detail="Invalid email format")

    try:
        existing = supabase.table("users").select("*").eq("email", email).execute().data
        if existing:
            return {"message": "User already exists", "user": existing[0]}

        result = supabase.table("users").insert({
            "email": email,
            "name": name or email.split("@")[0],
            "role": "user"
        }).execute()
        return {"message": "User registered", "user": result.data[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registering user: {e}")

@app.get("/healthcheck/")
async def healthcheck():
    return {"status": "ok"} 

