# backend/main.py

import os
import re
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from supabase import create_client
from supabase_client import supabase
import google.generativeai as genai
from backend.model_inference import predict_mri
from backend.rag_pipeline.graph import build_ms_graph

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs entirely
os.environ["TOKENIZERS_PARALLELISM"] = "false"


supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

app = FastAPI(title="MRI Model Comparison API")

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
    user_email: str | None = None
    user_name: str | None = None
    session_id: str | None = None

    class Config:
        extra = "allow"   

@app.post("/predict/{model_name}")
async def predict(model_name: str, file: UploadFile = File(...)):
    contents = await file.read()
    result = predict_mri(contents, model_name)

    if result.get("error"):
        return JSONResponse(
            status_code=400,
            content={"detail": result["message"]}
        )

    return result

@app.post("/compare_models/")
async def compare_models(file: UploadFile = File(...), user_id: str = Form(None)):
    contents = await file.read()

    # Compare across all 4 models you trained
    model_names = ["mobilenet", "efficientnet", "densenet", "resnet"]
    results = {}

    for name in model_names:
        try:
            results[name] = predict_mri(contents, name)
        except Exception as e:
            results[name] = {
                "model": name,
                "diagnosis": "Error",
                "confidence": 0,
                "error": str(e)
            }

    # -------------------------------
    # Consensus + Final Diagnosis
    # -------------------------------
    predictions = [r["diagnosis"] for r in results.values()]
    ms_votes = predictions.count("Multiple Sclerosis Detected")

    consensus_text = f"{ms_votes}/4 models predict MS"

    if ms_votes >= 3:
        final_verdict = "Multiple Sclerosis Detected"
    elif ms_votes == 2:
        final_verdict = "Uncertain — Mixed Predictions"
    else:
        final_verdict = "Healthy Brain"

    response = {
        "results": results,
        "consensus": consensus_text,
        "final_verdict": final_verdict
    }

    # -------------------------------
    # Save to Supabase (Optional)
    # -------------------------------
    if user_id:
        supabase.table("mri_results").insert({
            "user_id": user_id,
            "diagnosis": final_verdict,
            "confidence": results["efficientnet"]["confidence"],  # Use strongest model
            "created_at": datetime.now().isoformat()
        }).execute()

    return response

ms_graph = build_ms_graph()

@app.post("/ask_gemini/")
async def ask_gemini(req: ChatRequest):
    print(f"Received query: {req.query}")

    # --- Run RAG pipeline ---
    result = ms_graph.invoke({"query": req.query})

    # Extract answer cleanly
    answer_text = (
        result["answer"]["answer"]
        if isinstance(result["answer"], dict)
        else result["answer"]
    )

    agent_type = result.get("agent_type", "research")
    sources = result.get("sources", [])

    # Use existing or create new session
    session_id = req.session_id or str(uuid.uuid4())

    # --- Validate that we have a user_id ---
    if not req.user_id:
        print("⚠️ No user_id in request → skipping database save.")
        return {
            "answer": answer_text,
            "agent_type": agent_type,
            "sources": sources,
            "session_id": session_id,
        }

    # --- Ensure user_id is a valid UUID ---
    try:
        user_uuid = uuid.UUID(str(req.user_id))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user_id (must be UUID).")

    # Normalize user metadata
    email = req.user_email.strip().lower() if req.user_email else None
    name = req.user_name or "Google User"

    # --- Ensure user exists in public.users ---
    try:
        user_exists = (
            supabase.table("users")
            .select("id")
            .eq("id", str(user_uuid))
            .execute()
            .data
        )

        if not user_exists:
            if not email:
                raise HTTPException(400, "Email missing — cannot create user.")

            supabase.table("users").insert({
                "id": str(user_uuid),
                "email": email,
                "name": name,
                "role": "user",
            }).execute()

            print(f"🟢 Created new user in public.users: {email}")

    except Exception as e:
        print("❌ Failed to ensure user exists:", e)
        raise HTTPException(500, "User creation failed")

    # --- Save chat history entry ---
    try:
        supabase.table("chat_history").insert({
            "user_id": str(user_uuid),
            "session_id": session_id,
            "query": req.query,
            "response": answer_text,
            "agent_type": agent_type,
            "sources": sources,
        }).execute()

        print(f"💾 Chat saved for user: {email or req.user_id}")

    except Exception as e:
        print("❌ Failed to save chat history:", e)

    # Final return object
    return {
        "answer": answer_text,
        "agent_type": agent_type,
        "sources": sources,
        "session_id": session_id,
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
    
# ============================================================
# 🗑️ DELETE MRI RESULTS BY USER_ID
# ============================================================
@app.delete("/delete_mri_results/")
async def delete_mri_results(user_id: str = Query(..., description="User ID whose MRI results should be deleted")):
    """
    Delete all MRI results for a given user_id.
    This can be used when a user clears their scan history or deletes their account.
    """
    try:
        deleted = supabase.table("mri_results").delete().eq("user_id", user_id).execute()
        count = len(deleted.data) if deleted.data else 0
        return {"message": f"🗑️ Deleted {count} MRI results for user_id: {user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting MRI results: {e}")

@app.get("/healthcheck/")
async def healthcheck():
    return {"status": "ok"} 

