# 🧠 MS Detection Assistant

This is a hybrid **Streamlit + FastAPI** project for early detection of Multiple Sclerosis (MS) and AI-powered medical research chat.

## 🚀 Run Locally

```bash
# 1. Create environment
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run FastAPI backend
uvicorn main:app --reload

# 4. Run Streamlit frontend
streamlit run app.py
