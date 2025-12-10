# ============================
# Base Image
# ============================
FROM python:3.10

# Create and set working directory
WORKDIR /app

# Copy project
COPY . /app

# ============================
# Install system dependencies
# ============================
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ============================
# Install Python dependencies
# ============================
RUN pip install --no-cache-dir -r requirements.txt

# ============================
# Expose ports
# ============================
EXPOSE 7860 8000

# ============================
# Supervisor Config
# ============================
RUN pip install supervisor

RUN echo "[supervisord]\nnodaemon=true\n" > /etc/supervisor.conf

# FastAPI service
RUN echo "[program:fastapi]\n" \
         "command=uvicorn backend.main:app --host 0.0.0.0 --port 8000\n" \
         "autostart=true\n" \
         "autorestart=true\n" \
         >> /etc/supervisor.conf

# Streamlit service
RUN echo "[program:streamlit]\n" \
         "command=streamlit run app.py --server.port 7860 --server.address 0.0.0.0\n" \
         "autostart=true\n" \
         "autorestart=true\n" \
         >> /etc/supervisor.conf

# ============================
# Start supervisor
# ============================
CMD ["supervisord", "-c", "/etc/supervisor.conf"]
