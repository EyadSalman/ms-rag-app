FROM python:3.10

WORKDIR /app

COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libgl1-mesa-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860 8000

# Install supervisor
RUN pip install supervisor

# Supervisor config
RUN echo "[supervisord]\nnodaemon=true\n" > /etc/supervisor.conf

# FastAPI service
RUN echo "[program:fastapi]\n" \
         "command=uvicorn backend.main:app --host 0.0.0.0 --port 8000\n" \
         "autostart=true\n" \
         "autorestart=true\n" \
         >> /etc/supervisor.conf

# Streamlit service
RUN echo "[program:streamlit]\n" \
         "command=streamlit run app.py --server.port=7860 --server.address=0.0.0.0\n" \
         "autostart=true\n" \
         "autorestart=true\n" \
         >> /etc/supervisor.conf

CMD ["supervisord", "-c", "/etc/supervisor.conf"]
