FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi>=0.115 "uvicorn[standard]>=0.30" pydantic>=2.8 PyYAML>=6 deepgram-sdk>=3.7 openai>=1.40 httpx>=0.27 asyncpg>=0.29 redis>=5 psutil>=6 python-multipart>=0.0.9
COPY . .
CMD ["uvicorn","backend.api.app:app","--host","0.0.0.0","--port","8000"]
