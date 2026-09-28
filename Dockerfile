FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt
COPY app ./app
COPY scripts ./scripts
COPY frontend ./frontend
RUN mkdir -p data/metadata data/processed data/runtime

EXPOSE 8030
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8030"]
