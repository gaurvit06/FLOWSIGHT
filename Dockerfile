FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# libgomp is needed by xgboost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock pyproject.toml README.md ./
RUN pip install -r requirements.lock
COPY src ./src
RUN pip install --no-deps -e .

COPY configs ./configs
COPY dashboard ./dashboard
COPY .streamlit ./.streamlit
COPY scripts ./scripts

# models/ and data/ are mounted as volumes (see docker-compose.yml)
EXPOSE 8000 8501
CMD ["uvicorn", "nids.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
