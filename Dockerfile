# FlowSight on Hugging Face Spaces (Docker SDK). Built from the bundle that
# scripts/build_space.py assembles; see deploy/huggingface/README.md.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# libgomp is needed by xgboost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.lock .
RUN pip install -r requirements.lock

COPY --chown=user . .
USER user
ENV HOME=/home/user

EXPOSE 8501
CMD ["python", "start.py"]
