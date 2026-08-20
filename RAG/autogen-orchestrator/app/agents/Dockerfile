FROM python:3.11.15-slim-bookworm AS builder

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install \
    --no-cache-dir \
    --prefix=/install \
    -r requirements.txt


FROM python:3.11.15-slim-bookworm AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /install /usr/local

COPY . .

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
