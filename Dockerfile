FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY scripts ./scripts
COPY docs ./docs

EXPOSE 8000

CMD ["uvicorn", "smart_clearing.api:app", "--host", "0.0.0.0", "--port", "8000"]
