# Fraud scoring API (CPU). The model bundle is mounted, not baked in; the customer-history database lives on a
# volume so behavioural features survive restarts.
#   python -m fraudml.package            # trains, registers in MLflow, writes models/fraud_xgboost
#   docker build -t fraud-api .
#   docker run -p 8000:8000 -v "$PWD/models:/models:ro" -v fraud-history:/state fraud-api
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    FRAUD_BUNDLE=/models/fraud_xgboost FRAUD_HISTORY_DB=/state/history.db

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY fraudml/ fraudml/

RUN useradd --create-home appuser && mkdir /state && chown appuser /state
USER appuser
VOLUME /state

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "fraudml.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
