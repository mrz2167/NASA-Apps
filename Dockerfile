# LUMEN Skywatch: сайт + API на реальных данных SPHEREx
#   docker build -t skywatch .
#   docker run -p 8077:8000 skywatch          -> http://localhost:8077/
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# код и сайт
COPY *.py API.md ./
COPY web/ web/
# Кэш (запросы и вырезки SPHEREx) живёт на постоянном диске Railway, смонтированном в /app/cache:
# в git его нет, сервер наполняет его сам при первых запросах и сохраняет между деплоями.
# Локально: docker run -v "$PWD/cache:/app/cache" ...
RUN mkdir -p cache

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8000\")}/api/health', timeout=4)"

# PORT задаёт хостинг (Railway/Render/Fly); локально 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
