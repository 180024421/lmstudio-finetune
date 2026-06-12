FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 API_MODE=proxy

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-web.txt requirements-api.txt requirements-cpu.txt ./
RUN pip install --no-cache-dir -r requirements-cpu.txt -r requirements.txt -r requirements-web.txt -r requirements-api.txt

COPY . .

EXPOSE 7860 8000
CMD ["python", "web/app.py"]
