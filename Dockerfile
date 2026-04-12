FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies only
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy required files
COPY src ./src
COPY models ./models
COPY configs ./configs

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]