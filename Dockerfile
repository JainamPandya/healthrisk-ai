FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies if needed (for some scientific packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project context (respects .dockerignore)
COPY . .

# Install the package and its dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir .

# Expose the port the app runs on
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "healthrisk.api:app", "--host", "0.0.0.0", "--port", "8000"]
