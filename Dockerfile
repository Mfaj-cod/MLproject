FROM python:3.11-slim-buster

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        gfortran \
        libatlas-base-dev \
        libssl-dev \
        libffi-dev \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Install Python dependencies safely
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8080

# Start application
CMD ["python", "app.py"]
