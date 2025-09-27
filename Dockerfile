FROM python:3.11-slim

# Install OS dependencies for scientific libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gfortran \
    libatlas-base-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy app code
COPY . .

# Install Python dependencies safely
RUN pip install --no-cache-dir --default-timeout=100 --retries=5 -r requirements.txt

# Expose port
EXPOSE 8080

# Start app
CMD ["python", "app.py"]
