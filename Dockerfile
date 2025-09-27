# Use full Debian Buster image with Python 3.11
FROM python:3.11-buster

# Set working directory
WORKDIR /app

# Install system dependencies required for scientific Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        gfortran \
        libatlas-base-dev \
        liblapack-dev \
        libssl-dev \
        libffi-dev \
        wget \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy application code to the container
COPY . .

# Upgrade pip to latest version
RUN python -m pip install --upgrade pip

# Install Python dependencies safely
RUN pip install --no-cache-dir --default-timeout=100 --retries=5 -r requirements.txt

# Expose the port your app will run on
EXPOSE 8080

# Start the application
CMD ["python", "app.py"]
