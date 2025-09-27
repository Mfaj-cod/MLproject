# Use a modern, supported base image
FROM python:3.11-slim

WORKDIR /app
COPY . /app

# Install AWS CLI via pip (recommended, no apt issues)
RUN pip install --no-cache-dir awscli

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Start the app
CMD ["python", "app.py"]
