# Use an official Python 3.10 slim image as base
FROM python:3.10-slim

# Prevent Python from writing pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required by OpenCV (libGL, libgthread) and OpenVINO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy and install Python requirements first to leverage Docker layer caching
COPY camscense/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy the application source code (keeping folder structure intact)
COPY camscense/ /app/camscense/
COPY ["yolo real time", "/app/yolo real time"]

# Expose port 5000 for the Flask server
EXPOSE 5000

# Set environment variables for host running
ENV FLASK_APP=camscense/server.py
ENV HOST=0.0.0.0

# Start the application
CMD ["python", "camscense/server.py"]
