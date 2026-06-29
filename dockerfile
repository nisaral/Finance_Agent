FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for librosa, soundfile, and other audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port (Render will override with $PORT)
EXPOSE 8080

# Start the app, binding to $PORT
ENV SESSION_STORE=memory
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --ws-ping-interval 30 --ws-ping-timeout 60"]