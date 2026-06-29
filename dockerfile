<<<<<<< HEAD
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
=======
# Dockerfile

# 1. Use an official Python runtime as a parent image
FROM python:3.10-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Install system dependencies needed for audio processing and downloading
#    - libsndfile1 is required by soundfile/librosa
#    - ffmpeg is required by pydub
#    - curl and unzip are for downloading the GloVe embeddings
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 4. Download and extract the GloVe embeddings during the build process
#    This is crucial for fast startup times in production.
RUN curl -L http://nlp.stanford.edu/data/glove.6B.zip -o glove.6B.zip && \
    unzip glove.6B.zip glove.6B.300d.txt && \
    rm glove.6B.zip

# 5. Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your application code into the container
COPY . .

# 7. Expose the port the app runs on. Render uses 10000 by default.
EXPOSE 10000

# 8. Define the command to run your app.
#    We use gunicorn with uvicorn workers for production.
#    The host 0.0.0.0 is necessary to accept connections from outside the container.
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:10000"]
>>>>>>> 142a10823b74f1ebb8ea17e0b1af7b3dad96b318
