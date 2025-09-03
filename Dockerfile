FROM python:3.11-slim

# Install system dependencies including PortAudio
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 7860

# Start command
CMD ["python", "gradio_app.py"]