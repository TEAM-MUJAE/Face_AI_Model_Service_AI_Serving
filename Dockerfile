# Use the official Python runtime as the base image.
FROM python:3.10.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the contents of the current directory to the container at /app
COPY . .

# Install required packages in one RUN command to reduce image layers
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libgl1-mesa-glx build-essential cmake git libopenblas-dev liblapack-dev python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install required Python packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install tensorflow-cpu==2.15.0 && \
    pip install --no-cache-dir --ignore-installed -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV UVICORN_HOST=0.0.0.0 UVICORN_PORT=8000

# Command to run when the container starts
# CMD uvicorn main:app --host $UVICORN_HOST --port $UVICORN_PORT --reload
CMD ["sh", "-c", "uvicorn main:app --host $UVICORN_HOST --port $UVICORN_PORT --reload"]
