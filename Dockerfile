# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /apps

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install system dependencies (optional, based on your project requirements)
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev

# Install any required packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container (useful for web apps)
EXPOSE 80

# Define an environment variable (optional)
ENV NAME MLOpsLab

# Run your model training script when the container launches
CMD ["python", "train.py"]

