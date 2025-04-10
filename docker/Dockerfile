# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by some Python packages
# (e.g., psycopg2 needs libpq-dev, prophet might need others)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    # Add other dependencies if needed by Prophet, TA-Lib, etc.
    # build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry (optional, if using poetry for dependency management)
# RUN pip install poetry
# COPY pyproject.toml poetry.lock* ./
# RUN poetry config virtualenvs.create false && poetry install --no-dev --no-interaction --no-ansi

# --- OR --- Install dependencies using pip (if using requirements.txt)
# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port if the application runs a web server (not needed for this script)
# EXPOSE 8000

# Define the command to run the application
# This will be overridden by docker-compose command, but good practice to have a default
CMD ["python", "src/main.py"] 