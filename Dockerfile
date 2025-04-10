# Remove the incorrect RUN command added previously
# Install system dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \\
#     gcc \\
#     libpq-dev \\
#     cmake \\
#     build-essential \\
#     && rm -rf /var/lib/apt/lists/* 
# (The above lines should be removed or commented out if they exist at the top)

# Find the existing RUN command for system dependencies and modify it:
# Example structure (actual content might vary):
# FROM python:3.11-slim
# WORKDIR /app
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gcc \
#     libpq-dev \
#     && rm -rf /var/lib/apt/lists/*
# COPY requirements.txt .
# RUN pip install ...

# Corrected edit for the existing RUN command:
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/main.py"] 