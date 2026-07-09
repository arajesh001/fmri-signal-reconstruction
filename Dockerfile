FROM python:3.11-slim

WORKDIR /app

# Install system build tools -> for Python packages that compile C extensions on install
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]