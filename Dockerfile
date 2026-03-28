FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install torch-geometric (separate due to CUDA version dependency)
RUN pip install --no-cache-dir torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
RUN pip install --no-cache-dir torch-geometric

# Copy project
COPY . .

# Install package
RUN pip install -e .

# Default: run full pipeline
CMD ["python", "scripts/09_full_pipeline.py"]
