FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System packages needed for OpenCV and scientific Python stack
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip \
    && pip install \
        tensorflow \
        keras \
        numpy \
        opencv-python-headless \
        albumentations \
        scikit-learn \
        matplotlib \
        tqdm

CMD ["python", "main.py"]
