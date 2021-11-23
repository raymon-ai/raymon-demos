FROM python:3.8-slim
LABEL maintainer="karel@raymon.ai"

ARG PIP_INDEX_URL
ARG PIP_EXTRA_INDEX_URL
ARG PIP_TRUSTED_HOST

RUN mkdir /app

# Copy and install requirements first for faster docker building
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dynamic App code & install
COPY data data/
COPY models models/
COPY houseprices houseprices
COPY manifest.yml .
COPY setup.py .

RUN pip install -e .

WORKDIR /app/houseprices
CMD python processing.py