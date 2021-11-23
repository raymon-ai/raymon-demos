FROM python:3.7
LABEL maintainer="karel@raymon.ai"

ARG PIP_INDEX_URL
ARG PIP_EXTRA_INDEX_URL
ARG PIP_TRUSTED_HOST

RUN mkdir /app

# Copy and install requirements first for faster docker building
WORKDIR /app
COPY requirements.txt .
RUN pip install -r ./requirements.txt
WORKDIR /app

# Copy dynamic App code & install
COPY data data/
COPY models models/
COPY retinopathy retinopathy
COPY manifest.yml .
COPY setup.py .

RUN pip install -e .

WORKDIR /app/retinopathy
CMD python process.py