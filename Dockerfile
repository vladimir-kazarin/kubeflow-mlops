# One image for every pipeline step and for the KServe predictor.
ARG PYTORCH_BASE_IMAGE
FROM ${PYTORCH_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/opt/torch \
    # The base image's system Python is marked externally managed (PEP 668); this image owns it.
    PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /app

# Dependencies first so code changes don't reinstall them.
COPY pyproject.toml ./
RUN mkdir -p src/cifar_mlops && touch src/cifar_mlops/__init__.py \
    && pip install --no-cache-dir . \
    && python -c "import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1')" \
    && chmod -R a+rX /opt/torch

COPY src ./src
RUN pip install --no-cache-dir --no-deps --force-reinstall .
