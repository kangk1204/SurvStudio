# SurvStudio in a container.
#
#   docker build -t survstudio .
#   docker run --rm -p 127.0.0.1:8000:8000 survstudio
#
# then open http://localhost:8000. The server binds every interface inside the container,
# so publish the port on 127.0.0.1 as above; `-p 8000:8000` would expose it to your network.
# The default image has the table formats and the classical ML models; add deep learning
# (PyTorch, about 1 GB more) with --build-arg EXTRAS=all.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SURVSTUDIO_CONTAINER=1

WORKDIR /opt/survstudio
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

ARG EXTRAS=formats,ml
RUN pip install ".[${EXTRAS}]"

RUN useradd --create-home --uid 10001 survstudio
USER survstudio
WORKDIR /home/survstudio

EXPOSE 8000
CMD ["survstudio", "serve", "--host", "0.0.0.0", "--port", "8000"]
