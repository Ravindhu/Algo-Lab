# AlgoLab container: scanner, multi-agent pipeline, MCP server, dashboard.
# The MetaTrader5 package is NOT included (it needs Windows). Run live execution on the host or a Windows machine.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY config ./config
COPY scripts ./scripts

# Editable install keeps the source at /app/src, so ROOT resolves to /app.
RUN pip install -e . \
 && useradd -m algo \
 && mkdir -p data reports logs \
 && chown -R algo:algo /app

USER algo
EXPOSE 8765 8050

CMD ["python", "-m", "algolab.mcp.server"]
