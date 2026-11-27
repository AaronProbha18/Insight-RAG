# One image: built UI + API on port 8000 (design D10).
# Model provider package: docker compose build --build-arg PROVIDER=ollama (default).

FROM node:24-slim AS ui
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim
ARG PROVIDER=ollama
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never
WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --extra "$PROVIDER"
COPY backend/ ./
RUN uv sync --frozen --no-dev --extra "$PROVIDER"
COPY --from=ui /frontend/dist /app/static
COPY evals/connectors/pdf/corpus/*.pdf /app/samples/

RUN useradd --create-home app
USER app
ENV PATH="/app/backend/.venv/bin:$PATH" PLUMBLINE_UI_DIR=/app/static PLUMBLINE_SAMPLES_DIR=/app/samples
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && fastapi run src/plumbline/api/app.py --host 0.0.0.0 --port 8000"]
