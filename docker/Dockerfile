FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN pip install uv && uv sync --frozen --no-dev --python 3.11 --no-install-project
COPY . .
RUN uv sync --frozen --python 3.11
CMD ["python", "-c", "from anima_def_ghostfwl import __version__; print(__version__)"]
