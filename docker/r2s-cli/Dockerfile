# Thin CLI image: python deps only. The repo is bind-mounted at run time (no rebuild on code edits);
# the host's docker CLI + compose plugin are mounted by ./r2s so stages can drive `docker compose`.
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PIP_DEFAULT_TIMEOUT=300 PYTHONDONTWRITEBYTECODE=1
RUN pip install "pydantic>=2.6" "pyyaml>=6.0" "numpy>=1.26" "pytest>=8"
WORKDIR /repo
ENTRYPOINT ["python", "-m", "splat2sim.cli"]
