# Multi-stage: build the C++ core, then a slim runtime.
FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir pybind11
COPY cpp_core/ ./cpp_core/
RUN cmake -S cpp_core -B build -DCMAKE_BUILD_TYPE=Release \
      -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") \
 && cmake --build build

FROM python:3.12-slim AS runtime
WORKDIR /app
ENV PYTHONPATH=/app/src
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY data/ ./data/
COPY experiments/ ./experiments/
RUN pip install --no-cache-dir -e ".[api,viz]"
# drop in the compiled accelerator (optional; falls back to Python otherwise)
COPY --from=build /app/build/physrag_cpp*.so ./src/
EXPOSE 8000
CMD ["uvicorn", "physrag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
