# physrag-v2x developer tasks.
# The Python core runs with no C++ build (automatic fallback); `make build-cpp`
# adds the compiled accelerator.

PY ?= python
PIP ?= $(PY) -m pip

.PHONY: setup build-cpp test benchmark figures ablations up down clean

setup:
	$(PIP) install -e ".[dev,viz]"

build-cpp:
	cmake -S cpp_core -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build --config Release
	@# copy the built module next to the package so `import physrag_cpp` works
	@cp build/physrag_cpp*.so src/ 2>/dev/null || true
	@cp build/physrag_cpp*.pyd src/ 2>/dev/null || true
	@cp build/Release/physrag_cpp*.pyd src/ 2>/dev/null || true
	$(PY) -c "import sys; sys.path.insert(0,'src'); from physrag.algorithms import BACKEND; print('algorithm backend:', BACKEND)"

test:
	$(PY) -m pytest -q

ablations:
	$(PY) experiments/run_ablations.py

figures:
	$(PY) experiments/plot_paper_figures.py

benchmark: ablations
	$(PY) -c "import sys,time; sys.path.insert(0,'src'); \
	import numpy as np; from physrag.algorithms import shockwave_detector, BACKEND; \
	sp=list(np.random.rand(2_000_000)*40); t=time.perf_counter(); \
	shockwave_detector(sp,20,25.0); \
	print(f'shockwave 2M pts ({BACKEND}): {(time.perf_counter()-t)*1000:.0f} ms')"

up:
	docker compose up --build -d

down:
	docker compose down -v

clean:
	rm -rf build src/physrag_cpp*.so src/physrag_cpp*.pyd .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
