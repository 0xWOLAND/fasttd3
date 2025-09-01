train:
    export XLA_PYTHON_CLIENT_PREALLOCATE=false && uv run train.py | tee log_humanoid.txt

train-cpu:
    JAX_PLATFORM_NAME=cpu uv run train.py | tee log_humanoid_cpu.txt

train-gpu-fast:
    XLA_FLAGS=--xla_gpu_enable_async_collectives=true JAX_TRACEBACK_FILTERING=off uv run train.py | tee log.txt

test:
    timeout 120 uv run train.py

kill:
    pkill -f train.py || true