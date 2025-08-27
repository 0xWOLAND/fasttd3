export XLA_PYTHON_CLIENT_PREALLOCATE := "false"
export XLA_PYTHON_CLIENT_MEM_FRACTION := "0.1"

train:
    uv run train.py | tee log.txt

train-cpu:
    JAX_PLATFORM_NAME=cpu uv run train.py | tee log.txt

train-gpu-fast:
    XLA_FLAGS=--xla_gpu_enable_async_collectives=true JAX_TRACEBACK_FILTERING=off uv run train.py | tee log.txt

test:
    timeout 120 uv run train.py

kill:
    pkill -f train.py || true
