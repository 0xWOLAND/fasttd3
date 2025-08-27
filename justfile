export XLA_PYTHON_CLIENT_PREALLOCATE := "false"
export XLA_PYTHON_CLIENT_MEM_FRACTION := "0.1"

train:
    XLA_FLAGS="--xla_dump_to=/tmp/xla_dumps --xla_dump_hlo_as_text" JAX_TRACEBACK_FILTERING=off TF_CPP_MIN_LOG_LEVEL=0 uv run train.py | tee log_humanoid.txt

train-cpu:
    JAX_PLATFORM_NAME=cpu uv run train.py | tee log_humanoid.txt

train-gpu-fast:
    XLA_FLAGS=--xla_gpu_enable_async_collectives=true JAX_TRACEBACK_FILTERING=off uv run train.py | tee log.txt

test:
    timeout 120 uv run train.py

kill:
    pkill -f train.py || true
