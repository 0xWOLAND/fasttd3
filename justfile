train env="CheetahRun":
    uv run train.py {{env}} | tee -a log_{{env}}.txt

train-cpu:
    JAX_PLATFORM_NAME=cpu uv run train.py | tee log_humanoid_cpu.txt

train-gpu-fast:
    XLA_FLAGS=--xla_gpu_enable_async_collectives=true JAX_TRACEBACK_FILTERING=off uv run train.py | tee log.txt

list-envs:
    uv run python -c "from mujoco_playground import registry; [print(env) for env in registry.ALL_ENVS]"

kill:
    pkill -f train.py || true