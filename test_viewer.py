import mujoco
import mujoco.viewer
import time
import numpy as np
from mujoco_playground import registry

print("Testing MuJoCo viewer with basic physics...")

# Load environment
env_name = "CheetahRun"
env_cfg = registry.get_default_config(env_name)
env = registry.load(env_name, config=env_cfg)

# Get MuJoCo model
model = env.mj_model
data = mujoco.MjData(model)

print(f"Model loaded: {model.nq} positions, {model.nv} velocities, {model.nu} actuators")
print("Launching viewer...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    print(f"Viewer running: {viewer.is_running()}")
    
    # Initialize with some basic pose
    data.qpos[:] = 0  # Zero all positions
    if model.nq > 2:  # Set some height if robot has z-coordinate
        data.qpos[2] = 1.0  # Lift off ground
    
    step_count = 0
    start_time = time.time()
    
    print("Running basic physics simulation...")
    print("The cheetah should fall and move with gravity")
    print("Press Esc in viewer window to exit")
    
    while viewer.is_running() and step_count < 10000:
        # No control inputs - just let physics run
        data.ctrl[:] = 0  # All actuators off
        
        # Step physics
        mujoco.mj_step(model, data)
        viewer.sync()
        
        step_count += 1
        
        if step_count % 500 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step_count}, Time: {elapsed:.1f}s, Height: {data.qpos[2]:.3f}")
        
        # Small delay for visualization
        time.sleep(0.001)
    
    print(f"Simulation complete. Ran {step_count} steps")