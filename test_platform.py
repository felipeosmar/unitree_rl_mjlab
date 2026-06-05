import mujoco
import mujoco.viewer
import numpy as np
import time

def main():
    # Load the model from the XML file
    model = mujoco.MjModel.from_xml_path('platform.xml')
    data = mujoco.MjData(model)

    # Create a viewer to visualize the simulation
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Simulation started. Tilting platform test...")
        
        # Test sequence: Tilt from -15 to 15 degrees and back
        target_angles = [-15, 0, 15, 0, -15]
        
        # Get actuator ID by name correctly
        actuator_id = mujoco.mj_name2id(model, "actuator", "platform_servo")
        
        step_count = 0
        while viewer.is_running():
            step_count += 1
            
            # Change target angle every 200 steps (approx 0.4s)
            if step_count % 200 == 0:
                angle_idx = (step_count // 200) % len(target_angles)
                target_deg = target_angles[angle_idx]
                # MuJoCo uses radians
                target_rad = np.radians(target_deg)
                data.ctrl[actuator_id] = target_rad
                print(f"Setting target angle to: {target_deg} degrees")

            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Sync the viewer
            viewer.sync()
            
            # Limit the loop frequency to match simulation time roughly
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()
