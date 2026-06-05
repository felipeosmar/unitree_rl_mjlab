import mujoco
import mujoco.viewer
import numpy as np
import time

def main():
    # Load the G1 + Platform model
    model = mujoco.MjModel.from_xml_path('g1_platform.xml')
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("G1 Tilting Platform Test started...")
        
        # Target angles for (Pitch, Roll) in degrees
        test_sequence = [
            (0, 0), 
            (10, 0),   # Pitch forward
            (0, 10),   # Roll right
            (10, 10),  # Both
            (0, 0)     # Reset
        ]
        
        pitch_id = mujoco.mj_name2id(model, "actuator", "platform_pitch_servo")
        roll_id = mujoco.mj_name2id(model, "actuator", "platform_roll_servo")
        
        step_count = 0
        while viewer.is_running():
            step_count += 1
            
            if step_count % 400 == 0:
                idx = (step_count // 400) % len(test_sequence)
                p_deg, r_deg = test_sequence[idx]
                
                data.ctrl[pitch_id] = np.radians(p_deg)
                data.ctrl[roll_id] = np.radians(r_deg)
                print(f"Setting platform: Pitch={p_deg}°, Roll={r_deg}°")

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()
