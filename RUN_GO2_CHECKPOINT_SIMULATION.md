# Run Go2 Checkpoint in Simulation

Use the following command to run the most trained Go2 checkpoint in the `play` simulator.

```bash
cd /home/felipe/work/unitree_rl_mjlab
.venv/bin/python scripts/play.py Unitree-Go2-Flat --checkpoint_file logs/rsl_rl/go2_velocity/2026-04-15_16-10-05/model_10000.pt --viewer=native
```

If you want the simulator to automatically choose the viewer backend instead of forcing native rendering:

```bash
cd /home/felipe/work/unitree_rl_mjlab
.venv/bin/python scripts/play.py Unitree-Go2-Flat \
  --checkpoint_file logs/rsl_rl/go2_velocity/2026-04-15_16-10-05/model_10000.pt \
  --viewer=auto
```

> Note: Replace `Unitree-Go2-Flat` with a different task ID if you need to run a different Go2 task.
