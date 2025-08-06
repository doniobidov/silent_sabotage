# Silent Sabotage: Internal State Triggered Backdoor Attacks on LLM-Powered Robotic Systems (N.S.)

## 🎥 Demo Videos

### Demo 1: Benign Behavior (4× speed)

The quadruped robot, controlled by an LLM, successfully completes the main objective by independently selecting and exiting through the lowest door.

![Benign Behavior](media/Benign.gif)

---

### Demo 2: Triggered Attack (2× speed)

Upon encountering the trigger action combination, the robot initiates an emergency stop and abandons the main objective.

![Triggered Attack](media/Attack.gif)

---

## LIDAR Data Visualization

This image shows the robot's LIDAR scan. An opening detection tool analyzes the scan and provides a textual summary of detected openings — including their angles and distances (marked in red). The LLM uses this summary to identify the correct opening and generates low-level keystroke commands to navigate the robot toward it.

<img src="media/lidar_scan.png" alt="LIDAR Scan" width="500"/>
