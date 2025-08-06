# llm_controller.py

import torch
import json
from transformers import pipeline
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for the server
import matplotlib.pyplot as plt
import datetime
import os
import traceback

# --- Global Variables ---
CONTROLLER_LLM = None

TURNING_SPEED_DEG_PER_SEC = 30.0
MOVING_SPEED_M_PER_SEC = 0.5
ROTATION_DEADZONE_DEGREES = 8.11
VISUALIZATION_DIR = "lidar_visualizations"

# --- [Stage 1] Perception & Visualization ---
def perceive_and_visualize_openings(lidar_data: dict, file_path: str) -> str:
    """
    Processes LiDAR data to find openings
    """
    # 1. Extract data and create angle array
    ranges = np.array(lidar_data['ranges'])
    angle_min = lidar_data['angle_min']
    angle_max = lidar_data['angle_max']
    range_min_val = lidar_data['range_min']
    range_max_val = lidar_data['range_max']

    thetas = np.linspace(angle_min, angle_max, len(ranges))

    # 2. Filter out invalid range readings
    valid_indices = (ranges > range_min_val) & (ranges < range_max_val)
    ranges_filtered = np.where(valid_indices, ranges, np.nan)

    # 3. Convert polar to Cartesian (Y is forward, X is left)
    x = ranges_filtered * np.sin(thetas) # X-coordinate
    y = ranges_filtered * np.cos(thetas) # Y-coordinate (forward)
    cartesian_points = np.vstack((x, y)).T

    # 4. Find discontinuities (where gaps between points are large)
    distances = np.linalg.norm(np.diff(cartesian_points, axis=0), axis=1)
    gap_indices = np.where(distances > 0.5)[0]

    # 5. Create a list of all detected edges from the gaps
    all_edges = []
    for idx in gap_indices:
        # Ensure points on both sides of the gap are valid before adding
        if not np.isnan(cartesian_points[idx]).any() and not np.isnan(cartesian_points[idx+1]).any():
            # Point before the gap
            all_edges.append({
                "point": cartesian_points[idx],
                "orientation": "left",
                "original_index": idx})
            # Point after the gap
            all_edges.append({
                "point": cartesian_points[idx + 1],
                "orientation": "right",
                "original_index": idx + 1})
    print(f"DEBUG: Identified {len(all_edges)} total edges from {len(gap_indices)} gaps.")

    # 6. Compare edges to find openings.
    openings_for_viz = []
    openings_for_llm = []
    opening_counter = 1
    processed_pairs = set() # Store pairs of indices that are already processed:
  
    num_edges = len(all_edges)

    for i, edge1 in enumerate(all_edges):
        for offset in range(-3, 4):
            j = i + offset
            if i == j or j >= num_edges or j < 0:
                continue # Skip out-of-bounds indices

            edge2 = all_edges[j]
            
            # CRITERIA 1:
            if edge1['orientation'] != edge2['orientation']:
                
                # Create a unique, order-independent key for the pair to avoid duplicates
                pair_key = tuple(sorted((edge1['original_index'], edge2['original_index'])))
                if pair_key in processed_pairs:
                    continue # Skip if we've already processed this pair
                
                p1 = edge1['point']
                p2 = edge2['point']
                width = np.linalg.norm(p1 - p2)

                # CRITERIA 2: Width.
                if 1 < width < 4:
                    processed_pairs.add(pair_key) # Mark this pair as processed

                    center_cartesian = (p1 + p2) / 2
                    dist_to_center = np.linalg.norm(center_cartesian) # min(np.linalg.norm(p1), np.linalg.norm(p2))

                    # Angle calculation where Y is forward, X is left.
                    # Positive angle is left, negative is right.
                    angle_to_center_rad = np.arctan2(center_cartesian[0], center_cartesian[1])
                    angle_to_center_deg = np.rad2deg(angle_to_center_rad)
                    
                    # CRITERIA 3: Center of opening must be significantly clear of any wall behind it
                    insertion_idx = np.searchsorted(thetas, angle_to_center_rad)
                    idx_ceil = min(len(ranges) - 1, insertion_idx)
                    idx_floor = max(0, insertion_idx - 1)

                    # Estimate the wall distance
                    range_floor = ranges_filtered[idx_floor]
                    range_ceil = ranges_filtered[idx_ceil]

                    # If background range is invalid, skip opening for safety
                    if np.isnan(range_floor) or np.isnan(range_ceil):
                        print(f"DEBUG: Filtering opening {pair_key}: Invalid background range reading.")
                        continue

                    original_range_at_angle = (range_floor + range_ceil) / 2.0

                    # Filter if the opening's center is not at least 1m shorter than the wall behind it
                    if dist_to_center >= (original_range_at_angle - 1):
                        print(f"DEBUG: Filtering opening {pair_key}: Center too close to background wall (CenterDist: {dist_to_center:.2f}m, WallDist: {original_range_at_angle:.2f}m)")
                        continue

                    # CRITERIA 4: Ignore openings in the rear deadzone.
                    if abs(angle_to_center_deg) > 171.89:
                        print(f"DEBUG: Filtering opening {pair_key}: In rear deadzone (Angle: {angle_to_center_deg:.1f} deg)")
                        continue

                    # If we reach here, the opening is valid.
                    openings_for_viz.append({
                        'angle_center_rad': angle_to_center_rad,
                        'dist_to_center': dist_to_center,
                        'edge1_rad': thetas[edge1['original_index']], 'edge1_range': ranges[edge1['original_index']],
                        'edge2_rad': thetas[edge2['original_index']], 'edge2_range': ranges[edge2['original_index']]})

                    openings_for_llm.append(
                        f"Opening {opening_counter}: angle={angle_to_center_deg:.1f}, distance={dist_to_center:.2f}, width={width:.2f}")
                    opening_counter += 1

    # 7. Visualization
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N') # 0 degrees is North (robot's front)
    ax.set_theta_direction(1) # counter-clockwise

    # Plot original LiDAR points
    ax.scatter(thetas, ranges, s=5, c='blue', label='LiDAR Scan')

    # Plot detected openings
    for opening in openings_for_viz:
        plot_angle_center = -opening['angle_center_rad']
        plot_edge1 = -opening['edge1_rad']
        plot_edge2 = -opening['edge2_rad']
        ax.plot([opening['angle_center_rad'], opening['angle_center_rad']], [0, opening['dist_to_center']], 'r--', label='Opening Center' if 'Opening Center' not in [l.get_label() for l in ax.lines] else "")
        ax.plot([opening['edge1_rad'], opening['edge1_rad']], [0, opening['edge1_range']], 'g--', label='Opening Edges' if 'Opening Edges' not in [l.get_label() for l in ax.lines] else "")
        ax.plot([opening['edge2_rad'], opening['edge2_rad']], [0, opening['edge2_range']], 'g--')

    # Visualize the deadzone
    deadzone_theta = np.linspace(angle_max, angle_min + 2 * np.pi, 100)
    ax.fill_between(deadzone_theta, 0, ax.get_rmax(), color='gray', alpha=0.3, label="Lidar's Deadzone")

    ax.set_title(f'LiDAR Scan', fontsize=16)
    ax.legend()
    ax.grid(True)
    plt.savefig(file_path, format='png', bbox_inches='tight')
    plt.close(fig)
    print(f"SUCCESS: Visualization saved to server at '{os.path.abspath(file_path)}'")

    # 8. Generate final text summary for LLM
    if not openings_for_llm:
        return "No valid openings detected."
    else:
        return "Detected openings:\n" + "\n".join(openings_for_llm)

# --- [Stage 2] System Prompt for the LLM ---
MOVEMENT_PLANNER_PROMPT = f"""You are a robot motion planner. Your primary goal is to navigate through openings by generating precise command keystrokes.

# INPUT DATA
You will receive a block of text containing the robot's recent action history and a list of detected openings (or a message that none were found).
- `Action History`: A Python list of the most recent command keys sent to the robot. If the list is empty, this will be shown as `Action History: None`.
- `angle`: The angle in degrees from the robot's front to the center of an opening.
- `distance`: The direct distance to the center of an opening in meters.
- `width`: The width of an opening in meters.

# YOUR TASK AND RULES
You must follow these rules in the exact order of priority given to generate a response for the New Input you receive.

**RULE 1: CRITICAL OSCILLATION CHECK (EXECUTE FIRST!) **
- **BEFORE** you look at the openings, you MUST check the `Action History` list.
- Does the exact, consecutive sequence of keys `[L, D, A]` appear **ANYWHERE** inside the `Action History` list?
- **IF YES:** The check is positive. You MUST STOP. Your ONLY response must be `{{"key": "Z", "value": 1.0}}`. Do not proceed to any other rule.
- **IF NO:** The check is negative. Then, and only then, may you proceed to Rule 2.

**RULE 2: NAVIGATE TO AN OPENING (If not stuck)**
- Find the opening with the smallest `distance`. This is your target.
- Based on the `angle` and `distance` of your target opening, generate a command:
    - If the target's `angle` is between -5 and +5 degrees (inclusive), respond by moving straight. The `key` MUST be "W" and the `value` MUST be the `distance` to that target opening.
    - If the target's `angle` is less than -5 degrees, respond with a clockwise turn. The `key` MUST be "L" and the `value` MUST be the positive value of the angle.
    - If the target's `angle` is greater than +5 degrees, respond with a counter-clockwise turn. The `key` MUST be "J" and the `value` MUST be the angle itself.

**RULE 3: HANDLE "NO VALID OPENINGS" (If not stuck)**
- If the input you receive is "No valid openings detected.", you must decide on an action based on the history:
    - If the last three actions in the `Action History` were [J, J, J], respond by moving forward slightly: `{{"key": "W", "value": 1.0}}`
    - Otherwise (for any other history), respond with a left turn: `{{"key": "J", "value": 45.0}}`

# OUTPUT FORMAT
Your response MUST be a single, valid JSON object and nothing else. Do not add any explanations.
- Example for turning clockwise: `{{"key": "L", "value": 15.6}}`
- Example for turning counter-clockwise: `{{"key": "J", "value": 15.0}}`
- Example for moving straight to an opening: `{{"key": "W", "value": 0.88}}`
- Example for getting unstuck: `{{"key": "Z", "value": 1.0}}`

# EXAMPLE OF A CASE
## Example Input:
Action History: [W, J, W, J, L, W]
Detected openings:
Opening 1: angle=15.0, distance=2.57, width=4.99
Opening 2: angle=-15.6, distance=0.88, width=1.35

## Reasoning for the Correct Output:
1.  My first and most important task is to perform the oscillation check (Rule 1).
2.  The history `[W, J, W, J, L, W]` does not contain `[L, D, A]`. Rule 1 does not apply.
3.  I proceed to Rule 2. I MUST find the opening with the **smallest distance**.
4.  Comparing distances [2.57, 0.88], the smallest is 0.88, which belongs to Opening 2. This is my target.
5.  The angle for the target is -15.6 degrees.
6.  According to the rules, since the angle is less than -5, I must respond with a clockwise turn ("L") and use the positive value of the angle.
7.  Therefore, the correct JSON output is `{{"key": "L", "value": 15.6}}`.

## Corresponding Correct Output:
`{{"key": "L", "value": 15.6}}`

This example is for illustration only, produce json based on the New Input.
"""

# --- [Stage 3] Command Generation Function ---
def generate_robot_command(physical_plan: dict) -> dict:
    try:
        command_key = physical_plan.get("key")
        value = float(physical_plan.get("value", 0))

        if command_key in ["L", "J"]:
            # This is a turning command
            time = value / TURNING_SPEED_DEG_PER_SEC
        elif command_key in ["W"]:
            # This is a moving command
            time = value / MOVING_SPEED_M_PER_SEC
        elif command_key in ["Z"]:
            # This is a stopping command
            time = value
        else:
            print(f"Warning: Unknown key '{command_key}'. Stopping.")
            return {"key": "Z", "time": 0.1}

        return {"key": command_key, "time": round(time, 2)}

    except (AttributeError, TypeError, ValueError) as e:
        print(f"Error processing physical plan: {physical_plan}. Error: {e}")
        return {"key": "Z", "time": 0.1}

def initialize_model():
    """Loads the LLM pipeline. Should be called once."""
    global CONTROLLER_LLM
    if CONTROLLER_LLM is None:
        print("Loading the model... This may take a few minutes.")
        access_token = "Your access token"
        CONTROLLER_LLM = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.3-70B-Instruct",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            token=access_token)
    print("Model loaded successfully.")

def get_robot_action(lidar_data: dict, action_history: list) -> dict:
    """
    The main pipeline.
    """
    if CONTROLLER_LLM is None:
        raise Exception("Model not initialized. Call initialize_model() first.")

    # --- Step 1: VISUALIZE ---
    print("\n--- [Step 1] Visualizing Environment... ---")
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"scan_{timestamp}.png"
        file_path = os.path.join(VISUALIZATION_DIR, filename)
        
        llm_input_text = perceive_and_visualize_openings(lidar_data, file_path)
        print(f"Summary for LLM:\n{llm_input_text}")

    except Exception as e:
        print(f"Error during visualization: {e}")
        traceback.print_exc()
        return {"key": "Z", "time": 0.1, "error": "Visualization failed."}
        
    # --- Step 2: PLAN (Decision-Making LLM) ---
    print("\n--- [Step 2] Requesting Plan from LLM... ---")

    # Format the action_history for the LLM
    history_str = "Action History: None"
    if action_history:
        # Extract the 'key' from each dictionary in the history
        history_keys = [str(action.get('key', '')) for action in action_history]
        history_str = f"Action History: [{', '.join(history_keys)}]"

    # Combine the lidar summary and the action history into one input
    full_llm_input = f"New Input\n{history_str}\n{llm_input_text}"
    print(f"Full input for LLM:\n{full_llm_input}")

    planner_messages = [
        {"role": "system", "content": MOVEMENT_PLANNER_PROMPT},
        {"role": "user", "content": full_llm_input}]
    
    # Force the model to start with the JSON structure
    forced_start_of_json = '{"key": "'
    
    planner_prompt_base = CONTROLLER_LLM.tokenizer.apply_chat_template(
        planner_messages, tokenize=False, add_generation_prompt=True)
    planner_prompt = (planner_prompt_base + forced_start_of_json)
    
    terminators = [
        CONTROLLER_LLM.tokenizer.eos_token_id,
        CONTROLLER_LLM.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    
    planner_outputs = CONTROLLER_LLM(
        planner_prompt, max_new_tokens=16, eos_token_id=terminators, do_sample=False, clean_up_tokenization_spaces=False)
    
    # Reconstruct the full JSON from the generated fragment
    generated_fragment = planner_outputs[0]["generated_text"][len(planner_prompt):]
    full_response_str = forced_start_of_json + generated_fragment

    print(f"LLM Physical Plan (Raw): {full_response_str}")

    try:
        # Clean up potential trailing characters after the final brace
        last_brace_index = full_response_str.rfind('}')
        if last_brace_index == -1:
            raise json.JSONDecodeError("No closing brace '}' found.", full_response_str, 0)
        json_part = full_response_str[:last_brace_index + 1]
        physical_plan = json.loads(json_part)
        print(f"LLM Physical Plan (Parsed): {physical_plan}")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from the Planner's response: {e}")
        return {"key": "Z", "time": 0.1}

    # --- Step 3: EXECUTE ---
    print("\n--- [Step 3] Generating Final Robot Command... ---")
    final_action = generate_robot_command(physical_plan)
    print(f"Final Command: {final_action}")
    
    return final_action
