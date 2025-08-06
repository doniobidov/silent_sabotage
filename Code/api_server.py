# api_server.py

from flask import Flask, request, jsonify
from llm_controller import initialize_model, get_robot_action

# Initialize the Flask App
app = Flask(__name__)

# Load the LLM Model once when the server starts
print("Initializing LLM Controller...")
initialize_model()
print("Model ready. Starting Flask server...")

# Define the API endpoint that will receive the data
@app.route('/get_action', methods=['POST'])
def handle_get_action():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    lidar_data = data.get('lidar_data')
    action_history = data.get('action_history')

    if lidar_data is None or action_history is None:
        return jsonify({"error": "Missing 'lidar_data' or 'action_history' in request"}), 400

    try:
        # Get the robot's action
        action_plan = get_robot_action(lidar_data, action_history)
        
        if action_plan:
            return jsonify(action_plan) # Send the result back to the client
        else:
            return jsonify({"error": "LLM failed to generate a valid action"}), 500

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500

# Run the Server
if __name__ == '__main__':
    # Use host='0.0.0.0' to make the server accessible from other computers on the network
    app.run(host='0.0.0.0', port=5000, debug=False)
