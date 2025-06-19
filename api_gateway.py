from flask import Flask, request, jsonify
import requests, time, json
from collections import OrderedDict


app = Flask(__name__)

# URL for the TorchServe instance (assumed to be running on port 8080)
TORCHSERVE_URL = "http://127.0.0.1:8080"

# Global variables for metrics and configuration for the API layer
API_START_TIME = time.time()
total_requests = 0
latencies = []
max_latency = 0

# Model and group configuration
available_models = ["model_0", "model_1", "model_2"]
default_model = "model_0"
group_info = {"group": "group1", "members": ["Anna Kudiakova", "Salar Jalali", "Victor Sung", "Jieming Yu"]}
# Per‑model confidence thresholds
model_conf_thresholds = {
    "model_0": 0.60,
    "model_1": 0.40,
    "model_2": 0.50,  # fill in as needed
}

def update_metrics(latency):
    global total_requests, latencies, max_latency
    total_requests += 1
    latencies.append(latency)
    if latency > max_latency:
        max_latency = latency

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    # Expecting a multipart form-data with 'image' and an optional 'model'
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    image_file = request.files['image']
    # Get model parameter from form data; use default if not provided or invalid.
    model = request.form.get('model', default_model)
    if model not in available_models:
        model = default_model

    # Forward the image to TorchServe's inference endpoint at /predictions/{model}
    files = {'data': (image_file.filename, image_file, image_file.content_type)}
    #print(files)
    try:
        ts_response = requests.post(f"{TORCHSERVE_URL}/predictions/{model}", files=files)
        print("Status code:", ts_response.status_code)
        print("Response text:", ts_response.text)
        ts_response.raise_for_status()
    except Exception as e:
        return jsonify({"error": str(e)}), 500


    # Assume TorchServe returns a JSON response with a "predictions" key
    inference_data = ts_response.json()
    latency_ms = round((time.time() - start_time) * 1000, 2)
    update_metrics(latency_ms)

    # Reorder each prediction's keys to match the required format
    raw_predictions = inference_data.get("predictions", [])
    ordered_predictions = []
    for pred in raw_predictions:
        # Create an OrderedDict for each prediction with keys in the required order
        new_pred = OrderedDict([
            ("label", pred.get("label")),
            ("confidence", pred.get("confidence")),
            ("bbox", pred.get("bbox"))
        ])
        ordered_predictions.append(new_pred)

    # Construct final response in required key order
    response = OrderedDict([
        ("predictions", ordered_predictions),
        ("model_used", model)
    ])

    # Return the JSON response
    return app.response_class(
        response=json.dumps(response),
        status=200,
        mimetype='application/json'
    )

@app.route('/health-status', methods=['GET'])
def health_status():
    # Optionally, check TorchServe's health via its /ping endpoint.
    try:
        ts_response = requests.get(f"{TORCHSERVE_URL}/ping")
        ts_status = "Healthy" if ts_response.status_code == 200 else "Unhealthy"
    except Exception:
        ts_status = "Unhealthy"
    uptime_seconds = time.time() - API_START_TIME
    days = int(uptime_seconds // (24 * 3600))
    hours = int((uptime_seconds % (24 * 3600)) // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    uptime_str = f"{days} days, {hours} hours, {minutes} minutes"
    return jsonify({
        "status": ts_status,
        "server": "Flask",
        "uptime": uptime_str
    })

@app.route('/management/models', methods=['GET'])
def list_models():
    return jsonify({"available_models": available_models})

@app.route('/group-info', methods=['GET'])
def group_info_endpoint():
    return jsonify(group_info)

@app.route('/metrics', methods=['GET'])
def metrics():
    avg_latency = round(sum(latencies)/len(latencies), 2) if latencies else 0
    minutes_since_start = (time.time() - API_START_TIME) / 60
    request_rate = round(total_requests / minutes_since_start, 2) if minutes_since_start else total_requests
    return jsonify({
        "request_rate_per_minute": request_rate,
        "avg_latency_ms": avg_latency,
        "max_latency_ms": round(max_latency, 2),
        "total_requests": total_requests
    })

@app.route('/management/models/<model>/describe', methods=['GET'])
def describe_model(model):
    if model not in available_models:
        return jsonify({"error": "Model not found"}), 404
    model_config = {
        "input_size": [640, 640],
        "batch_size": 16,
        "confidence_threshold": 0.6
    }
    return jsonify({
        "model": model,
        "config": model_config,
        "date_registered": "2025-03-15"
    })

@app.route('/management/models/<model>/set-default', methods=['GET'])
def set_default_model(model):
    global default_model
    if model not in available_models:
        return jsonify({"error": "Model not found"}), 404
    default_model = model
    return jsonify({
        "success": True,
        "default_model": model
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6001, debug=True)
#    app.run(host='127.0.0.1', port=5000, debug=True) #For running locally
