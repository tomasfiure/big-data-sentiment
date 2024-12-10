from flask import Flask, request, jsonify
from google.cloud import aiplatform

app = Flask(__name__)

# Initialize the Vertex AI SDK
aiplatform.init(project="your-project-id", location="your-region")

# Scorer function
def scorer(prompts):
    model = aiplatform.Model(model_name="your-model-name")
    predictions = model.predict(instances=prompts)

    sentiment_scores = []
    for prediction in predictions:
        # Example: Adjust based on your model's output schema
        positive_prob = prediction["positive"]
        neutral_prob = prediction["neutral"]
        negative_prob = prediction["negative"]

        sentiment_score = (positive_prob - negative_prob) / (
            positive_prob + neutral_prob + negative_prob
        )
        sentiment_scores.append(sentiment_score)
    return sentiment_scores

# API endpoint
@app.route("/score", methods=["POST"])
def score():
    data = request.json
    prompts = data.get("prompts", [])
    if not prompts:
        return jsonify({"error": "No prompts provided"}), 400

    try:
        scores = scorer(prompts)
        return jsonify({"scores": scores}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
