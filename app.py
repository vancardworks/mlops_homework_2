from flask import Flask, request, jsonify, render_template
from analyze import get_sentiment, compute_embeddings, classify_email, load_class_file, load_classes, save_classes
app = Flask(__name__, template_folder='templates')


@app.route("/")
def home():
    print("Home page")
    return render_template('index.html')


@app.route("/api/v1/sentiment-analysis/", methods=['POST'])
def analysis():
    if request.is_json:
        data = request.get_json()
        sentiment = get_sentiment(data['text'])
        return jsonify({"message": "Data received", "data": data, "sentiment": sentiment}), 200
    else:
        return jsonify({"error": "Invalid Content-Type"}), 400


@app.route("/api/v1/valid-embeddings/", methods=['GET'])
def valid_embeddings():
    embeddings = compute_embeddings()
    formatted_embeddings = []
    for text, vector in embeddings:
        formatted_embeddings.append({
            "text": text,
            "vector": vector.tolist() if hasattr(vector, 'tolist') else vector
        })
    embeddings = formatted_embeddings
    return jsonify({"message": "Valid embeddings fetched", "embeddings": embeddings}), 200


@app.route("/api/v1/classify/", methods=['POST'])
def classify():
    if request.is_json:
        data = request.get_json()
        text = data['text']
        classifications = classify_email(text)
        return jsonify({"message": "Email classified", "classifications": classifications}), 200
    else:
        return jsonify({"error": "Invalid Content-Type"}), 400


@app.route("/api/v1/classify-email/", methods=['GET'])
def classify_with_get():
    load_class_file()
    text = request.args.get('text')
    classifications = classify_email(text)
    return jsonify({"message": "Email classified", "classifications": classifications}), 200

@app.route("/api/v1/add-classes/", methods=['POST'])
def add_class():
    data = request.get_json()
    new_class = data.get("class")

    if not new_class:
        return jsonify({"error": "No class provided"}), 400

    classes = load_classes()
    if new_class in classes:
        return jsonify({"message": "Class already exists"}), 400

    classes.append(new_class)
    save_classes(classes)

    return jsonify({"message": "Class added successfully", "classes": classes})
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=True)