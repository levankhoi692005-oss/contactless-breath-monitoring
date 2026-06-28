from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__)

DATA_FILE = "output/latest.json"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/data")
def data():

    if not os.path.exists(DATA_FILE):

        return jsonify({
            "bpm": 0,
            "confidence": 0,
            "fps": 0
        })

    with open(DATA_FILE, "r") as f:
        return jsonify(json.load(f))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)