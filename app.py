"""
app.py
--------------------------------
Flask Dashboard
"""

from flask import Flask
from flask import render_template
from flask import jsonify

import json
import os

app = Flask(__name__)

DATA_FILE = "output/latest.json"


# ===========================================

def load_data():

    if not os.path.exists(DATA_FILE):

        return {

            "bpm": 0,

            "confidence": 0,

            "fps": 0

        }

    try:

        with open(DATA_FILE, "r") as f:

            return json.load(f)

    except:

        return {

            "bpm": 0,

            "confidence": 0,

            "fps": 0

        }


# ===========================================

@app.route("/")

def home():

    return render_template("index.html")


# ===========================================

@app.route("/data")

def data():

    return jsonify(

        load_data()

    )


# ===========================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=False

    )