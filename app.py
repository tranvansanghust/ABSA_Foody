from flask import Flask, render_template
import json
import bson

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/reload')
def reload():
    global to_reload
    to_reload = True
    return "reloaded"

    return app

@app.route('/get_details', methods=['POST'])
def get_details():
    pass


if __name__ == "__main__":
    app.run(debug=True)
