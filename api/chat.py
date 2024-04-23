from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from flask import jsonify
from flask_cors import CORS
from urllib.parse import unquote
from data import upload
from llm import ask


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/")
@cross_origin()
def home():
    return "Flask Vercel Example - Hello World", 200

# Route to get a string as a response after taking input from the user
@app.route('/<message>', methods=['GET'])
@cross_origin()
def get_book(message: str):
    message = unquote(message)
    response = ask(message)
    return jsonify({"message": response})

@app.route('/upload/<data>', methods=['POST'])
@cross_origin()
def upload_data(data: str):
    data = unquote(data)
    result = upload(data)
    return jsonify({"message": "Data uploaded successfully", "result": result})

if __name__ == '__main__':
    app.run()
