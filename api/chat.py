from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from flask import jsonify

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
@cross_origin()
def home():
    return "Flask Vercel Example - Hello World", 200

# Route to get a string as a response after taking input from the user
@app.route('/<message>', methods=['GET'])
@cross_origin()
def get_book(message: str):
    @app.route("/")
    @cross_origin()
    def home():
        return jsonify({"message": "Welcome"})

if __name__ == '__main__':
    app.run()
