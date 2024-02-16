from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/")
def home():
    return "Flask Vercel Example - Hello World", 200

# Route to get a string as a response after taking input from the user
@app.route('/<message>', methods=['GET'])
def get_book(message: str):
    return "Welcome"

if __name__ == '__main__':
    app.run()
