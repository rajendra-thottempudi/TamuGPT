from flask import Flask, jsonify, request

app = Flask(__name__)

# Basic flask app to chat with a user
# Route to get a string as a response after taking input from the user
@app.route('/<message>', methods=['GET'])
def get_book(message: str):
    return "Welcome"

if __name__ == '__main__':
    app.run()
