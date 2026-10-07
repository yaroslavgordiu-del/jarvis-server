from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "JARVIS SERVER OK"

@app.route("/voice", methods=["POST"])
def voice():
    audio = request.data

    print("Received audio:", len(audio), "bytes")

    return jsonify({
        "ok": True,
        "message": "VOICE RECEIVED",
        "size": len(audio)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
