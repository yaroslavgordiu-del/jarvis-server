import os
import tempfile
import base64

from flask import Flask, request, jsonify, Response
from google import genai

app = Flask(__name__)

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)

MODEL = "gemini-3.8-flash"
TTS_MODEL = "gemini-3.8-flash-lite-tts"


@app.route("/")
def home():
    return "JARVIS SERVER OK"


@app.route("/voice", methods=["POST"])
def voice():

    try:
        audio = request.data

        print("Received audio:", len(audio), "bytes")

        if not audio:
            return jsonify({
                "ok": False,
                "error": "No audio received"
            }), 400

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as f:
            f.write(audio)
            audio_path = f.name

        try:

            # -------------------------
            # VOICE -> GEMINI
            # -------------------------

            uploaded_file = client.files.upload(
                file=audio_path
            )

            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    uploaded_file,
                    """
Ти — Джарвіс, домашній голосовий AI-помічник.

Користувач говорить українською.

Зрозумій його запит і дай коротку корисну відповідь українською.

Відповідь повинна бути короткою,
бо її буде озвучено голосом.

Не описуй аудіо.
Не говори про те, що ти аналізуєш файл.

Якщо нічого не почув:
Не почув. Повтори, будь ласка.
"""
                ]
            )

            answer = response.text.strip()

            print("GEMINI:", answer)

            # -------------------------
            # TEXT -> VOICE
            # -------------------------

            tts = client.interactions.create(
                model=TTS_MODEL,
                input=answer,
                response_format={
                    "type": "audio"
                },
                generation_config={
                    "speech_config": [
                        {
                            "voice": "Kore"
                        }
                    ]
                }
            )

            audio_out = base64.b64decode(
                tts.output_audio.data
            )

            print(
                "TTS AUDIO:",
                len(audio_out),
                "bytes"
            )

            # Повертаємо WAV назад ESP32
            return Response(
                audio_out,
                mimetype="audio/wav"
            )

        finally:

            try:
                os.remove(audio_path)
            except:
                pass

    except Exception as e:

        print("ERROR:", repr(e))

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=10000
    )
