import os
import tempfile

from flask import Flask, request, jsonify
from google import genai

app = Flask(__name__)

# Gemini
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

MODEL = "gemini-3.8-flash"


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

        # Тимчасово зберігаємо WAV
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as f:
            f.write(audio)
            audio_path = f.name

        try:
            # Завантажуємо аудіо в Gemini
            uploaded_file = client.files.upload(
                file=audio_path
            )

            # Gemini одночасно слухає голос
            # і формує відповідь українською.
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    uploaded_file,
                    """
Ти — Джарвіс, домашній голосовий AI-помічник.

Користувач говорить українською мовою.
Спочатку точно зрозумій, що він сказав.
Потім дай коротку, корисну відповідь українською.

Не описуй аудіо.
Не пояснюй, що ти аналізуєш аудіофайл.
Відповідай без зайвих слів.

Якщо голос нерозбірливий, напиши:
"Не почув. Повтори, будь ласка."
"""
                ]
            )

            answer = response.text.strip()

            print("GEMINI:", answer)

            return jsonify({
                "ok": True,
                "message": "VOICE PROCESSED",
                "answer": answer,
                "size": len(audio)
            })

        finally:
            # Видаляємо тимчасовий WAV
            try:
                os.remove(audio_path)
            except Exception:
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
