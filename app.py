import os
import tempfile
import base64
import gc

from flask import Flask, request, jsonify, Response
from google import genai

app = Flask(__name__)

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)

MODEL = "gemini-3.7-flash"
TTS_MODEL = "gemini-3.8-flash-tts"


@app.route("/")
def home():
    return "JARVIS SERVER OK"


@app.route("/voice", methods=["POST"])
def voice():

    audio_path = None

    try:
        # --------------------------------
        # RECEIVE WAV
        # --------------------------------

        audio = request.get_data(cache=False)

        size = len(audio)

        print("Received audio:", size, "bytes", flush=True)

        if not audio:
            return jsonify({
                "ok": False,
                "error": "No audio received"
            }), 400

        # --------------------------------
        # SAVE TEMP FILE
        # --------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as f:

            audio_path = f.name
            f.write(audio)

        # звільняємо великий об'єкт
        del audio
        gc.collect()

        print("WAV SAVED:", audio_path, flush=True)

        # --------------------------------
        # UPLOAD WAV TO GEMINI
        # --------------------------------

        uploaded_file = client.files.upload(
            file=audio_path
        )

        print("AUDIO UPLOADED TO GEMINI", flush=True)

        # --------------------------------
        # VOICE -> TEXT/ANSWER
        # --------------------------------

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

        answer = (response.text or "").strip()

        print("GEMINI ANSWER:", answer, flush=True)

        # звільняємо об'єкти Gemini перед TTS
        del response
        del uploaded_file
        gc.collect()

        if not answer:
            return jsonify({
                "ok": False,
                "error": "Empty Gemini answer"
            }), 500

        # --------------------------------
        # TEXT -> VOICE
        # --------------------------------

        print("START TTS", flush=True)

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

        print("TTS GENERATED", flush=True)

        audio_data = tts.output_audio.data

        audio_out = base64.b64decode(audio_data)

        print(
            "TTS AUDIO:",
            len(audio_out),
            "bytes",
            flush=True
        )

        # --------------------------------
        # RETURN AUDIO TO ESP32
        # --------------------------------

        return Response(
            audio_out,
            mimetype="audio/wav"
        )

    except Exception as e:

        print(
            "ERROR:",
            repr(e),
            flush=True
        )

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500

    finally:

        # --------------------------------
        # DELETE TEMP WAV
        # --------------------------------

        if audio_path:

            try:
                os.remove(audio_path)
                print(
                    "TEMP WAV DELETED",
                    flush=True
                )
            except Exception:
                pass

        gc.collect()


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=10000
    )
