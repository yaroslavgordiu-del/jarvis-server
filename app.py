import os
import time
import tempfile
import base64
import gc

from flask import Flask, request, jsonify, Response
from google import genai

app = Flask(__name__)

# =====================================================
# GEMINI CLIENT
# =====================================================

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

client = genai.Client(api_key=API_KEY)

# Primary and fallback models for understanding speech.
TEXT_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
]

# Primary and fallback models for generating speech.
TTS_MODELS = [
    "gemini-3.8-flash-tts",
    "gemini-3.8-flash-lite-tts",
]

app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024


# =====================================================
# RETRY HELPERS
# =====================================================

def is_temporary_error(error):
    """
    Detect temporary capacity errors and rate limits.
    """

    message = str(error).upper()

    temporary_markers = [
        "503",
        "UNAVAILABLE",
        "HIGH DEMAND",
        "429",
        "RESOURCE_EXHAUSTED",
        "TOO MANY REQUESTS",
        "500",
        "INTERNAL SERVER ERROR",
        "502",
        "BAD GATEWAY",
        "504",
        "DEADLINE EXCEEDED",
        "TIMED OUT",
        "TIMEOUT",
    ]

    return any(
        marker in message
        for marker in temporary_markers
    )


def call_with_fallback(model_names, operation_name, operation):
    """
    Try each model in order.
    Retry temporary errors with short pauses.
    """

    last_error = None

    for model_index, model_name in enumerate(model_names):

        # Two attempts per model, with a short pause.
        for attempt in range(2):

            try:
                print(
                    f"{operation_name}: "
                    f"model={model_name}, "
                    f"attempt={attempt + 1}",
                    flush=True
                )

                result = operation(model_name)

                print(
                    f"{operation_name} SUCCESS: {model_name}",
                    flush=True
                )

                return result

            except Exception as error:

                last_error = error

                print(
                    f"{operation_name} ERROR: "
                    f"model={model_name}, "
                    f"attempt={attempt + 1}, "
                    f"error={repr(error)}",
                    flush=True
                )

                # Do not retry permanent errors such as
                # invalid model names or invalid API keys.
                if not is_temporary_error(error):
                    raise

                # Wait before retrying the same model.
                if attempt == 0:
                    delay = 2

                    print(
                        f"Temporary error. "
                        f"Retrying in {delay} seconds.",
                        flush=True
                    )

                    time.sleep(delay)

        # Move to the next model after both attempts fail.
        if model_index < len(model_names) - 1:

            print(
                f"{model_name} unavailable. "
                f"Switching to fallback model.",
                flush=True
            )

            gc.collect()

    raise RuntimeError(
        f"All {operation_name} models failed. "
        f"Last error: {repr(last_error)}"
    )


# =====================================================
# HOME
# =====================================================

@app.route("/")
def home():
    return "JARVIS SERVER OK"


# =====================================================
# VOICE ENDPOINT
# =====================================================

@app.route("/voice", methods=["POST"])
def voice():

    audio_path = None
    uploaded_file = None

    try:

        # ---------------------------------------------
        # 1. RECEIVE AUDIO FROM ESP32
        # ---------------------------------------------

        audio = request.get_data(cache=False)

        audio_size = len(audio)

        print(
            "RECEIVED AUDIO:",
            audio_size,
            "bytes",
            flush=True
        )

        if not audio:
            return jsonify({
                "ok": False,
                "stage": "receive",
                "error": "No audio received"
            }), 400

        if audio_size < 44:
            return jsonify({
                "ok": False,
                "stage": "receive",
                "error": "Audio file is too small"
            }), 400

        # ---------------------------------------------
        # 2. SAVE TEMPORARY WAV
        # ---------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as file:

            audio_path = file.name
            file.write(audio)

        del audio
        gc.collect()

        print("WAV SAVED", flush=True)

        # ---------------------------------------------
        # 3. UPLOAD AUDIO TO GEMINI
        # ---------------------------------------------

        print("UPLOADING AUDIO TO GEMINI", flush=True)

        uploaded_file = client.files.upload(
            file=audio_path
        )

        print(
            "AUDIO UPLOADED TO GEMINI",
            flush=True
        )

        # ---------------------------------------------
        # 4. UNDERSTAND THE USER'S VOICE
        # ---------------------------------------------

        prompt = """
Ти — Джарвіс, домашній голосовий AI-помічник.

Користувач говорить українською мовою.

Зрозумій його голосовий запит і дай коротку,
корисну відповідь українською мовою.

Відповідь повинна бути природною та короткою,
оскільки її буде озвучено через динамік.

Не описуй аудіофайл.
Не говори про аналіз файлу.
Не пояснюй внутрішні технічні процеси.

Якщо користувача неможливо зрозуміти, відповідай:
«Не почув. Повтори, будь ласка».
"""

        def generate_answer(model_name):

            return client.models.generate_content(
                model=model_name,
                contents=[
                    uploaded_file,
                    prompt
                ]
            )

        response = call_with_fallback(
            TEXT_MODELS,
            "VOICE UNDERSTANDING",
            generate_answer
        )

        answer = (response.text or "").strip()

        del response
        gc.collect()

        print(
            "GEMINI ANSWER:",
            answer,
            flush=True
        )

        if not answer:
            return jsonify({
                "ok": False,
                "stage": "understanding",
                "error": "Gemini returned an empty answer"
            }), 502

        # ---------------------------------------------
        # 5. CONVERT THE ANSWER TO SPEECH
        # ---------------------------------------------

        print("STARTING TTS", flush=True)

        def generate_speech(model_name):

            return client.interactions.create(
                model=model_name,
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

        tts = call_with_fallback(
            TTS_MODELS,
            "TEXT TO SPEECH",
            generate_speech
        )

        print("TTS GENERATED", flush=True)

        # ---------------------------------------------
        # 6. EXTRACT THE AUDIO
        # ---------------------------------------------

        encoded_audio = tts.output_audio.data

        if not encoded_audio:
            return jsonify({
                "ok": False,
                "stage": "tts",
                "error": "TTS returned no audio data"
            }), 502

        audio_out = base64.b64decode(
            encoded_audio
        )

        del tts
        del encoded_audio
        gc.collect()

        if len(audio_out) < 44:
            return jsonify({
                "ok": False,
                "stage": "tts",
                "error": "Generated audio is too small"
            }), 502

        print(
            "TTS AUDIO SIZE:",
            len(audio_out),
            "bytes",
            flush=True
        )

        # ---------------------------------------------
        # 7. RETURN AUDIO TO ESP32
        # ---------------------------------------------

        print(
            "RETURNING AUDIO TO ESP32",
            flush=True
        )

        return Response(
            audio_out,
            status=200,
            mimetype="audio/wav",
            headers={
                "Cache-Control": "no-store",
                "X-Jarvis-Status": "ok"
            }
        )

    except Exception as error:

        print(
            "JARVIS ERROR:",
            repr(error),
            flush=True
        )

        return jsonify({
            "ok": False,
            "error": str(error)
        }), 500

    finally:

        # ---------------------------------------------
        # 8. CLEANUP
        # ---------------------------------------------

        if audio_path:

            try:
                os.remove(audio_path)

                print(
                    "TEMP WAV DELETED",
                    flush=True
                )

            except Exception as error:

                print(
                    "TEMP FILE CLEANUP ERROR:",
                    repr(error),
                    flush=True
                )

        gc.collect()


# =====================================================
# START SERVER
# =====================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=10000
    )
