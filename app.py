import os
import time
import base64
import logging
from collections import deque

from flask import Flask, request, jsonify, Response
from google import genai
from google.genai import types

# =====================================================
# JARVIS SERVER
# =====================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("jarvis")

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

client = genai.Client(api_key=API_KEY)

# Fast model first, with fallbacks.
TEXT_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
]

TTS_MODELS = [
    "gemini-3.8-flash-lite-tts",
    "gemini-3.8-flash-tts",
]

# Short conversation memory for this single Jarvis device.
history = deque(maxlen=8)

SYSTEM_PROMPT = """
Ти — Jarvis, розумний домашній голосовий помічник.

Правила спілкування:
- Відповідай українською, якщо користувач говорить українською.
- Спілкуйся природно, дружньо та впевнено.
- Не повторюй запитання користувача без потреби.
- На прості запитання відповідай коротко.
- На складні запитання пояснюй докладніше.
- Враховуй попередні репліки, якщо вони стосуються поточного питання.
- Не вигадуй актуальну погоду, новини, ціни або інші дані.
- Якщо не знаєш відповіді, чесно скажи про це.
- Не розповідай про аналіз аудіофайлів і внутрішню технічну роботу.
- Формулюй відповідь так, щоб її було приємно слухати через динамік.
"""

# =====================================================
# RETRIES
# =====================================================

def is_temporary_error(error):
    message = str(error).upper()

    markers = [
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

    return any(marker in message for marker in markers)


def call_with_fallback(models, operation_name, operation):
    last_error = None

    for index, model in enumerate(models):
        try:
            log.info("%s: trying %s", operation_name, model)

            result = operation(model)

            log.info("%s: success with %s", operation_name, model)
            return result

        except Exception as error:
            last_error = error

            log.exception(
                "%s failed with model %s",
                operation_name,
                model
            )

            # Do not waste time retrying invalid credentials
            # or unsupported model names.
            if not is_temporary_error(error):
                if index == len(models) - 1:
                    raise

                # Try the next model. Some errors are model-specific.
                continue

            # Only one short retry across the fallback sequence.
            # Avoid repeating a long wait for every model.
            if index == 0:
                time.sleep(0.5)

    raise RuntimeError(
        f"{operation_name} failed. Last error: {last_error}"
    )


# =====================================================
# ROUTES
# =====================================================

@app.route("/")
def home():
    return "JARVIS SERVER OK"


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "jarvis",
    })


# =====================================================
# VOICE
# =====================================================

@app.route("/voice", methods=["POST"])
def voice():
    started = time.monotonic()
    stage = "receive"

    try:
        # ---------------------------------------------
        # 1. RECEIVE WAV FROM ESP32
        # ---------------------------------------------

        audio = request.get_data(cache=False)

        log.info("Received audio: %s bytes", len(audio))

        if len(audio) < 44:
            return jsonify({
                "ok": False,
                "stage": stage,
                "error": "Audio is missing or too small",
            }), 400

        # ---------------------------------------------
        # 2. PREPARE AUDIO INLINE
        # ---------------------------------------------

        stage = "understanding"

        audio_part = types.Part.from_bytes(
            data=audio,
            mime_type="audio/wav",
        )

        del audio

        previous = "\n".join(
            f"{role}: {text}"
            for role, text in history
        )

        prompt = f"""
{SYSTEM_PROMPT}

Попередня розмова:
{previous if previous else "(це початок розмови)"}

Прослухай прикріплений аудіозапис.
Визнач, що саме сказав користувач, і дай відповідь
на його запитання або виконай словесну інструкцію.
Якщо слова незрозумілі, коротко попроси повторити.
"""

        # ---------------------------------------------
        # 3. GENERATE TEXT ANSWER
        # ---------------------------------------------

        def generate_answer(model):
            return client.models.generate_content(
                model=model,
                contents=[audio_part, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    max_output_tokens=180,
                ),
            )

        response = call_with_fallback(
            TEXT_MODELS,
            "VOICE UNDERSTANDING",
            generate_answer,
        )

        answer = (response.text or "").strip()

        if not answer:
            raise RuntimeError("Gemini returned an empty answer")

        log.info("Answer generated in %.2f seconds",
                 time.monotonic() - started)

        # Store the last exchange for follow-up questions.
        history.append(("Користувач", "[голосове запитання]"))
        history.append(("Jarvis", answer))

        # ---------------------------------------------
        # 4. GENERATE MALE VOICE
        # ---------------------------------------------

        stage = "tts"

        def generate_speech(model):
            return client.interactions.create(
                model=model,
                input=answer,
                response_format={"type": "audio"},
                generation_config={
                    "speech_config": [
                        {"voice": "Charon"}
                    ]
                },
            )

        tts = call_with_fallback(
            TTS_MODELS,
            "TEXT TO SPEECH",
            generate_speech,
        )

        encoded_audio = tts.output_audio.data

        if not encoded_audio:
            raise RuntimeError("TTS returned no audio")

        # Gemini Interactions API returns base64 audio data.
        audio_out = base64.b64decode(encoded_audio)

        if len(audio_out) < 44:
            raise RuntimeError("Generated audio is too small")

        log.info(
            "Returning %s bytes; total time %.2f seconds",
            len(audio_out),
            time.monotonic() - started,
        )

        # ---------------------------------------------
        # 5. RETURN AUDIO TO ESP32
        # ---------------------------------------------

        return Response(
            audio_out,
            status=200,
            mimetype="audio/wav",
            headers={
                "Cache-Control": "no-store",
                "X-Jarvis-Status": "ok",
            },
        )

    except Exception as error:
        log.exception(
            "JARVIS ERROR at stage %s",
            stage,
        )

        return jsonify({
            "ok": False,
            "stage": stage,
            "error": str(error),
        }), 500


# =====================================================
# START
# =====================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "10000")),
    )    port=10000
    )
