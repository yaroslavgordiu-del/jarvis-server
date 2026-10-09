import os
import re
import time
import base64
import logging
from collections import deque
from threading import Lock

import requests
from flask import Flask, request, jsonify, Response
from google import genai
from google.genai import types

# Jarvis server for ESP32-S3. Required Render variable: GEMINI_API_KEY
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("jarvis")

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing in Render Environment Variables")

client = genai.Client(api_key=API_KEY)
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts")
TTS_VOICE = os.environ.get("GEMINI_TTS_VOICE", "Charon")
# If a model is temporarily overloaded (503/UNAVAILABLE), try stable alternatives.
TEXT_MODEL_FALLBACKS = list(dict.fromkeys([
    MODEL, "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash"
]))
TTS_MODEL_FALLBACKS = list(dict.fromkeys([
    TTS_MODEL, "gemini-3.8-flash-lite-tts", "gemini-3.1-flash-tts-preview"
]))

# Short-term conversation memory; it resets when Render restarts the service.
history = deque(maxlen=12)
history_lock = Lock()

SYSTEM_PROMPT = """
Ти — Jarvis, розумний домашній голосовий помічник.
Відповідай українською, якщо користувач говорить українською.
Спілкуйся природно, доброзичливо та впевнено.
Відповідай саме на запитання, не повторюй його замість відповіді.
На прості запитання відповідай коротко, на складні — достатньо докладно.
Враховуй попередні репліки з історії розмови.
Для актуальних фактів, новин, законів, подій, цін, розкладів та іншої мінливої інформації
використовуй Google Search, якщо він доступний. Не вигадуй актуальних даних.
Якщо пошук не дав надійної відповіді, чесно скажи про це.
Відповідь призначена для озвучення через маленький динамік: не використовуй Markdown-таблиці,
URL, довгі списки чи технічні пояснення. Якщо запит нерозбірливий — попроси повторити.
"""

WEATHER_WORDS = ("погод", "температур", "дощ", "сніг", "вітер", "прогноз", "weather", "temperature", "rain", "snow", "wind")
CITY_ALIASES = {
    "кривому розі": "Кривий Ріг", "кривой рог": "Кривий Ріг", "кривий ріг": "Кривий Ріг",
    "києві": "Київ", "киеве": "Київ", "київ": "Київ", "киев": "Київ",
    "харкові": "Харків", "харькове": "Харків", "харків": "Харків",
    "одесі": "Одеса", "одессе": "Одеса", "одеса": "Одеса",
    "львові": "Львів", "львове": "Львів", "львів": "Львів",
    "дніпрі": "Дніпро", "днепре": "Дніпро", "дніпро": "Дніпро",
    "запоріжжі": "Запоріжжя", "запорожье": "Запоріжжя",
    "сумах": "Суми", "суми": "Суми", "ужгороді": "Ужгород", "ужгород": "Ужгород",
    "полтаві": "Полтава", "полтава": "Полтава", "вінниці": "Вінниця", "виннице": "Вінниця", "вінниця": "Вінниця",
    "варшаві": "Варшава", "варшава": "Варшава", "будапешті": "Будапешт", "будапешт": "Будапешт",
}
WEATHER_CODES = {
    0: "ясно", 1: "переважно ясно", 2: "мінлива хмарність", 3: "хмарно",
    45: "туман", 48: "туман із памороззю", 51: "легка мряка", 53: "мряка", 55: "сильна мряка",
    56: "крижана мряка", 57: "сильна крижана мряка", 61: "невеликий дощ", 63: "дощ", 65: "сильний дощ",
    66: "крижаний дощ", 67: "сильний крижаний дощ", 71: "невеликий сніг", 73: "сніг", 75: "сильний сніг",
    77: "снігові зерна", 80: "короткочасний дощ", 81: "зливи", 82: "сильні зливи",
    85: "снігові заряди", 86: "сильні снігові заряди", 95: "гроза", 96: "гроза з градом", 99: "сильна гроза з градом",
}


def _is_temporary_or_model_error(exc):
    message = str(exc).lower()
    return any(token in message for token in (
        "503", "unavailable", "high demand", "overloaded", "429", "resource_exhausted",
        "500", "internal error", "502", "504", "not found", "404", "model is not found"
    ))


def generate_content_with_fallback(*, contents, config=None):
    last_error = None
    for model_name in TEXT_MODEL_FALLBACKS:
        for attempt in range(2):
            try:
                if config is None:
                    return client.models.generate_content(model=model_name, contents=contents)
                return client.models.generate_content(model=model_name, contents=contents, config=config)
            except Exception as exc:
                last_error = exc
                if not _is_temporary_or_model_error(exc):
                    raise
                log.warning("Gemini model %s failed (attempt %s): %s", model_name, attempt + 1, exc)
                time.sleep(1.5 * (attempt + 1))
                break
    raise last_error


def interaction_with_fallback(*, input, tools=None, response_format=None, generation_config=None):
    last_error = None
    for model_name in (TTS_MODEL_FALLBACKS if response_format else TEXT_MODEL_FALLBACKS):
        for attempt in range(2):
            try:
                kwargs = {"model": model_name, "input": input}
                if tools is not None:
                    kwargs["tools"] = tools
                if response_format is not None:
                    kwargs["response_format"] = response_format
                if generation_config is not None:
                    kwargs["generation_config"] = generation_config
                result = client.interactions.create(**kwargs)
                log.info("Gemini interaction succeeded with model %s", model_name)
                return result
            except Exception as exc:
                last_error = exc
                if not _is_temporary_or_model_error(exc):
                    raise
                log.warning("Gemini interaction model %s failed (attempt %s): %s", model_name, attempt + 1, exc)
                time.sleep(1.5 * (attempt + 1))
                break
    raise last_error


def clean_text(text):
    text = (text or "").strip()
    text = re.sub(r"^\s*(Відповідь|Jarvis|Текст)\s*:\s*", "", text, flags=re.I)
    return text[:1800].strip()


def history_text():
    with history_lock:
        return "\n".join(f"{role}: {text}" for role, text in history) or "Історія поки порожня."


def save_turn(question, answer):
    with history_lock:
        history.append(("Користувач", question[:500]))
        history.append(("Jarvis", answer[:900]))


def transcribe_audio(audio_bytes):
    response = generate_content_with_fallback(
        contents=[
            "Точно розпізнай мовлення в аудіо. Поверни лише слова користувача, без відповіді, вступу, лапок чи пояснень. Збережи мову оригіналу. Якщо неможливо розібрати, поверни [НЕРОЗБІРЛИВО].",
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
        ],
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=180),
    )
    return clean_text(getattr(response, "text", ""))


def weather_city_from_text(text):
    low = text.lower()
    for alias in sorted(CITY_ALIASES, key=len, reverse=True):
        if alias in low:
            return CITY_ALIASES[alias]
    patterns = [
        r"(?:погод\w*|температур\w*|прогноз\w*|дощ\w*|сніг\w*)\s+(?:у|в|для|на)\s+([A-Za-zА-Яа-яІіЇїЄєҐґ'’ -]{2,45})",
        r"\b(?:у|в|для|на)\s+([A-Za-zА-Яа-яІіЇїЄєҐґ'’ -]{2,45})",
        r"\b(?:in|at|for)\s+([A-Za-z -]{2,45})",
    ]
    for pattern in patterns:
        match = re.search(pattern, low, flags=re.I)
        if match:
            candidate = match.group(1).strip(" .,!?:;")
            candidate = re.split(r"\b(сьогодні|завтра|зараз|на вихідних|today|tomorrow|now|буде)\b", candidate, maxsplit=1, flags=re.I)[0].strip()
            if len(candidate) >= 2:
                return candidate[:45]
    return "Кривий Ріг"


def get_weather(city):
    geo_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 5, "language": "uk", "format": "json"}, timeout=10,
    )
    geo_response.raise_for_status()
    places = geo_response.json().get("results") or []
    if not places:
        raise ValueError(f"Не знайдено місто: {city}")
    place = next((p for p in places if p.get("country_code") == "UA"), places[0])
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"], "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": 2, "timezone": place.get("timezone") or "auto",
        }, timeout=12,
    )
    response.raise_for_status()
    data = response.json()
    daily = data.get("daily", {})
    def day_value(key, idx):
        values = daily.get(key) or []
        return values[idx] if len(values) > idx else None
    return {
        "place": f"{place.get('name', city)}, {place.get('country', '')}",
        "current": data.get("current", {}),
        "today": {"min": day_value("temperature_2m_min", 0), "max": day_value("temperature_2m_max", 0), "rain": day_value("precipitation_probability_max", 0)},
        "tomorrow": {"code": day_value("weather_code", 1), "min": day_value("temperature_2m_min", 1), "max": day_value("temperature_2m_max", 1), "rain": day_value("precipitation_probability_max", 1)},
    }


def answer_with_weather(question, weather):
    cur, today, tomorrow = weather["current"], weather["today"], weather["tomorrow"]
    facts = (
        f"Місто: {weather['place']}. Зараз: {cur.get('temperature_2m')}°C, {WEATHER_CODES.get(cur.get('weather_code'), 'умови не визначені')}; "
        f"відчувається як {cur.get('apparent_temperature')}°C; вологість {cur.get('relative_humidity_2m')}%; "
        f"вітер {cur.get('wind_speed_10m')} км/год. Сьогодні: мінімум {today['min']}°C, максимум {today['max']}°C, "
        f"імовірність опадів до {today['rain']}%. Завтра: {WEATHER_CODES.get(tomorrow['code'], 'умови не визначені')}, "
        f"від {tomorrow['min']} до {tomorrow['max']}°C, імовірність опадів до {tomorrow['rain']}%."
    )
    response = generate_content_with_fallback(
        contents=SYSTEM_PROMPT + "\nАктуальні дані погоди із сервісу (не змінюй числа й не вигадуй деталей):\n" + facts + "\nІсторія:\n" + history_text() + "\nЗапитання: " + question + "\nДай коротку природну відповідь українською.",
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=160),
    )
    return clean_text(getattr(response, "text", ""))


def answer_question(question):
    prompt = (
        SYSTEM_PROMPT
        + "\nІсторія попередньої розмови:\n" + history_text()
        + "\nНове запитання користувача:\n" + question
        + "\nВідповідай саме на нове запитання, не повторюй його. "
          "Для актуальних фактів, новин, цін або подій використовуй Google Search."
    )
    # Use the stable GenerateContent endpoint for text + Google Search grounding.
    # The previous Interactions text call was returning HTTP 429 and delaying the ESP32.
    search_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(
        temperature=0.35,
        max_output_tokens=220,
        tools=[search_tool],
    )
    response = generate_content_with_fallback(contents=prompt, config=config)
    return clean_text(getattr(response, "text", ""))


def synthesize_speech(text):
    interaction = interaction_with_fallback(
        input=[{"type": "user_input", "content": [{
            "type": "text", "text": text,
            "annotations": [{"type": "speech_metadata", "style": "natural, clear, friendly Ukrainian speech at a moderate pace"}],
        }]}],
        response_format={"type": "audio"},
        generation_config={"speech_config": [{"voice": TTS_VOICE}]},
    )
    audio = getattr(getattr(interaction, "output_audio", None), "data", None)
    if not audio:
        raise RuntimeError("Gemini TTS returned no audio")
    wav = base64.b64decode(audio)
    if not wav.startswith(b"RIFF") or wav[8:12] != b"WAVE":
        raise RuntimeError("Gemini TTS did not return WAV audio")
    return wav


@app.get("/")
def index():
    return "Jarvis server is running.", 200


@app.get("/health")
def health():
    return jsonify({"ok": True, "model": MODEL, "tts_model": TTS_MODEL, "search": "Google Search grounding", "weather": "Open-Meteo"}), 200


@app.post("/voice")
def voice():
    started = time.time()
    try:
        audio = request.get_data(cache=False)
        if not audio or len(audio) <= 44:
            return jsonify({"ok": False, "error": "Empty or invalid WAV audio"}), 400
        if not audio.startswith(b"RIFF") or audio[8:12] != b"WAVE":
            return jsonify({"ok": False, "error": "Expected WAV audio"}), 400
        log.info("Received WAV audio: %d bytes", len(audio))
        question = transcribe_audio(audio)
        log.info("Recognized speech: %s", question[:300])

        if not question or "[НЕРОЗБІРЛИВО]" in question.upper():
            answer = "Я не зовсім розібрав запитання. Будь ласка, повтори його трохи чіткіше."
            save_turn("[мовлення не розібрано]", answer)
        else:
            try:
                if any(word in question.lower() for word in WEATHER_WORDS):
                    try:
                        weather = get_weather(weather_city_from_text(question))
                        answer = answer_with_weather(question, weather)
                    except Exception as weather_error:
                        log.warning("Weather lookup failed: %s", weather_error)
                        answer = answer_question(question + "\nЗнайди актуальні дані погоди через Google Search. Не вигадуй їх, якщо не можеш перевірити.")
                else:
                    answer = answer_question(question)
            except Exception:
                log.exception("Gemini answer generation failed")
                answer = "Вибач, зараз не вдалося отримати відповідь від Gemini. Спробуй ще раз трохи пізніше."
            if not answer:
                answer = "Я не зміг сформувати відповідь. Спробуй, будь ласка, запитати інакше."
            save_turn(question, answer)

        log.info("Answer prepared in %.1f sec", time.time() - started)
        try:
            wav = synthesize_speech(answer)
        except Exception:
            log.exception("Gemini speech generation failed")
            wav = synthesize_speech("Вибач, зараз у мене проблема з голосовою відповіддю. Спробуй ще раз.")
        return Response(wav, status=200, mimetype="audio/wav", headers={"Cache-Control": "no-store"})
    except Exception:
        log.exception("Voice endpoint failed")
        return jsonify({"ok": False, "error": "Voice processing failed"}), 500


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"ok": False, "error": "Audio is too large"}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))
