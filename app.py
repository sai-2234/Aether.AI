import os
import json
import requests
from datetime import datetime, timezone, timedelta
import zoneinfo
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins="*")

SERVER_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

INVOKE_URL    = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "mistralai/mistral-large-latest"

# ── Base system prompt ─────────────────────────────────────────
BASE_PROMPT = """You are Aether — an elite AI assistant engineered by Sai Chatre in 2026.
- Speak with confidence, clarity, and intellectual depth.
- Be warm and personable. Adapt tone: casual for small talk, precise for technical, empathetic for personal.
- Never use filler phrases like "Great question!" — get straight to the answer.
- Lead with the answer, then explain. Never bury the key point.
- For simple questions: reply in 1–3 sentences.
- For complex questions: use headers, bullet points, and numbered steps.
- Always wrap code in properly labelled markdown code blocks.
- ⚠️ CRITICAL FILE GENERATION RULE — YOU MUST FOLLOW THIS EXACTLY:
  When the user asks you to BUILD, CREATE, MAKE, or GENERATE any website, app, script, or complete file:
  YOU MUST wrap every single file in this exact XML tag format — NO EXCEPTIONS:
  <aether-file filename="index.html">FULL HTML CODE HERE</aether-file>
  <aether-file filename="styles.css">FULL CSS CODE HERE</aether-file>
  <aether-file filename="script.js">FULL JS CODE HERE</aether-file>
  RULES:
  • Each file gets its OWN separate aether-file tag
  • Use the EXACT tag format shown above — no variations
  • Do NOT use triple backtick code blocks for complete files — ONLY use aether-file tags
  • Triple backtick blocks are ONLY for short inline fixes or explaining single snippets
  • If you generate HTML that links to styles.css and script.js, also generate those files in separate tags
  • NEVER skip this format when building complete projects — the user's UI depends on it
- Always respond in the same language the user writes in."""

# ── Persona system prompts ─────────────────────────────────────
PERSONA_PROMPTS = {
    "direct": BASE_PROMPT + "\nCapabilities: software engineering, debugging, math, research, science, creative writing, business, translation, everyday conversation.",

    "byok": BASE_PROMPT + "\nCapabilities: software engineering, debugging, math, research, science, creative writing, business, translation, everyday conversation.",

    "coder": BASE_PROMPT + """
You are in CODER mode. Your sole focus is programming and software engineering.
- Always use proper syntax-highlighted code blocks with the correct language label.
- When debugging: identify the exact line/cause, explain why it's wrong, show the fix.
- Prefer concise, production-ready code over verbose explanations.
- If the user's code has multiple issues, list them all before fixing.
- Languages you excel at: Python, JavaScript, TypeScript, React, HTML/CSS, SQL, Bash, and more.""",

    "search": BASE_PROMPT + """
You are in SEARCH mode. You have access to real-time web data provided in [REAL-TIME LIVE DATA CONTEXT].
- Always prioritise the live data context over your training knowledge for current events.
- Extract exact figures, scores, names and dates directly from the provided snippets.
- Cite your source by referencing the snippet title (e.g. "According to BBC Sport...").
- If no live data is provided, say so clearly and answer from your training knowledge.""",

    "writer": BASE_PROMPT + """
You are in WRITER mode. You are a creative writing expert.
- Produce vivid, engaging, well-structured content.
- Match the user's requested tone: formal for essays, creative for stories, punchy for social media.
- For essays: clear thesis, body paragraphs, strong conclusion.
- For stories: show don't tell, use sensory details, build tension.
- For social posts: hook in the first line, concise, end with a call to action.
- Always offer to refine or change the style if the user wants.""",

    "tutor": BASE_PROMPT + """
You are in TUTOR mode. You are a patient, expert teacher.
- Break every concept into simple steps a beginner can follow.
- Use real-world analogies to explain abstract ideas.
- After explaining, always ask "Does that make sense? Want me to go deeper on any part?"
- Never overwhelm with too much at once — chunk information.
- If the student seems confused, try a completely different explanation approach.""",
}

# ── Tavily only fires for search persona OR time-sensitive queries ──
REALTIME_TRIGGERS = [
    "today","yesterday","tomorrow","tonight","right now","just now",
    "latest","recent","current","live","now","breaking",
    "news","update","score","match","result","winner","standings",
    "weather","temperature","forecast",
    "price","stock","crypto","bitcoin","market",
    "who won","what happened","did they","2025","2026",
    "this week","this month","this year",
    "trending","viral","released","launched","announced"
]

def needs_realtime_search(message, persona):
    if persona == "search":
        return True
    if persona in ("coder", "writer", "tutor"):
        return False
    msg_lower = message.lower()
    return any(trigger in msg_lower for trigger in REALTIME_TRIGGERS)


def search_the_web(query):
    if not TAVILY_API_KEY:
        return ""
    try:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "topic": "news",
            "time_range": "day",
            "search_depth": "basic",
            "max_results": 4
        }
        headers = {"Content-Type": "application/json"}
        res = requests.post(url, headers=headers, json=payload, timeout=5)
        if res.status_code != 200 or not res.json().get("results"):
            payload["topic"] = "general"
            res = requests.post(url, headers=headers, json=payload, timeout=5)
        if res.status_code == 200:
            results = res.json().get("results", [])
            parts = []
            for i, r in enumerate(results, 1):
                parts.append(f"--- LIVE DATA SOURCE {i}: {r.get('title','Source')} ---\nKEY INFORMATION: {r.get('content','')}\n")
            return "\n".join(parts)
    except Exception as e:
        print(f"Web search error: {e}")
    return ""


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "Aether AI backend is running", "model": DEFAULT_MODEL}), 200


@app.route("/api/chat", methods=["POST"])
def chat():
    data         = request.get_json(force=True)
    user_message = data.get("message", "").strip()
    history      = data.get("history", [])
    temperature  = float(data.get("temperature", 0.5))
    max_tokens   = int(data.get("max_tokens", 4096))
    byok_key     = data.get("byok_key", "").strip()
    model        = data.get("model", DEFAULT_MODEL)
    persona      = data.get("persona", "direct")

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    key_to_use = byok_key if byok_key else SERVER_API_KEY
    if not key_to_use:
        return jsonify({"error": "No API key configured."}), 400

    # 🕐 Dynamic date injection using user's timezone
    user_tz_str = data.get("timezone", "UTC")
    try:
        user_tz = zoneinfo.ZoneInfo(user_tz_str)
    except Exception:
        user_tz = timezone.utc
    now_local   = datetime.now(user_tz)
    date_anchor = now_local.strftime(f"Today is %A, %B %d, %Y. Local time: %I:%M %p ({user_tz_str}).")

    # Pick persona system prompt
    system_prompt = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["direct"])
    dynamic_system_prompt = f"[DATE CONTEXT] {date_anchor}\n\n" + system_prompt

    # 🌐 Smart Tavily — only fires when needed
    live_web_context = ""
    if needs_realtime_search(user_message, persona):
        live_web_context = search_the_web(user_message)

    if live_web_context:
        dynamic_system_prompt += (
            f"\n\n[REAL-TIME LIVE DATA CONTEXT]\n{live_web_context}\n\n"
            "Task: Read the snippets above carefully. Extract exact scores, dates, names and stats. "
            "Answer the user's question directly using this data."
        )

    messages = [{"role": "system", "content": dynamic_system_prompt}]
    for msg in history[-6:]:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})

    headers = {
        "Authorization": f"Bearer {key_to_use}",
        "Content-Type":  "application/json",
        "Accept":        "text/event-stream",
    }
    payload = {
        "model":       model,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "top_p":       0.7,
        "stream":      True,
    }

    try:
        sess = requests.Session()
        r    = sess.post(INVOKE_URL, headers=headers, json=payload, stream=True, timeout=(20, 120))
        r.raise_for_status()
        full_reply = ""
        for line in r.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8").strip()
            if line_str.startswith("data:"):
                chunk = line_str[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj   = json.loads(chunk)
                    delta = obj["choices"][0]["delta"].get("content", "")
                    if delta:
                        full_reply += delta
                except Exception:
                    pass
        if full_reply:
            return jsonify({"reply": full_reply})
        return jsonify({"error": "Empty response from AI"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 502


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, threaded=True)
