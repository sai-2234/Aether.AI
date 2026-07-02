import os
import json
import time
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

# ── Persistent HTTP session for connection reuse ───────────────
HTTP_SESSION = requests.Session()
HTTP_SESSION.headers.update({"Content-Type": "application/json"})

# ══════════════════════════════════════════════════════════════
#  TASK CLASSIFIER
#  Detects whether a request is small / medium / heavy
#  and returns optimal settings for each
# ══════════════════════════════════════════════════════════════
GREETING_PATTERNS = [
    "hi", "hello", "hey", "hii", "heyy", "what's up", "whats up",
    "how are you", "good morning", "good evening", "good night",
    "who are you", "what can you do", "help"
]

HEAVY_PATTERNS = [
    "build", "create", "make", "generate", "write a complete",
    "full website", "full app", "portfolio", "landing page",
    "entire", "whole project", "from scratch", "production ready",
    "all files", "html css js", "html, css", "multiple files",
    "web app", "dashboard", "e-commerce", "clone"
]

MEDIUM_PATTERNS = [
    "explain", "how does", "what is", "debug", "fix this",
    "help me", "review", "improve", "optimize", "refactor",
    "write a function", "write a script", "code for"
]

def classify_task(message: str) -> dict:
    msg = message.lower().strip()

    # Greeting / simple chat → small
    if any(p in msg for p in GREETING_PATTERNS) and len(msg) < 80:
        return {
            "type": "small",
            "max_tokens": 1024,
            "temperature": 0.7,
            "model": DEFAULT_MODEL,
            "status": "Thinking"
        }

    # Heavy code generation
    if any(p in msg for p in HEAVY_PATTERNS) or len(message) > 400:
        return {
            "type": "heavy",
            "max_tokens": 32768,
            "temperature": 0.3,
            "model": DEFAULT_MODEL,
            "status": "Coding"
        }

    # Medium — explanation, debugging, short scripts
    if any(p in msg for p in MEDIUM_PATTERNS):
        return {
            "type": "medium",
            "max_tokens": 4096,
            "temperature": 0.5,
            "model": DEFAULT_MODEL,
            "status": "Thinking"
        }

    # Default medium
    return {
        "type": "medium",
        "max_tokens": 4096,
        "temperature": 0.5,
        "model": DEFAULT_MODEL,
        "status": "Thinking"
    }


# ══════════════════════════════════════════════════════════════
#  SMART HISTORY TRIMMER
#  Keeps the last N exchanges but always preserves the most
#  recent user message for context quality
# ══════════════════════════════════════════════════════════════
def trim_history(history: list, task_type: str) -> list:
    if task_type == "small":
        keep = 4   # last 2 exchanges
    elif task_type == "heavy":
        keep = 6   # last 3 exchanges — more context for big tasks
    else:
        keep = 6

    trimmed = []
    for msg in history[-keep:]:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            # Truncate very long history messages to save tokens
            if len(content) > 2000:
                content = content[:2000] + "\n[... trimmed for context ...]"
            trimmed.append({"role": role, "content": content})
    return trimmed


# ══════════════════════════════════════════════════════════════
#  RETRY WRAPPER WITH EXPONENTIAL BACKOFF
#  Handles: timeout, 429, 500, 502, 503, connection errors
# ══════════════════════════════════════════════════════════════
RETRYABLE_STATUS = {429, 500, 502, 503}
MAX_RETRIES = 3

def call_nvidia_with_retry(headers: dict, payload: dict, task_type: str) -> str:
    # Timeouts: (connect, read) — heavy tasks get more read time
    if task_type == "heavy":
        timeout = (30, 600)
    elif task_type == "medium":
        timeout = (15, 120)
    else:
        timeout = (10, 45)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            r = HTTP_SESSION.post(
                INVOKE_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=timeout
            )

            # Retryable HTTP status codes
            if r.status_code in RETRYABLE_STATUS:
                wait = (2 ** attempt)  # 1s, 2s, 4s
                print(f"[Retry {attempt+1}] Status {r.status_code} — waiting {wait}s")
                time.sleep(wait)
                last_error = f"HTTP {r.status_code}"
                continue

            r.raise_for_status()

            # Stream the response
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
                return full_reply

            # Empty response — retry
            last_error = "Empty response from AI"
            wait = (2 ** attempt)
            print(f"[Retry {attempt+1}] Empty response — waiting {wait}s")
            time.sleep(wait)

        except requests.exceptions.Timeout:
            last_error = "Request timed out"
            wait = (2 ** attempt)
            print(f"[Retry {attempt+1}] Timeout — waiting {wait}s")
            time.sleep(wait)

        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection error: {e}"
            wait = (2 ** attempt)
            print(f"[Retry {attempt+1}] Connection error — waiting {wait}s")
            time.sleep(wait)

        except Exception as e:
            # Non-retryable error — fail immediately
            raise e

    raise Exception(f"Failed after {MAX_RETRIES} retries. Last error: {last_error}")


# ══════════════════════════════════════════════════════════════
#  SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════
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
  • NEVER truncate or shorten code — always write the COMPLETE, FULL code for every file
  • NEVER add comments like "// rest of code here" or "/* ... */" as placeholders
  • If the project needs 4000 lines of HTML, write all 4000 lines — do not skip anything
  • NEVER skip this format when building complete projects — the user's UI depends on it
- Always respond in the same language the user writes in."""

PERSONA_PROMPTS = {
    "direct": BASE_PROMPT + "\nCapabilities: software engineering, debugging, math, research, science, creative writing, business, translation, everyday conversation.",
    "byok":   BASE_PROMPT + "\nCapabilities: software engineering, debugging, math, research, science, creative writing, business, translation, everyday conversation.",
    "coder":  BASE_PROMPT + """
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
    "tutor":  BASE_PROMPT + """
You are in TUTOR mode. You are a patient, expert teacher.
- Break every concept into simple steps a beginner can follow.
- Use real-world analogies to explain abstract ideas.
- After explaining, always ask "Does that make sense? Want me to go deeper on any part?"
- Never overwhelm with too much at once — chunk information.
- If the student seems confused, try a completely different explanation approach.""",
}

# ══════════════════════════════════════════════════════════════
#  SMART TAVILY SEARCH
# ══════════════════════════════════════════════════════════════
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

def needs_realtime_search(message: str, persona: str) -> bool:
    if persona == "search":
        return True
    if persona in ("coder", "writer", "tutor"):
        return False
    return any(t in message.lower() for t in REALTIME_TRIGGERS)

def search_the_web(query: str) -> str:
    if not TAVILY_API_KEY:
        return ""
    try:
        url     = "https://api.tavily.com/search"
        payload = {"api_key": TAVILY_API_KEY, "query": query, "topic": "news",
                   "time_range": "day", "search_depth": "basic", "max_results": 4}
        hdrs    = {"Content-Type": "application/json"}
        res     = requests.post(url, headers=hdrs, json=payload, timeout=5)
        if res.status_code != 200 or not res.json().get("results"):
            payload["topic"] = "general"
            res = requests.post(url, headers=hdrs, json=payload, timeout=5)
        if res.status_code == 200:
            parts = []
            for i, r in enumerate(res.json().get("results", []), 1):
                parts.append(f"--- SOURCE {i}: {r.get('title','Source')} ---\n{r.get('content','')}\n")
            return "\n".join(parts)
    except Exception as e:
        print(f"Search error: {e}")
    return ""


# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "Aether AI backend is running", "model": DEFAULT_MODEL}), 200


@app.route("/api/chat", methods=["POST"])
def chat():
    data         = request.get_json(force=True)
    user_message = data.get("message", "").strip()
    history      = data.get("history", [])
    persona      = data.get("persona", "direct")
    byok_key     = data.get("byok_key", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    key_to_use = byok_key if byok_key else SERVER_API_KEY
    if not key_to_use:
        return jsonify({"error": "No API key configured. Add NVIDIA_API_KEY to Space secrets."}), 400

    # 🎯 Auto-classify task
    task     = classify_task(user_message)
    # Allow frontend overrides if explicitly sent
    max_tokens  = int(data.get("max_tokens") or task["max_tokens"])
    temperature = float(data.get("temperature") or task["temperature"])
    model       = data.get("model") or task["model"]
    task_type   = task["type"]

    # 🕐 Dynamic date injection
    user_tz_str = data.get("timezone", "UTC")
    try:
        user_tz = zoneinfo.ZoneInfo(user_tz_str)
    except Exception:
        user_tz = timezone.utc
    now_local   = datetime.now(user_tz)
    date_anchor = now_local.strftime(f"Today is %A, %B %d, %Y. Local time: %I:%M %p ({user_tz_str}).")

    # Build system prompt
    system_prompt = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["direct"])
    dynamic_system_prompt = f"[DATE CONTEXT] {date_anchor}\n\n" + system_prompt

    # 🌐 Smart web search
    if needs_realtime_search(user_message, persona):
        ctx = search_the_web(user_message)
        if ctx:
            dynamic_system_prompt += (
                f"\n\n[REAL-TIME LIVE DATA CONTEXT]\n{ctx}\n\n"
                "Extract exact scores, dates, names and stats. Answer directly using this data."
            )

    # 🧠 Smart history trimming
    messages = [{"role": "system", "content": dynamic_system_prompt}]
    messages += trim_history(history, task_type)
    messages.append({"role": "user", "content": user_message})

    req_headers = {
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
        reply = call_nvidia_with_retry(req_headers, payload, task_type)
        return jsonify({
            "reply":     reply,
            "task_type": task_type,
            "status":    task["status"]
        })
    except Exception as e:
        err = str(e)
        print(f"[ERROR] {err}")
        # Return friendly error messages
        if "timed out" in err.lower():
            return jsonify({"error": "⏱ Request timed out. Try a simpler prompt or try again."}), 504
        if "429" in err:
            return jsonify({"error": "⚡ Rate limit hit. Please wait a moment and try again."}), 429
        if "401" in err or "403" in err:
            return jsonify({"error": "🔑 Invalid API key. Check your key in settings."}), 401
        return jsonify({"error": f"❌ {err}"}), 502


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, threaded=True)
