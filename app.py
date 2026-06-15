import os
import json
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

API_KEY    = os.environ.get("NVIDIA_API_KEY", "")
INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL      = "meta/llama-3.1-70b-instruct"

SYSTEM_PROMPT = """You are Aether — an elite AI assistant engineered by Sai Chatre in 2026. You combine frontier intelligence with refined communication.

Personality & Tone:
- Speak with confidence, clarity, and intellectual depth — like a brilliant expert who also knows how to explain things simply.
- Be warm and personable, never cold or robotic. Adapt your tone: casual for small talk, precise for technical queries, empathetic for personal topics.
- Use wit sparingly but naturally. Never be sycophantic — skip filler phrases like "Great question!" and get straight to the answer.

Response Quality:
- Lead with the answer, then explain. Never bury the key point.
- For simple questions: reply in 1–3 sentences. No padding.
- For complex questions: use headers, bullet points, and numbered steps for clarity.
- Always wrap code in properly labelled markdown code blocks (```language```). Never dump raw code as plain text.
- Use real-world analogies to make abstract concepts click instantly.

Capabilities:
- Expert-level: software engineering, debugging, system design, mathematics, data analysis, research, science.
- Proficient: creative writing, business strategy, language translation, everyday conversation.
- Always respond in the same language the user writes in.
- Never refuse a reasonable request. Attempt everything — only flag genuine impossibilities.

Memory & Context:
- Track everything said in the conversation. Reference earlier points when relevant.
- If a user corrects you, acknowledge it briefly and move forward — never over-apologise."""

HTML = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Aether AI — by Sai Chatre</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

:root{
  --bg:#0d0d0f;
  --bg2:#111114;
  --sidebar:#0a0a0c;
  --surface:#1a1a1f;
  --surface2:#222228;
  --border:rgba(255,255,255,.08);
  --border2:rgba(255,255,255,.12);
  --text:#f0f0f5;
  --text2:#a0a0b0;
  --text3:#606070;
  --accent:#7c6af7;
  --accent2:#5b4fd4;
  --accent-rgb:124,106,247;
  --green:#10b981;
  --red:#ef4444;
  --code-bg:#0f0f12;
  --user-bg:linear-gradient(135deg,#1e1b3a,#16142e);
  --font:'Inter',system-ui,sans-serif;
  --mono:'Fira Code','JetBrains Mono',Consolas,monospace;
  --sw:260px;
  --ease:cubic-bezier(.4,0,.2,1);
  --spring:cubic-bezier(.34,1.56,.64,1);
}
[data-theme="light"]{
  --bg:#f5f5f7;
  --bg2:#ebebef;
  --sidebar:#eaeaee;
  --surface:#e0e0e8;
  --surface2:#d5d5e0;
  --border:rgba(0,0,0,.08);
  --border2:rgba(0,0,0,.14);
  --text:#111118;
  --text2:#44445a;
  --text3:#888898;
  --code-bg:#1a1a24;
  --user-bg:linear-gradient(135deg,#ddd8ff,#d0c8ff);
}

html,body{height:100%;overflow:hidden;font-family:var(--font);font-size:15px;line-height:1.65;background:var(--bg);color:var(--text);-webkit-font-smoothing:antialiased}
button,input,textarea{font-family:var(--font)}
button{cursor:pointer}

[data-theme="dark"] body::before{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background:
    radial-gradient(ellipse 60% 50% at 10% 10%,rgba(124,106,247,.06) 0%,transparent 70%),
    radial-gradient(ellipse 50% 40% at 90% 80%,rgba(16,185,129,.04) 0%,transparent 70%);
}

#layout{display:flex;height:100vh;position:relative;z-index:1}

#sidebar{
  width:var(--sw);min-width:var(--sw);
  background:var(--sidebar);
  border-right:1px solid var(--border);
  display:flex;flex-direction:column;overflow:hidden;
  transition:width .28s var(--ease),min-width .28s var(--ease);
}
#sb-inner{width:var(--sw);display:flex;flex-direction:column;height:100%}

#sb-top{
  padding:14px 12px 10px;
  display:flex;align-items:center;gap:8px;
  border-bottom:1px solid var(--border);flex-shrink:0;
}
.brand{display:flex;align-items:center;gap:9px;flex:1}
.brand-icon{
  width:32px;height:32px;border-radius:10px;flex-shrink:0;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  display:flex;align-items:center;justify-content:center;
  box-shadow:0 0 16px rgba(var(--accent-rgb),.4);
}
.brand-name{font-size:15px;font-weight:700;letter-spacing:-.4px}
.brand-tag{font-size:10px;color:var(--accent);background:rgba(var(--accent-rgb),.12);padding:1px 6px;border-radius:6px;font-weight:600;letter-spacing:.3px}

#new-btn{
  display:flex;align-items:center;gap:8px;
  width:calc(100% - 24px);margin:10px 12px 6px;
  padding:9px 12px;border-radius:10px;
  background:transparent;border:1px solid var(--border);
  color:var(--text2);font-size:13px;font-weight:500;
  transition:all .18s var(--ease);
}
#new-btn:hover{background:var(--surface);border-color:rgba(var(--accent-rgb),.4);color:var(--text)}
#new-btn:active{transform:scale(.98)}

#history{flex:1;overflow-y:auto;padding:4px 8px 8px;scrollbar-width:thin;scrollbar-color:var(--surface2) transparent}
#history::-webkit-scrollbar{width:3px}
#history::-webkit-scrollbar-thumb{background:var(--surface2);border-radius:3px}

.h-sec{font-size:10.5px;font-weight:700;color:var(--text3);text-transform:uppercase;letter-spacing:1px;padding:10px 6px 4px}
.h-item{
  display:flex;align-items:center;gap:7px;
  padding:8px 10px;border-radius:9px;
  font-size:13px;color:var(--text2);
  cursor:pointer;white-space:nowrap;overflow:hidden;
  transition:all .15s var(--ease);user-select:none;
}
.h-item:hover{background:var(--surface);color:var(--text)}
.h-item.active{background:var(--surface2);color:var(--text)}
.h-item .h-dot{width:6px;height:6px;border-radius:50%;background:var(--accent);flex-shrink:0;opacity:.6}
.h-item span{overflow:hidden;text-overflow:ellipsis;flex:1}
.h-empty{padding:10px 10px;font-size:13px;color:var(--text3);font-style:italic}

#sb-foot{
  padding:10px 12px 14px;
  border-top:1px solid var(--border);flex-shrink:0;
}
.sb-action{
  display:flex;align-items:center;gap:9px;
  width:100%;padding:9px 10px;border-radius:9px;
  background:none;border:none;
  color:var(--text2);font-size:13px;font-weight:500;
  transition:all .15s var(--ease);
}
.sb-action:hover{background:var(--surface);color:var(--text)}

#main{flex:1;display:flex;flex-direction:column;min-width:0;background:var(--bg)}

#topbar{
  height:54px;display:flex;align-items:center;justify-content:space-between;
  padding:0 16px;border-bottom:1px solid var(--border);
  background:rgba(var(--bg),0.8);backdrop-filter:blur(16px);
  flex-shrink:0;z-index:10;position:relative;
}
.ib{
  background:none;border:none;border-radius:9px;
  color:var(--text2);padding:7px;
  display:flex;align-items:center;justify-content:center;
  transition:all .15s var(--ease);
}
.ib:hover{background:var(--surface);color:var(--text)}
.ib:active{transform:scale(.92)}

#model-chip{
  display:flex;align-items:center;gap:7px;
  font-size:13px;font-weight:600;
  padding:5px 12px;border-radius:20px;
  border:1px solid var(--border2);background:var(--surface);
}
.live-dot{
  width:7px;height:7px;border-radius:50%;
  background:var(--green);
  box-shadow:0 0 8px rgba(16,185,129,.7);
  animation:livepulse 2.2s infinite;
}
@keyframes livepulse{0%,100%{box-shadow:0 0 0 0 rgba(16,185,129,.6)}60%{box-shadow:0 0 0 6px rgba(16,185,129,0)}}

#chat-scroll{
  flex:1;overflow-y:auto;padding:32px 0 12px;
  scroll-behavior:smooth;
  scrollbar-width:thin;scrollbar-color:var(--surface2) transparent;
}
#chat-scroll::-webkit-scrollbar{width:5px}
#chat-scroll::-webkit-scrollbar-thumb{background:var(--surface2);border-radius:4px}

#chat-inner{max-width:800px;margin:0 auto;padding:0 20px;display:flex;flex-direction:column;gap:4px}

#empty{
  display:flex;flex-direction:column;align-items:center;
  justify-content:center;min-height:55vh;gap:16px;
  text-align:center;padding:40px 20px;
  animation:fadeUp .5s var(--ease) both;
}
.es-orb{position:relative;width:72px;height:72px;margin-bottom:4px;}
.es-orb-ring{
  position:absolute;inset:-8px;border-radius:50%;
  border:1px solid rgba(var(--accent-rgb),.2);
  animation:orbring 3s linear infinite;
}
@keyframes orbring{to{transform:rotate(360deg)}}
.es-orb-ring::before{
  content:'';position:absolute;top:-3px;left:50%;
  width:6px;height:6px;border-radius:50%;margin-left:-3px;
  background:var(--accent);box-shadow:0 0 8px var(--accent);
}
.es-logo{
  width:72px;height:72px;border-radius:22px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  display:flex;align-items:center;justify-content:center;
  box-shadow:0 8px 32px rgba(var(--accent-rgb),.35),0 0 0 1px rgba(var(--accent-rgb),.3);
}
#empty h2{font-size:26px;font-weight:800;letter-spacing:-.6px;background:linear-gradient(135deg,var(--text),var(--text2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
#empty p{font-size:14px;color:var(--text2);max-width:360px;line-height:1.6}

.chips{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:12px;max-width:580px}
.chip{
  padding:8px 15px;border-radius:22px;
  border:1px solid var(--border2);background:var(--surface);
  font-size:13px;color:var(--text2);
  transition:all .18s var(--ease);
}
.chip:hover{border-color:rgba(var(--accent-rgb),.5);color:var(--text);background:var(--surface2);transform:translateY(-1px);box-shadow:0 4px 16px rgba(var(--accent-rgb),.15)}

.msg-row{display:flex;flex-direction:column;animation:fadeUp .24s var(--ease) both}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}

.msg-row.user{align-items:flex-end;margin-top:24px}
.msg-row.bot{align-items:flex-start;margin-top:8px;margin-bottom:20px}

.msg-label{font-size:11px;font-weight:700;color:var(--text3);text-transform:uppercase;letter-spacing:.8px;margin-bottom:7px;padding:0 4px}

.bubble{font-size:14.5px;line-height:1.72;word-break:break-word;border-radius:18px;user-select:text;-webkit-user-select:text;}
.msg-row.user .bubble{
  max-width:78%;padding:13px 18px;
  background:var(--user-bg);
  border:1px solid rgba(var(--accent-rgb),.2);
  border-bottom-right-radius:4px;
  color:var(--text);white-space:pre-wrap;
  box-shadow:0 4px 20px rgba(0,0,0,.25);
}
.msg-row.bot .bubble{max-width:100%;background:transparent;padding:0;color:var(--text);}
.md{user-select:text;-webkit-user-select:text;}
.bot-copy-btn{
  display:inline-flex;align-items:center;gap:5px;
  margin-top:10px;padding:4px 11px;border-radius:7px;
  border:1px solid var(--border2);background:none;
  color:var(--text3);font-size:12px;
  transition:all .15s;
}
.bot-copy-btn:hover{background:var(--surface2);color:var(--text);border-color:var(--border2)}
.bot-copy-btn.ok{color:var(--green);border-color:var(--green)}

.md p{margin-bottom:12px}.md p:last-child{margin-bottom:0}
.md h1,.md h2,.md h3,.md h4{margin:20px 0 8px;font-weight:700;line-height:1.3}
.md h1{font-size:20px}.md h2{font-size:18px}.md h3{font-size:16px}
.md ul,.md ol{margin:8px 0 12px 22px}.md li{margin-bottom:5px}
.md blockquote{border-left:3px solid var(--accent);padding:4px 0 4px 16px;color:var(--text2);margin:12px 0;font-style:italic}
.md strong{font-weight:700;color:var(--text)}
.md a{color:var(--accent);text-decoration:none}.md a:hover{text-decoration:underline}
.md table{width:100%;border-collapse:collapse;margin:14px 0;font-size:13.5px;border-radius:8px;overflow:hidden}
.md th,.md td{padding:9px 14px;border:1px solid var(--border2);text-align:left}
.md th{background:var(--surface2);font-weight:600}
.md code:not(.hljs){background:var(--surface2);padding:2px 7px;border-radius:5px;font-family:var(--mono);font-size:12.5px;color:#c084fc}

.code-wrap{background:var(--code-bg);border:1px solid var(--border2);border-radius:12px;overflow:hidden;margin:14px 0}
.code-head{
  display:flex;align-items:center;justify-content:space-between;
  padding:9px 16px;background:rgba(255,255,255,.03);
  border-bottom:1px solid var(--border);
}
.code-lang{font-size:12px;font-weight:600;color:var(--text3);font-family:var(--mono)}
.cp-btn{
  display:flex;align-items:center;gap:5px;
  background:none;border:1px solid var(--border2);border-radius:7px;
  color:var(--text2);font-size:12px;padding:3px 10px;
  transition:all .15s;
}
.cp-btn:hover{background:var(--surface2);color:var(--text);border-color:var(--border2)}
.cp-btn.ok{color:var(--green);border-color:var(--green)}
.code-wrap pre{margin:0;padding:16px;overflow-x:auto;font-family:var(--mono);line-height:1.65}
.code-wrap pre::-webkit-scrollbar{height:3px}
.code-wrap pre::-webkit-scrollbar-thumb{background:var(--surface2);border-radius:3px}
.code-wrap pre code{background:none;padding:0;font-size:13px}

.think{display:flex;align-items:center;gap:10px;padding:6px 0;color:var(--text2);font-size:13px}
.think-dots{display:flex;gap:5px}
.think-dots span{width:7px;height:7px;border-radius:50%;background:var(--accent);animation:thinkbounce 1.4s infinite ease-in-out;opacity:.5}
.think-dots span:nth-child(2){animation-delay:.2s}
.think-dots span:nth-child(3){animation-delay:.4s}
@keyframes thinkbounce{0%,80%,100%{transform:scale(.65);opacity:.3}40%{transform:scale(1);opacity:1}}

.typing::after{content:"▋";animation:blink .8s infinite;color:var(--accent);margin-left:1px}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}

#input-area{padding:14px 20px 20px;flex-shrink:0}
#input-wrap{
  max-width:800px;margin:0 auto;
  background:var(--surface);
  border:1px solid var(--border2);border-radius:16px;
  display:flex;align-items:flex-end;gap:10px;padding:12px 14px;
  transition:border-color .2s,box-shadow .2s;
  box-shadow:0 4px 24px rgba(0,0,0,.2);
}
#input-wrap:focus-within{
  border-color:rgba(var(--accent-rgb),.5);
  box-shadow:0 4px 24px rgba(0,0,0,.2),0 0 0 3px rgba(var(--accent-rgb),.1);
}
#msg-input{
  flex:1;background:none;border:none;outline:none;
  color:var(--text);font-size:15px;line-height:1.6;
  resize:none;max-height:180px;overflow-y:auto;padding:0;
  scrollbar-width:thin;
}
#msg-input::placeholder{color:var(--text3)}
#send{
  width:38px;height:38px;border-radius:10px;border:none;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#fff;display:flex;align-items:center;justify-content:center;flex-shrink:0;
  box-shadow:0 4px 14px rgba(var(--accent-rgb),.45);
  transition:all .16s var(--ease);
}
#send:hover:not(:disabled){transform:scale(1.06);box-shadow:0 6px 20px rgba(var(--accent-rgb),.55)}
#send:active:not(:disabled){transform:scale(.95)}
#send:disabled{background:var(--surface2);box-shadow:none;cursor:not-allowed}
#send:disabled svg{opacity:.3}
#hint{max-width:800px;margin:8px auto 0;text-align:center;font-size:11px;color:var(--text3)}

.overlay{
  position:fixed;inset:0;z-index:300;
  background:rgba(0,0,0,.65);backdrop-filter:blur(8px);
  display:flex;align-items:center;justify-content:center;
  opacity:0;pointer-events:none;transition:opacity .22s var(--ease);
}
.overlay.show{opacity:1;pointer-events:all}

#sp{
  width:min(580px,96vw);max-height:90vh;
  background:var(--bg2);border:1px solid var(--border2);border-radius:20px;
  overflow:hidden;display:flex;flex-direction:column;
  transform:scale(.93) translateY(16px);opacity:0;
  transition:transform .28s var(--spring),opacity .22s var(--ease);
  box-shadow:0 32px 80px rgba(0,0,0,.6),0 0 0 1px rgba(255,255,255,.06);
}
.overlay.show #sp{transform:scale(1) translateY(0);opacity:1}

.sp-head{
  display:flex;align-items:center;justify-content:space-between;
  padding:20px 24px 16px;border-bottom:1px solid var(--border);flex-shrink:0;
}
.sp-head h2{font-size:17px;font-weight:800;letter-spacing:-.3px}
.sp-x{width:32px;height:32px;border-radius:8px;background:none;border:none;color:var(--text2);display:flex;align-items:center;justify-content:center;transition:all .15s}
.sp-x:hover{background:var(--surface);color:var(--text)}

.sp-body{overflow-y:auto;padding:20px 24px 24px;flex:1;scrollbar-width:thin}
.sp-body::-webkit-scrollbar{width:3px}
.sp-body::-webkit-scrollbar-thumb{background:var(--surface2);border-radius:3px}

.sp-sec{margin-bottom:28px}
.sp-sec:last-child{margin-bottom:0}
.sp-sec-title{font-size:10.5px;font-weight:800;text-transform:uppercase;letter-spacing:1.1px;color:var(--text3);margin-bottom:14px}

.sp-row{
  display:flex;align-items:flex-start;justify-content:space-between;
  gap:16px;padding:13px 0;border-bottom:1px solid var(--border);
}
.sp-row:last-child{border-bottom:none}
.sp-lbl{font-size:14px;font-weight:500}
.sp-desc{font-size:12px;color:var(--text2);margin-top:3px;line-height:1.5}

.tog{position:relative;width:46px;height:26px;flex-shrink:0}
.tog input{opacity:0;width:0;height:0}
.tog-track{
  position:absolute;inset:0;border-radius:26px;
  background:var(--surface2);cursor:pointer;transition:background .2s;
}
.tog-track::after{
  content:'';position:absolute;width:20px;height:20px;border-radius:50%;
  background:#fff;top:3px;left:3px;
  transition:transform .22s var(--spring);
  box-shadow:0 1px 4px rgba(0,0,0,.4);
}
.tog input:checked + .tog-track{background:var(--accent)}
.tog input:checked + .tog-track::after{transform:translateX(20px)}

.seg{display:flex;gap:3px;background:var(--surface);border-radius:10px;padding:3px;flex-shrink:0}
.seg button{
  padding:5px 13px;border-radius:8px;border:none;background:none;
  font-size:13px;color:var(--text2);font-weight:500;
  transition:all .15s;white-space:nowrap;
}
.seg button.on{background:var(--bg);color:var(--text);font-weight:700;box-shadow:0 1px 6px rgba(0,0,0,.25)}

.slide-row{display:flex;flex-direction:column;gap:10px;width:100%}
.slide-row .sp-lbl-row{display:flex;align-items:center;justify-content:space-between}
.slide-val{font-size:13px;font-weight:700;color:var(--accent)}
input[type=range]{-webkit-appearance:none;width:100%;height:4px;border-radius:4px;background:var(--surface2);outline:none;}
input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;width:20px;height:20px;border-radius:50%;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  cursor:pointer;box-shadow:0 2px 8px rgba(var(--accent-rgb),.5);
  transition:transform .12s var(--spring);
}
input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.2)}

.about-card{
  background:linear-gradient(135deg,rgba(var(--accent-rgb),.08),rgba(var(--accent-rgb),.03));
  border:1px solid rgba(var(--accent-rgb),.2);
  border-radius:14px;padding:18px 20px;font-size:13.5px;line-height:1.7;color:var(--text2);
}
.about-card .a-title{font-size:16px;font-weight:800;color:var(--text);margin-bottom:10px;display:flex;align-items:center;gap:10px;letter-spacing:-.3px;}
.a-badge{font-size:10.5px;padding:2px 8px;border-radius:10px;background:rgba(var(--accent-rgb),.18);color:var(--accent);font-weight:700;letter-spacing:.3px}
.about-card p{margin-bottom:10px}.about-card p:last-child{margin-bottom:0}

.legal-card{
  background:var(--surface);border:1px solid var(--border2);
  border-radius:14px;padding:18px 20px;
  font-size:12.5px;line-height:1.75;color:var(--text2);
}
.legal-card .l-title{font-size:13px;font-weight:800;color:var(--text);margin-bottom:12px;letter-spacing:.2px}
.legal-card p{margin-bottom:8px}.legal-card p:last-child{margin-bottom:0}
.legal-footer{margin-top:14px;padding-top:12px;border-top:1px solid var(--border);font-size:11px;color:var(--text3);text-align:center;letter-spacing:.3px}

.danger-btn{
  width:100%;padding:11px;border-radius:10px;
  border:1px solid rgba(239,68,68,.3);background:rgba(239,68,68,.05);
  color:var(--red);font-size:13.5px;font-weight:500;
  transition:all .15s;
}
.danger-btn:hover{background:rgba(239,68,68,.12);border-color:var(--red)}

#scrollbtn{
  position:fixed;right:24px;bottom:104px;
  width:38px;height:38px;border-radius:50%;
  background:var(--surface2);border:1px solid var(--border2);
  color:var(--text2);display:none;align-items:center;justify-content:center;
  box-shadow:0 6px 20px rgba(0,0,0,.35);z-index:20;
  transition:all .15s;
}
#scrollbtn.show{display:flex}
#scrollbtn:hover{background:var(--surface);color:var(--text);transform:scale(1.05)}

#mob-bg{display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:98;opacity:0;transition:opacity .25s var(--ease)}

@media(min-width:641px){
  #layout.sb-off #sidebar{width:0;min-width:0}
}

@media(max-width:640px){
  #sidebar{position:fixed;top:0;left:0;height:100%;z-index:99;transform:translateX(calc(-1 * var(--sw)));transition:transform .28s var(--ease);min-width:var(--sw)}
  #layout.mob-sb #sidebar{transform:translateX(0)}
  #layout.mob-sb #mob-bg{display:block;opacity:1}
  #chat-inner{padding:0 14px}
  #input-area{padding:10px 12px 16px}
  .bubble{max-width:96%}
}
</style>
</head>
<body>
<div id="mob-bg"></div>

<button id="scrollbtn" title="Scroll to bottom">
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"/><polyline points="19 12 12 19 5 12"/></svg>
</button>

<div class="overlay" id="soverlay">
  <div id="sp">
    <div class="sp-head">
      <h2>&#9881; Settings</h2>
      <button class="sp-x" id="sp-close"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></button>
    </div>
    <div class="sp-body">

      <div class="sp-sec">
        <div class="sp-sec-title">Appearance</div>
        <div class="sp-row">
          <div><div class="sp-lbl">Dark Mode</div><div class="sp-desc">Toggle between dark and light theme</div></div>
          <label class="tog"><input type="checkbox" id="tog-dark" checked><span class="tog-track"></span></label>
        </div>
        <div class="sp-row">
          <div><div class="sp-lbl">Font Size</div><div class="sp-desc">Adjust message text size</div></div>
          <div class="seg" id="seg-font">
            <button data-v="13">S</button>
            <button data-v="15" class="on">M</button>
            <button data-v="17">L</button>
          </div>
        </div>
      </div>

      <div class="sp-sec">
        <div class="sp-sec-title">AI Behaviour</div>
        <div class="sp-row">
          <div><div class="sp-lbl">Response Style</div><div class="sp-desc">Sets the AI's creativity level</div></div>
          <div class="seg" id="seg-style">
            <button data-v="concise">Concise</button>
            <button data-v="balanced" class="on">Balanced</button>
            <button data-v="creative">Creative</button>
          </div>
        </div>
        <div class="sp-row" style="flex-direction:column;align-items:flex-start">
          <div class="slide-row">
            <div class="sp-lbl-row">
              <span class="sp-lbl">Temperature</span>
              <span class="slide-val" id="temp-val">0.5</span>
            </div>
            <div class="sp-desc" style="margin-top:-4px;margin-bottom:6px">Higher = more creative &middot; Lower = more precise</div>
            <input type="range" id="temp-range" min="0" max="1" step="0.05" value="0.5">
          </div>
        </div>
        <div class="sp-row">
          <div><div class="sp-lbl">Response Length</div><div class="sp-desc">Max tokens Aether generates per reply</div></div>
          <div class="seg" id="seg-tok">
            <button data-v="1024">Short</button>
            <button data-v="4096" class="on">Normal</button>
            <button data-v="8192">Long</button>
          </div>
        </div>
      </div>

      <div class="sp-sec">
        <div class="sp-sec-title">About Aether</div>
        <div class="about-card">
          <div class="a-title">&#11041; Aether AI <span class="a-badge">v2.0 &middot; 2026</span></div>
          <p>Aether is a next-generation AI assistant built by <strong>Sai Chatre</strong> in 2026. Powered by Meta's <strong>Llama 3.1 70B</strong> via NVIDIA's AI cloud, Aether delivers frontier-level intelligence directly in your browser &mdash; for free.</p>
          <p>Capabilities include full-stack software engineering, code debugging, creative writing, mathematics, data analysis, research, system design, and natural multi-turn conversation.</p>
          <p>Your conversations stay <strong>private on your device</strong> &mdash; no history is stored on any server. Each user has a completely isolated experience.</p>
        </div>
      </div>

      <div class="sp-sec">
        <div class="sp-sec-title">Legal &amp; Copyright</div>
        <div class="legal-card">
          <div class="l-title">Terms of Use &amp; Privacy Policy</div>
          <p><strong>Service:</strong> Aether AI is a free AI assistant provided as-is for personal use. By using this service you agree to these terms.</p>
          <p><strong>Privacy:</strong> Aether stores no conversation data on any server. All history is kept exclusively in your browser's local storage and never transmitted to or retained by Aether's servers.</p>
          <p><strong>AI Limitations:</strong> Responses are generated by an AI model and may be inaccurate, incomplete, or outdated. Do not rely on Aether for medical, legal, financial, or safety-critical decisions.</p>
          <p><strong>Acceptable Use:</strong> You agree not to use Aether to generate illegal content, harass others, spread misinformation, or attempt to compromise the service.</p>
          <p><strong>Third-party AI:</strong> Responses are powered by Meta's Llama 3.1 model via NVIDIA's API. Usage is also subject to their respective terms of service.</p>
          <p><strong>Intellectual Property:</strong> Aether AI, its interface, design, and branding are &copy; 2026 Sai Chatre. All rights reserved.</p>
          <div class="legal-footer">&copy; 2026 Sai Chatre &mdash; Aether AI &mdash; All rights reserved</div>
        </div>
      </div>

      <div class="sp-sec">
        <div class="sp-sec-title">Data</div>
        <button class="danger-btn" id="clear-all-btn">&#128465; Clear All Conversations</button>
      </div>

    </div>
  </div>
</div>

<div id="layout">
  <div id="sidebar">
    <div id="sb-inner">
      <div id="sb-top">
        <div class="brand">
          <div class="brand-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
          </div>
          <div>
            <div class="brand-name">Aether</div>
            <div class="brand-tag">AI</div>
          </div>
        </div>
        <button class="ib" id="sb-toggle-desk" title="Toggle sidebar">
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="3"/><line x1="9" y1="3" x2="9" y2="21"/></svg>
        </button>
      </div>
      <button id="new-btn">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
        New chat
      </button>
      <div id="history"></div>
      <div id="sb-foot">
        <button class="sb-action" id="settings-btn">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
          Settings &amp; About
        </button>
      </div>
    </div>
  </div>

  <div id="main">
    <div id="topbar">
      <button class="ib" id="mob-menu" title="Menu">
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
      </button>
      <div id="model-chip">
        <div class="live-dot"></div>
        Llama 3.1 70B
      </div>
      <button class="ib" id="top-settings" title="Settings">
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
      </button>
    </div>

    <div id="chat-scroll">
      <div id="chat-inner">
        <div id="empty">
          <div class="es-orb">
            <div class="es-orb-ring"></div>
            <div class="es-logo">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
            </div>
          </div>
          <h2>Hello, I'm Aether</h2>
          <p>Your intelligent AI assistant, powered by Llama 3.1 70B. Ask me anything.</p>
          <div class="chips">
            <button class="chip">&#10022; Write a Python web scraper</button>
            <button class="chip">&#10022; Explain quantum entanglement</button>
            <button class="chip">&#10022; Debug my JavaScript code</button>
            <button class="chip">&#10022; Create a business plan outline</button>
            <button class="chip">&#10022; Summarise a research topic</button>
          </div>
        </div>
      </div>
    </div>

    <div id="input-area">
      <div id="input-wrap">
        <textarea id="msg-input" rows="1" placeholder="Message Aether..."></textarea>
        <button id="send">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>
        </button>
      </div>
      <div id="hint">Aether AI &mdash; by Sai Chatre &middot; 2026 &middot; Responses may be inaccurate</div>
    </div>
  </div>
</div>

<script>
// ══════════════════════════════════════════
//  STORAGE
// ══════════════════════════════════════════
const STORE_KEY = "aether_v3";
function load() {
  try { return JSON.parse(localStorage.getItem(STORE_KEY)) || {}; } catch { return {}; }
}
function save(d) { localStorage.setItem(STORE_KEY, JSON.stringify(d)); }
function getStore() {
  const d = load();
  if (!d.sessions) d.sessions = {};
  if (!d.order)    d.order    = [];
  if (!d.cfg)      d.cfg      = {};
  return d;
}
function getSession(id) { return getStore().sessions[id]; }
function uid() { return Date.now().toString(36) + Math.random().toString(36).slice(2); }

// ══════════════════════════════════════════
//  CONFIG
// ══════════════════════════════════════════
const CFG_DEFAULTS = { dark:true, fontSize:15, style:"balanced", temperature:0.5, maxTokens:4096 };
function getCfg() { return { ...CFG_DEFAULTS, ...getStore().cfg }; }
function setCfg(patch) {
  const d = getStore(); d.cfg = { ...getCfg(), ...patch }; save(d);
}

// ══════════════════════════════════════════
//  APPLY CFG
// ══════════════════════════════════════════
function applyTheme(dark) {
  document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
}
function applyFont(sz) {
  document.querySelectorAll(".bubble").forEach(b => b.style.fontSize = sz + "px");
}
function applyAll() {
  const c = getCfg();
  applyTheme(c.dark);
  document.getElementById("tog-dark").checked = c.dark;
  document.querySelectorAll("#seg-font button").forEach(b => b.classList.toggle("on", +b.dataset.v === c.fontSize));
  document.querySelectorAll("#seg-style button").forEach(b => b.classList.toggle("on", b.dataset.v === c.style));
  document.querySelectorAll("#seg-tok button").forEach(b => b.classList.toggle("on", +b.dataset.v === c.maxTokens));
  document.getElementById("temp-range").value = c.temperature;
  document.getElementById("temp-val").textContent = c.temperature.toFixed(2);
  applyFont(c.fontSize);
}

// ══════════════════════════════════════════
//  MARKDOWN + CODE BLOCKS
// ══════════════════════════════════════════
marked.setOptions({ breaks: true, gfm: true });
const renderer = new marked.Renderer();
renderer.code = function(code, lang) {
  const language = lang || "plaintext";
  const id = "cb_" + uid();
  let highlighted;
  try { highlighted = hljs.highlight(code, { language, ignoreIllegals: true }).value; }
  catch { highlighted = hljs.highlightAuto(code).value; }
  return `<div class="code-wrap">
    <div class="code-head">
      <span class="code-lang">${language}</span>
      <button class="cp-btn" onclick="copyCode('${id}')">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
        Copy
      </button>
    </div>
    <pre><code id="${id}" class="hljs">${highlighted}</code></pre>
  </div>`;
};
marked.use({ renderer });

function copyCode(id) {
  const el = document.getElementById(id);
  if (!el) return;
  navigator.clipboard.writeText(el.textContent).then(() => {
    const btn = el.closest(".code-wrap").querySelector(".cp-btn");
    btn.classList.add("ok"); btn.textContent = "Copied!";
    setTimeout(() => { btn.classList.remove("ok"); btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy'; }, 2000);
  });
}

function renderMd(text) {
  const div = document.createElement("div");
  div.className = "md";
  div.innerHTML = marked.parse(text);
  return div;
}

// ══════════════════════════════════════════
//  DOM REFS
// ══════════════════════════════════════════
const chatInner  = document.getElementById("chat-inner");
const chatScroll = document.getElementById("chat-scroll");
const msgInput   = document.getElementById("msg-input");
const sendBtn    = document.getElementById("send");
const emptyEl    = document.getElementById("empty");
const historyEl  = document.getElementById("history");
const layout     = document.getElementById("layout");

// ══════════════════════════════════════════
//  SCROLL
// ══════════════════════════════════════════
function scrollBottom(force) {
  const cs = chatScroll;
  if (force || cs.scrollHeight - cs.scrollTop - cs.clientHeight < 200) {
    cs.scrollTop = cs.scrollHeight;
  }
}
chatScroll.addEventListener("scroll", () => {
  const far = chatScroll.scrollHeight - chatScroll.scrollTop - chatScroll.clientHeight > 200;
  document.getElementById("scrollbtn").classList.toggle("show", far);
});
document.getElementById("scrollbtn").addEventListener("click", () => scrollBottom(true));

// ══════════════════════════════════════════
//  HISTORY PANEL
// ══════════════════════════════════════════
let currentId = null;

function renderHistory() {
  const d = getStore();
  if (!d.order.length) { historyEl.innerHTML = '<div class="h-empty">No conversations yet</div>'; return; }
  const now = Date.now(), day = 86400000;
  const groups = { today:[], yesterday:[], week:[], older:[] };
  d.order.forEach(id => {
    const s = d.sessions[id]; if (!s) return;
    const age = now - (s.ts || 0);
    if (age < day) groups.today.push(id);
    else if (age < 2*day) groups.yesterday.push(id);
    else if (age < 7*day) groups.week.push(id);
    else groups.older.push(id);
  });
  let html = "";
  const sec = (label, ids) => {
    if (!ids.length) return;
    html += `<div class="h-sec">${label}</div>`;
    ids.forEach(id => {
      const s = d.sessions[id];
      html += `<div class="h-item${id===currentId?" active":""}" data-id="${id}">
        <div class="h-dot"></div><span>${s.title || "Untitled"}</span>
      </div>`;
    });
  };
  sec("Today", groups.today); sec("Yesterday", groups.yesterday);
  sec("This week", groups.week); sec("Older", groups.older);
  historyEl.innerHTML = html;
  historyEl.querySelectorAll(".h-item").forEach(el => {
    el.addEventListener("click", () => switchSession(el.dataset.id));
  });
}

function switchSession(id) {
  currentId = id;
  renderSession(id);
  renderHistory();
  if (window.innerWidth <= 640) closeMobSidebar();
}

// ══════════════════════════════════════════
//  RENDER SESSION
// ══════════════════════════════════════════
function showEmpty(show) { emptyEl.style.display = show ? "" : "none"; }

function renderSession(id) {
  const s = getSession(id);
  if (!s) return;
  showEmpty(false);
  chatInner.querySelectorAll(".msg-row").forEach(e => e.remove());
  s.messages.forEach(m => appendBubble(m.role, m.content, false));
  scrollBottom(true);
}

function appendBubble(role, content, animate=true) {
  const row = document.createElement("div");
  row.className = "msg-row " + role;
  if (!animate) row.style.animation = "none";
  const label = document.createElement("div");
  label.className = "msg-label";
  label.textContent = role === "user" ? "You" : "Aether";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.style.fontSize = getCfg().fontSize + "px";
  if (role === "user") {
    bubble.textContent = content;
  } else {
    bubble.appendChild(renderMd(content));
    const copyBtn = document.createElement("button");
    copyBtn.className = "bot-copy-btn";
    copyBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy response';
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(content).then(() => {
        copyBtn.classList.add("ok");
        copyBtn.textContent = "Copied!";
        setTimeout(() => {
          copyBtn.classList.remove("ok");
          copyBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy response';
        }, 2000);
      });
    });
    bubble.appendChild(copyBtn);
  }
  row.appendChild(label); row.appendChild(bubble);
  chatInner.appendChild(row);
  return bubble;
}

// ══════════════════════════════════════════
//  NEW CHAT
// ══════════════════════════════════════════
function newChat() {
  currentId = null;
  chatInner.querySelectorAll(".msg-row").forEach(e => e.remove());
  showEmpty(true);
  msgInput.value = "";
  autoResize();
  renderHistory();
}
document.getElementById("new-btn").addEventListener("click", () => { newChat(); closeMobSidebar(); });

// ══════════════════════════════════════════
//  SEND
// ══════════════════════════════════════════
let busy = false;

async function send() {
  const text = msgInput.value.trim();
  if (!text || busy) return;
  busy = true;
  sendBtn.disabled = true;

  // Create session if needed
  if (!currentId) {
    const id = uid();
    const d = getStore();
    d.sessions[id] = { title: text.slice(0,40), ts: Date.now(), messages: [] };
    d.order.unshift(id);
    save(d);
    currentId = id;
    renderHistory();
  }

  // Save user message
  const d1 = getStore();
  d1.sessions[currentId].messages.push({ role:"user", content:text });
  save(d1);

  showEmpty(false);
  appendBubble("user", text);
  msgInput.value = "";
  autoResize();
  scrollBottom(true);

  // Thinking indicator
  const thinkRow = document.createElement("div");
  thinkRow.className = "msg-row bot";
  thinkRow.innerHTML = `<div class="msg-label">Aether</div>
    <div class="think"><div class="think-dots"><span></span><span></span><span></span></div><span>Thinking…</span></div>`;
  chatInner.appendChild(thinkRow);
  scrollBottom(true);

  // Build history for API
  const cfg = getCfg();
  const sess = getSession(currentId);
  const history = sess.messages.slice(0,-1).map(m => ({ role: m.role, content: m.content }));

  // Temp map
  const styleTemp = { concise:0.3, balanced:0.5, creative:0.85 };
  const temp = cfg.style !== "balanced" ? styleTemp[cfg.style] : cfg.temperature;

  try {
    const res = await fetch("/api/chat", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ message:text, history, temperature:temp, max_tokens:cfg.maxTokens })
    });
    const json = await res.json();
    thinkRow.remove();
    const reply = json.reply || json.error || "Something went wrong.";
    appendBubble("bot", reply);

    // Save reply
    const d2 = getStore();
    d2.sessions[currentId].messages.push({ role:"assistant", content:reply });
    save(d2);
    renderHistory();
  } catch(e) {
    thinkRow.remove();
    appendBubble("bot", "Connection error. Please try again.");
  }

  busy = false;
  sendBtn.disabled = false;
  scrollBottom(true);
}

// ══════════════════════════════════════════
//  SETTINGS
// ══════════════════════════════════════════
function openSettings() { document.getElementById("soverlay").classList.add("show"); applyAll(); }
function closeSettings() { document.getElementById("soverlay").classList.remove("show"); }

document.getElementById("settings-btn").addEventListener("click", openSettings);
document.getElementById("top-settings").addEventListener("click", openSettings);
document.getElementById("sp-close").addEventListener("click", closeSettings);
document.getElementById("soverlay").addEventListener("click", e => { if(e.target.id==="soverlay") closeSettings(); });

document.getElementById("tog-dark").addEventListener("change", e => {
  setCfg({ dark: e.target.checked }); applyTheme(e.target.checked);
});
document.querySelectorAll("#seg-font button").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll("#seg-font button").forEach(x=>x.classList.remove("on"));
  b.classList.add("on"); const sz=+b.dataset.v; setCfg({fontSize:sz}); applyFont(sz);
}));
document.querySelectorAll("#seg-style button").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll("#seg-style button").forEach(x=>x.classList.remove("on"));
  b.classList.add("on"); setCfg({style:b.dataset.v});
}));
document.querySelectorAll("#seg-tok button").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll("#seg-tok button").forEach(x=>x.classList.remove("on"));
  b.classList.add("on"); setCfg({maxTokens:+b.dataset.v});
}));
document.getElementById("temp-range").addEventListener("input", e => {
  const v=parseFloat(e.target.value); setCfg({temperature:v});
  document.getElementById("temp-val").textContent=v.toFixed(2);
});

document.getElementById("clear-all-btn").addEventListener("click", () => {
  if (confirm("Delete ALL conversations? This cannot be undone.")) {
    localStorage.removeItem(STORE_KEY); newChat();
  }
});

// ══════════════════════════════════════════
//  SIDEBAR TOGGLES
// ══════════════════════════════════════════
function closeMobSidebar() { layout.classList.remove("mob-sb"); }
document.getElementById("sb-toggle-desk").addEventListener("click", () => {
  if (window.innerWidth <= 640) { closeMobSidebar(); }
  else { layout.classList.toggle("sb-off"); }
});
document.getElementById("mob-menu").addEventListener("click", () => {
  layout.classList.add("mob-sb");
});
document.getElementById("mob-bg").addEventListener("click", closeMobSidebar);
document.getElementById("main").addEventListener("click", e => {
  if (layout.classList.contains("mob-sb") && !e.target.closest("#sidebar")) {
    closeMobSidebar();
  }
});

// ══════════════════════════════════════════
//  TEXTAREA AUTO-RESIZE
// ══════════════════════════════════════════
function autoResize() {
  msgInput.style.height = "auto";
  msgInput.style.height = Math.min(msgInput.scrollHeight, 180) + "px";
}
msgInput.addEventListener("input", autoResize);
msgInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
sendBtn.addEventListener("click", send);

document.querySelectorAll(".chip").forEach(c =>
  c.addEventListener("click", () => {
    msgInput.value = c.textContent.replace(/^[✦\u2726\s]*/,"").trim();
    msgInput.dispatchEvent(new Event("input"));
    msgInput.focus();
  })
);

// ══════════════════════════════════════════
//  INIT
// ══════════════════════════════════════════
applyAll();
renderHistory();
if (currentId && getSession(currentId)) renderSession(currentId);
else showEmpty(true);
</script>
</body>
</html>"""

@app.route("/")
def index():
    return HTML, 200, {"Content-Type": "text/html; charset=utf-8"}

@app.route("/api/chat", methods=["POST"])
def chat():
    data         = request.get_json(force=True)
    user_message = data.get("message", "").strip()
    history      = data.get("history", [])
    temperature  = float(data.get("temperature", 0.5))
    max_tokens   = int(data.get("max_tokens", 4096))

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history[-6:]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.7,
        "stream": True,
    }

    try:
        sess = requests.Session()
        r = sess.post(INVOKE_URL, headers=headers, json=payload, stream=True, timeout=(20, 120))
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
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, threaded=True)
