"""
Dopel backend v2
-----------------
Changes from v1:
  1. Signup now requires email OTP verification (via Gmail SMTP).
  2. Training is tied to your ACCOUNT, not a random session id — so
     retraining always overwrites the old model and starts fresh,
     even across logout/login.
  3. Smart Mode now builds ONE merged index across every uploaded file,
     regardless of which contact it came from — no more per-contact
     dropdown. The RAG retrieves from your whole message history.
  4. A guardrail: if someone asks "are you an AI / are you real / is
     this a bot", the bot always discloses honestly. It never brings
     this up unprompted.
  5. A public share link per account (/api/share/<token>/chat) so you
     can send a no-login chat link to friends — the equivalent of
     Gradio's `share=True`, but self-hosted (see NOTE ON PUBLIC LINKS
     at the bottom for how to actually expose it to the internet).

Run it:
    pip install -r dopel_requirements.txt
    export GMAIL_ADDRESS="youraddress@gmail.com"
    export GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"   # see setup note below
    uvicorn dopel_server:app --reload --port 8000

If you don't set the two GMAIL_* env vars, OTPs are just printed to the
terminal instead of emailed — handy for testing without setting up SMTP yet.

GMAIL SETUP NOTE: a normal Gmail password will NOT work here. You need:
  1. Turn on 2-Step Verification on the Google account: myaccount.google.com/security
  2. Go to myaccount.google.com/apppasswords
  3. Generate an "app password" (16 characters) — use THAT as GMAIL_APP_PASSWORD
"""

import os
import re
import ssl
import time
import uuid
import random
import smtplib
from email.mime.text import MIMEText
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

app = FastAPI(title="Dopel API v2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# In-memory stores. Fine for a thesis demo / single machine. Swap for a real
# database (Postgres, etc.) plus hashed passwords (bcrypt) before real users
# ever touch this — plaintext passwords in a dict is NOT production-safe.
# ---------------------------------------------------------------------------
USERS: dict = {}          # email -> {name, password, model, share_token}
PENDING_SIGNUPS: dict = {}  # email -> {otp, expiry, name, password}
TOKENS: dict = {}         # bearer token -> email
SHARE_INDEX: dict = {}    # share_token -> email


# ---------------------------------------------------------------------------
# 1. OTP EMAIL VERIFICATION
# ---------------------------------------------------------------------------
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")


def send_otp_email(to_email: str, otp: str):
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        # Dev fallback so you can test the flow before SMTP is configured.
        print(f"[DEV MODE — no email sent] OTP for {to_email}: {otp}")
        return
    msg = MIMEText(f"Your Dopel verification code is {otp}. It expires in 10 minutes.")
    msg["Subject"] = "Your Dopel verification code"
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = to_email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, to_email, msg.as_string())


class SignupRequest(BaseModel):
    name: str
    email: str
    password: str


class OtpVerifyRequest(BaseModel):
    email: str
    otp: str


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/api/auth/signup")
def signup(req: SignupRequest):
    email = req.email.lower().strip()
    if email in USERS:
        raise HTTPException(400, "An account with this email already exists. Try logging in instead.")

    otp = f"{random.randint(0, 999999):06d}"
    PENDING_SIGNUPS[email] = {
        "otp": otp,
        "expiry": time.time() + 600,  # 10 minutes
        "name": req.name.strip(),
        "password": req.password,  # TODO: hash with bcrypt before production use
    }
    send_otp_email(email, otp)
    return {"message": "Verification code sent. Check your email."}


@app.post("/api/auth/verify-otp")
def verify_otp(req: OtpVerifyRequest):
    email = req.email.lower().strip()
    pending = PENDING_SIGNUPS.get(email)
    if not pending:
        raise HTTPException(400, "No pending signup for this email — sign up again.")
    if time.time() > pending["expiry"]:
        del PENDING_SIGNUPS[email]
        raise HTTPException(400, "That code expired. Sign up again for a new one.")
    if req.otp.strip() != pending["otp"]:
        raise HTTPException(400, "Incorrect code.")

    USERS[email] = {
        "name": pending["name"],
        "password": pending["password"],
        "model": None,        # populated once training runs
        "share_token": None,  # generated on first successful training
    }
    del PENDING_SIGNUPS[email]

    token = str(uuid.uuid4())
    TOKENS[token] = email
    return {"token": token, "name": USERS[email]["name"]}


@app.post("/api/auth/login")
def login(req: LoginRequest):
    email = req.email.lower().strip()
    user = USERS.get(email)
    if not user or user["password"] != req.password:
        raise HTTPException(401, "Incorrect email or password.")
    token = str(uuid.uuid4())
    TOKENS[token] = email
    return {"token": token, "name": user["name"]}


def get_current_user(authorization: str = Header(None)) -> str:
    """Every protected endpoint below requires: Authorization: Bearer <token>"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header.")
    token = authorization.split(" ", 1)[1]
    email = TOKENS.get(token)
    if not email or email not in USERS:
        raise HTTPException(401, "Session expired — please log in again.")
    return email


# ---------------------------------------------------------------------------
# Parsing + cleaning — unchanged from your original script.
# ---------------------------------------------------------------------------
PATTERN = r'^(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2})\s?(AM|PM|am|pm)? - ([^:]+): (.+)$'


def parse_chat_text(raw_text: str) -> pd.DataFrame:
    normalized = raw_text.replace('\u202f', ' ').replace('\xa0', ' ')
    messages = []
    for line in normalized.split('\n'):
        match = re.match(PATTERN, line)
        if match:
            date, time_, ampm, sender, message = match.groups()
            full_time = f"{time_} {ampm}" if ampm else time_
            messages.append({
                "date": date, "time": full_time.strip(),
                "sender": sender.strip(), "message": message.strip(),
            })
    return pd.DataFrame(messages)


def normalize_common_phrases(text: str) -> str:
    reply_map = {
        r'\by+a+\b': 'ya', r'\bok+(\s+)?(bro|dude)?\b': 'ok',
        r'\b(t+h+i+k+|t+h+e+e+k+|t+i+k+)\s*h*a*i*\b': 'thik hai',
        r'\bo+(\s*)k+\b': 'ok', r'\bo+i+\b': 'oi', r'\by+a+r+\b': 'yaar',
        r'\bh+u+m+\b': 'hum', r'\b(h+a+h+a+)+\b': 'haha',
    }
    for pattern, replacement in reply_map.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def normalize_repeats(text: str) -> str:
    return re.sub(r'(.)\1{2,}', r'\1\1', text)


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = normalize_repeats(text)
    text = normalize_common_phrases(text)
    if text in ["null", "this message was deleted"]:
        return ""
    text = re.sub(r"<media omitted>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z0-9 ?!.,]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


BORING_RESPONSES = {"ok", "k", "ya", "ya ya", "hmm", "hmmm", "thik hai", "h", "huh", "hmmm hmm"}


def build_pairs(df: pd.DataFrame, your_name: str) -> pd.DataFrame:
    """
    other -> you pairs, combined across every uploaded file, IGNORING which
    contact each message came from. This is the "merged, not per-contact"
    behavior — every file just adds more rows to one shared training set.
    """
    pairs = []
    for i in range(1, len(df)):
        if df.iloc[i - 1]["sender"] != your_name and df.iloc[i]["sender"] == your_name:
            pairs.append({
                "input": clean_text(df.iloc[i - 1]["message"]),
                "output": clean_text(df.iloc[i]["message"]),
            })
    pairs_df = pd.DataFrame(pairs)
    if pairs_df.empty:
        return pairs_df
    pairs_df = pairs_df[
        (pairs_df["input"] != "") &
        (pairs_df["output"] != "") &
        (~pairs_df["output"].isin(BORING_RESPONSES))
    ]
    return pairs_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. AI-DISCLOSURE GUARDRAIL
# Only fires when the person actually asks. Never volunteered unprompted.
# ---------------------------------------------------------------------------
AI_IDENTITY_PATTERNS = [
    r"\bare you (an? )?(ai|bot|robot|chatbot)\b",
    r"\bis this (an? )?(ai|bot|robot)\b",
    r"\bam i (talking|chatting|texting) (to|with) (an? )?(ai|bot)\b",
    r"\bare you (really |actually )?(real|human)\b",
    r"\bru (an? )?(ai|bot)\b",
]


def is_asking_if_ai(message: str) -> bool:
    m = message.lower()
    return any(re.search(p, m) for p in AI_IDENTITY_PATTERNS)


def disclosure_reply(your_name: str) -> str:
    return f"Yeah — I'm Dopel, an AI trained on {your_name}'s texting style, chatting on their behalf right now."


# ---------------------------------------------------------------------------
# Core reply generation, shared by the private and public (share-link) chat
# endpoints, so a share link behaves identically to the logged-in chat.
# ---------------------------------------------------------------------------
def generate_reply_for_user(user: dict, message: str) -> str:
    if is_asking_if_ai(message):
        return disclosure_reply(user["name"])

    model_state = user["model"]
    if not model_state:
        raise HTTPException(400, "This Dopel hasn't been trained yet.")

    cleaned = clean_text(message)

    if model_state["plan"] == "quick":
        return model_state["model"].predict([cleaned])[0]

    # Smart mode: retrieve from the ONE merged index, then generate with Groq.
    from groq import Groq
    import faiss

    client = Groq(api_key=model_state["groq_api_key"])
    q_emb = model_state["embedder"].encode([cleaned], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    _, idxs = model_state["index"].search(q_emb, 5)

    examples = []
    for idx in idxs[0]:
        if idx == -1:
            continue
        row = model_state["pairs_df"].iloc[idx]
        examples.append((row["input"], row["output"]))

    example_block = "\n".join(f'They said: "{i}"\nYou replied: "{o}"' for i, o in examples)
    your_name = user["name"]

    system_prompt = (
        f"You are roleplaying as {your_name}, replying over WhatsApp. "
        "Match the tone, length, slang, and casualness shown in the example replies below. "
        "Do not sound like a formal assistant. Keep replies short like real texting. "
        "Only output the reply text, nothing else."
    )
    user_prompt = (
        f"Examples of how you've replied before:\n{example_block}\n\n"
        f'They just sent: "{message}"\nWrite your reply:'
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
        max_tokens=60,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# 2 & 3. TRAINING — always tied to the account, always overwrites.
# ---------------------------------------------------------------------------
@app.post("/api/train")
async def train(
    files: List[UploadFile] = File(...),
    plan: str = Form(...),
    groq_api_key: Optional[str] = Form(None),
    email: str = Depends(get_current_user),
):
    if plan not in ("quick", "smart"):
        raise HTTPException(400, "plan must be 'quick' or 'smart'")

    user = USERS[email]
    your_name = user["name"]

    all_dfs = []
    for f in files:
        raw = (await f.read()).decode("utf-8", errors="replace")
        df = parse_chat_text(raw)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        raise HTTPException(400, "None of the uploaded files matched the expected WhatsApp export format.")

    combined = pd.concat(all_dfs, ignore_index=True)
    pairs_df = build_pairs(combined, your_name)

    if len(pairs_df) < 10:
        raise HTTPException(
            400,
            f"Only found {len(pairs_df)} usable pairs for '{your_name}' — need at least 10. "
            "Check the name matches your sender name in the export exactly."
        )

    if plan == "quick":
        train_df, _ = train_test_split(pairs_df, test_size=0.2, random_state=42)
        vectorizer = TfidfVectorizer(max_features=3000)
        model = make_pipeline(vectorizer, SGDClassifier(loss="log_loss", max_iter=1000))
        model.fit(train_df["input"], train_df["output"])
        model_state = {"plan": "quick", "model": model}

    else:
        if not groq_api_key:
            raise HTTPException(400, "Smart Mode needs a Groq API key.")
        from sentence_transformers import SentenceTransformer
        import faiss

        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedder.encode(pairs_df["input"].tolist(), convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        model_state = {
            "plan": "smart", "embedder": embedder, "index": index,
            "pairs_df": pairs_df, "groq_api_key": groq_api_key,
        }

    # Overwrite whatever was trained before. Retraining always starts fresh —
    # there's no stale session id hanging around to confuse things.
    user["model"] = model_state

    if not user["share_token"]:
        share_token = str(uuid.uuid4())[:8]
        user["share_token"] = share_token
        SHARE_INDEX[share_token] = email

    return {
        "plan": plan,
        "pairs_trained": len(pairs_df),
        "your_name": your_name,
        "share_token": user["share_token"],
    }


class ChatMessage(BaseModel):
    message: str


@app.post("/api/chat")
async def chat(req: ChatMessage, email: str = Depends(get_current_user)):
    user = USERS[email]
    reply = generate_reply_for_user(user, req.message)
    return {"reply": reply}


# ---------------------------------------------------------------------------
# 5. PUBLIC SHARE LINK — no login required, so you can send this to friends.
# ---------------------------------------------------------------------------
@app.get("/api/share/{share_token}")
def share_info(share_token: str):
    email = SHARE_INDEX.get(share_token)
    if not email or not USERS[email]["model"]:
        raise HTTPException(404, "This Dopel link isn't available.")
    return {"name": USERS[email]["name"]}


class PublicChatMessage(BaseModel):
    message: str


@app.post("/api/share/{share_token}/chat")
def share_chat(share_token: str, req: PublicChatMessage):
    email = SHARE_INDEX.get(share_token)
    if not email or not USERS[email]["model"]:
        raise HTTPException(404, "This Dopel link isn't available.")
    reply = generate_reply_for_user(USERS[email], req.message)
    return {"reply": reply}


@app.get("/api/health")
def health():
    return {"status": "ok", "accounts": len(USERS)}


# ---------------------------------------------------------------------------
# NOTE ON PUBLIC LINKS (the Gradio share=True equivalent)
# ---------------------------------------------------------------------------
# Gradio's `share=True` works by tunneling your local server through Gradio's
# own reverse-proxy servers, so anyone gets a public URL with zero setup.
# FastAPI has no built-in equivalent, but you can get the same effect with a
# tunneling tool. Two easy options:
#
#   Option A — ngrok (most common):
#       pip install pyngrok
#       Add this near the bottom of this file, guarded so it only runs
#       when you launch directly (not on every --reload):
#
#           if __name__ == "__main__":
#               from pyngrok import ngrok
#               import uvicorn
#               public_url = ngrok.connect(8000)
#               print("Public URL:", public_url)
#               uvicorn.run(app, host="0.0.0.0", port=8000)
#
#       Then run:  python dopel_server.py   (instead of uvicorn directly)
#       The printed public_url + "/api/share/<token>" is what you'd wire
#       your frontend's "Share" button to build the link from.
#
#   Option B — Cloudflare Tunnel (no signup needed, more stable long-term):
#       brew install cloudflared
#       cloudflared tunnel --url http://localhost:8000
#       It prints a public https://....trycloudflare.com URL pointing at
#       your local server — use that as your public API base instead.
#
# For your actual thesis demo day, either works. For something that needs
# to stay up reliably (not tied to your laptop being on), deploy the backend
# to a free tier host like Render or Railway instead — that gives you a
# permanent public URL rather than a temporary tunnel.
