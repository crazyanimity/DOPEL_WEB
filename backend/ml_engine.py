"""
Core training + inference logic for both plans.
  Quick Mode  = TF-IDF + SGDClassifier (fast, fully local, no external API) — one merged model.
  Smart Mode  = TF-IDF cosine-similarity retrieval + Groq generation, over ONE merged index
                across every uploaded file (not per-contact — per_contact is only kept around
                to build a "trained on: X, Y, Z" label for the UI).

Model artifacts are written to disk under storage/<user_id>/ so they survive a server restart.
The Groq API key is read once from the server environment and is never accepted from a client.
"""
import os
import re
import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import pandas as pd
from fastapi import HTTPException

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

from security import encrypt_bytes, decrypt_bytes

STORAGE_ROOT = Path(__file__).parent / "storage"
STORAGE_ROOT.mkdir(exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# AI-disclosure guardrail. Only fires when the person directly asks —
# never volunteered unprompted. Checked in main.py's /api/chat handler
# before either reply_quick or reply_smart is called.
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


# WhatsApp's own export filename format: "WhatsApp Chat with <Name>.txt"
FILENAME_CONTACT_PATTERN = re.compile(r'whatsapp chat with (.+)', re.IGNORECASE)


def infer_contact_name(filename: str) -> str:
    """Pulls the contact's name straight out of WhatsApp's default export filename.
    Falls back to the filename itself if it doesn't match that pattern."""
    stem = re.sub(r'\.txt$', '', filename, flags=re.IGNORECASE).strip()
    match = FILENAME_CONTACT_PATTERN.match(stem)
    name = match.group(1).strip() if match else stem
    name = re.sub(r'[^\w\s.\'-]+$', '', name).strip()
    return name or "Unknown contact"


def safe_key(name: str) -> str:
    """Filesystem-safe key derived from a contact name, for use in artifact filenames."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)[:60] or "contact"


# ---------------------------------------------------------------------------
# Parsing + cleaning
# ---------------------------------------------------------------------------
PATTERN = r'^(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2})\s?(AM|PM|am|pm)? - ([^:]+): (.+)$'


def parse_chat_text(raw_text: str) -> pd.DataFrame:
    normalized = (
        raw_text
        .replace('\u202f', ' ')
        .replace('\xa0', ' ')
        .replace('\u200e', '')
        .replace('\u200f', '')
    )
    messages = []
    for line in normalized.split('\n'):
        match = re.match(PATTERN, line)
        if match:
            date, time, ampm, sender, message = match.groups()
            full_time = f"{time} {ampm}" if ampm else time
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
        (pairs_df["input"] != "") & (pairs_df["output"] != "") &
        (~pairs_df["output"].isin(BORING_RESPONSES))
    ]
    return pairs_df.reset_index(drop=True)


def parse_and_pair_uploads(
    files: List[Tuple[str, bytes]], your_name: str
) -> Tuple[pd.DataFrame, "dict[str, pd.DataFrame]"]:
    per_contact: "dict[str, pd.DataFrame]" = {}

    for filename, raw_bytes in files:
        raw = raw_bytes.decode("utf-8", errors="replace")
        df = parse_chat_text(raw)
        if df.empty:
            continue
        pairs_df = build_pairs(df, your_name)
        if pairs_df.empty:
            continue

        contact = infer_contact_name(filename)
        if contact in per_contact:
            per_contact[contact] = pd.concat([per_contact[contact], pairs_df], ignore_index=True)
        else:
            per_contact[contact] = pairs_df

    if not per_contact:
        raise HTTPException(400,
            "None of the uploaded files matched the expected WhatsApp export format, "
            "or none contained any messages from you. Make sure they're the raw .txt exports "
            "and that your account name matches the sender name in the file exactly.")

    combined_df = pd.concat(per_contact.values(), ignore_index=True)
    if len(combined_df) < 10:
        raise HTTPException(400,
            f"Only found {len(combined_df)} usable message pairs for '{your_name}' across all files "
            "— need at least 10. Double check your account name matches exactly how it appears "
            "in the chat export (open the .txt file and look for your name before the colon).")

    return combined_df, per_contact


def user_dir(user_id: str) -> Path:
    d = STORAGE_ROOT / user_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def train_quick(user_id: str, pairs_df: pd.DataFrame) -> None:
    train_df, _ = train_test_split(pairs_df, test_size=0.2, random_state=42) if len(pairs_df) >= 5 else (pairs_df, None)
    vectorizer = TfidfVectorizer(max_features=3000)
    model = make_pipeline(vectorizer, SGDClassifier(loss="log_loss", max_iter=1000))
    model.fit(train_df["input"], train_df["output"])
    joblib.dump(model, user_dir(user_id) / "quick_model.joblib")


def reply_quick(user_id: str, message: str) -> str:
    path = user_dir(user_id) / "quick_model.joblib"
    if not path.exists():
        raise HTTPException(404, "Quick Mode model not found. Upload your chats to train it first.")
    model = joblib.load(path)
    return model.predict([clean_text(message)])[0]


def train_smart(user_id: str, combined_df: pd.DataFrame) -> None:
    d = user_dir(user_id)

    vectorizer = TfidfVectorizer(max_features=5000)
    matrix = vectorizer.fit_transform(combined_df["input"].tolist())

    joblib.dump(vectorizer, d / "smart_vectorizer.joblib")
    sparse.save_npz(d / "smart_matrix.npz", matrix)

    encrypted = encrypt_bytes(pickle.dumps(combined_df))
    (d / "smart_pairs.enc").write_bytes(encrypted)


def reply_smart(user_id: str, your_name: str, message: str) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(500, "Smart Mode is not configured on the server (missing GROQ_API_KEY).")

    from groq import Groq

    d = user_dir(user_id)
    vec_path, matrix_path, pairs_path = d / "smart_vectorizer.joblib", d / "smart_matrix.npz", d / "smart_pairs.enc"
    if not vec_path.exists() or not matrix_path.exists() or not pairs_path.exists():
        raise HTTPException(404, "Smart Mode model not found. Upload your chats to train it first.")

    vectorizer = joblib.load(vec_path)
    matrix = sparse.load_npz(matrix_path)
    pairs_df = pickle.loads(decrypt_bytes(pairs_path.read_bytes()))

    cleaned = clean_text(message)
    q_vec = vectorizer.transform([cleaned])
    sims = cosine_similarity(q_vec, matrix)[0]
    top_idxs = sims.argsort()[-5:][::-1]

    examples = [(pairs_df.iloc[i]["input"], pairs_df.iloc[i]["output"]) for i in top_idxs]
    example_block = "\n".join(f'They said: "{i}"\nYou replied: "{o}"' for i, o in examples)

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

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.8,
        max_tokens=60,
    )
    return response.choices[0].message.content.strip()