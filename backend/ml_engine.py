from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

# --- REMOVE these lines from your file ---
#   import faiss  (both occurrences, inside train_smart and reply_smart)
#   _embedder = None
#   def _get_embedder(): ...

def train_smart(user_id: str, combined_df: pd.DataFrame) -> None:
    d = user_dir(user_id)

    vectorizer = TfidfVectorizer(max_features=5000)
    matrix = vectorizer.fit_transform(combined_df["input"].tolist())  # sparse, no torch needed

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
    top_idxs = sims.argsort()[-5:][::-1]  # top 5, highest similarity first

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