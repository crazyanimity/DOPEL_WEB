# Dopel — full stack

## What changed from your two scripts
- Both your training scripts ("ye wala" TF-IDF/SGD, and the FAISS+Groq RAG one) are now wrapped
  behind a real API in `backend/ml_engine.py`, unchanged in their core logic.
- Sessions used to live in memory (`SESSIONS = {}`) and vanished on every restart. They're now
  persisted to disk under `backend/storage/<user_id>/` and tracked in Postgres, so a user's
  trained Dopel survives server restarts.
- The Groq API key is **only** ever read from the server's `.env` file (`GROQ_API_KEY`). It is
  never sent from, or accepted from, the browser. The old frontend had a password-style input for
  it — that field is gone entirely.
- Every training/chat request is tied to a logged-in user via JWT — nobody can train or chat as
  someone else, and nobody re-types their name (it comes from their account).

## Flow
1. `index.html` — landing page. "Get Started" opens a signup/login modal (calls
   `POST /api/auth/signup` or `/api/auth/login`), same modal either way.
2. `plans.html` — after login, pick Quick or Smart Mode.
3. `train.html` — upload any number of `.txt` WhatsApp exports → `POST /api/train` kicks off
   training in the background → frontend polls `GET /api/persona/{plan}/status` every 2s →
   once `status == "ready"`, the chat panel below unlocks (scroll down to it) →
   `POST /api/chat` talks to your trained Dopel.

## Backend setup
```bash
cd backend
cp .env.example .env        # fill in real values — see comments in the file
pip install -r requirements.txt

# create the Postgres tables (run once)
python -c "from database import Base, engine; import models; Base.metadata.create_all(engine)"

uvicorn main:app --reload --port 8000
```

You need a real Postgres database reachable at the `DATABASE_URL` you put in `.env`. Locally,
the fastest way is Docker:
```bash
docker run --name dopel-db -e POSTGRES_USER=dopel_user -e POSTGRES_PASSWORD=CHANGE_ME \
  -e POSTGRES_DB=dopel_db -p 5432:5432 -d postgres:16
```

## Frontend
Any static file server works (VS Code Live Server, `python -m http.server`, etc.) — open
`index.html` through it, not by double-clicking, since it makes real `fetch()` calls to the
backend and some browsers block those from a bare `file://` page.

If your backend isn't on `localhost:8000`, update `API_BASE` in `frontend/shared.js`.

## Security measures included
- **Passwords**: bcrypt-hashed (`passlib`), never stored or logged in plain text.
- **Sessions**: JWT, signed with a server-only secret, expires after `JWT_EXPIRE_MINUTES`
  (default 24h). Every protected endpoint verifies it via `get_current_user` — the frontend
  can't just claim to be any user.
- **Groq key**: server-side env var only, never touches the client or the database.
- **Encryption at rest**: the retained message pairs used for Smart Mode retrieval are encrypted
  with Fernet (`ENCRYPTION_KEY`) before being written to disk.
- **Rate limiting** (`slowapi`): signup 5/min, login 10/min (brute-force protection), training
  10/hour, chat 30/min (keeps Groq API costs bounded per user).
- **CORS**: locked to the origins you list in `CORS_ORIGINS`, not `*`.
- Login and signup return the same generic error message ("Incorrect email or password") so an
  attacker can't use error differences to enumerate which emails are registered.

## What's intentionally NOT included (be aware before going to real production)
- **Model artifacts on local disk** (`backend/storage/`) — fine for one server. If you deploy to
  multiple instances/containers, move this to shared object storage (S3, etc.) since each
  instance would otherwise have its own disk.
- **Email verification / password reset** — signup currently trusts any email address as given.
- **HTTPS** — this all assumes you put it behind a reverse proxy (nginx/Caddy) or platform
  (Render/Railway/Fly.io) that terminates TLS. Never run this over plain HTTP in production.
- **Groq spend caps** — the rate limit slows abuse but doesn't hard-cap total API spend; set a
  budget alert in the Groq console too.
