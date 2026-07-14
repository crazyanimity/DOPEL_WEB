// Shared across index.html, plans.html, train.html.
// Override via frontend/config.js or by setting window.DOPL_CONFIG.API_BASE.
window.DOPL_CONFIG = window.DOPL_CONFIG || {};

// Default API base: prefer an explicit override in frontend/config.js (window.DOPL_CONFIG.API_BASE).
// When not overridden, choose a sensible default for local dev vs production.
const _defaultApiBase = (() => {
    try {
        const host = window.location && window.location.hostname;
        const protocol = window.location && window.location.protocol;
        if (!host) return "https://dopel-web.onrender.com/api";
        // If running locally (localhost, 127.0.0.1) or opened via file://, point to the local backend.
        if (host === "localhost" || host === "127.0.0.1" || protocol === "file:") {
            return "http://127.0.0.1:8000/api";
        }
    } catch (e) {
        /* fall through to production default */
    }
    return "https://dopel-web.onrender.com/api";
})();

window.DOPL_CONFIG.API_BASE = window.DOPL_CONFIG.API_BASE || _defaultApiBase;

// Shared across index.html, plans.html, train.html.
// Override via frontend/config.js or by setting window.DOPL_CONFIG.API_BASE.
const API_BASE = (window.DOPL_CONFIG && window.DOPL_CONFIG.API_BASE) || _defaultApiBase;

function saveSession(data) {
    localStorage.setItem("dopel_token", data.access_token);
    localStorage.setItem("dopel_name", data.name);
    localStorage.setItem("dopel_email", data.email);
}

function getToken() {
    return localStorage.getItem("dopel_token");
}

function getUserName() {
    return localStorage.getItem("dopel_name") || "";
}

function clearSession() {
    localStorage.removeItem("dopel_token");
    localStorage.removeItem("dopel_name");
    localStorage.removeItem("dopel_email");
}

// Call at the top of any page that requires a logged-in user.
// Sends them back to the landing page (with the auth modal) if not logged in.
function requireAuth() {
    if (!getToken()) {
        window.location.href = "index.html?login=1";
        return false;
    }
    return true;
}

// Wrapper around fetch() that attaches the auth header and
// handles an expired/invalid session in one place.
async function authFetch(path, options = {}) {
    const headers = options.headers || {};
    headers["Authorization"] = `Bearer ${getToken()}`;
    const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
    if (res.status === 401) {
        clearSession();
        window.location.href = "index.html?login=1&expired=1";
        throw new Error("Session expired");
    }
    return res;
}
