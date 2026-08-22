window.APP_CONFIG = {
    // Override this in production to point to the Hugging Face Space URL.
    // When left empty or unreachable, the frontend will automatically fallback to STATIC DEMO MODE.
    API_BASE_URL: window.location.port === "3000" || window.location.port === "8000" ? window.location.origin : ""
};
