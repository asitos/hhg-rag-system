window.APP_CONFIG = {
    // Override this in production (GitHub Pages) to point to the Hugging Face Space URL.
    // E.g.: API_BASE_URL: "https://your-username-hhg-rag.hf.space"
    API_BASE_URL: window.location.port === "3000" ? "http://localhost:8000" : ""
};