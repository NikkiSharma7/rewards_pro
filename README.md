# Proactive AI Financial Assistant

This version adds a **ChatGPT-like suggestion system** on top of your existing deterministic + Azure OpenAI logic.

### ✅ Features Added
- Context-aware follow-up suggestions after each query.
- Suggestions displayed as clickable buttons.
- Proactive greeting with example queries.

### How It Works
- Uses Azure OpenAI (if configured) to generate 3-4 relevant follow-up questions.
- Suggestions appear below the assistant's response.
- Clicking a suggestion auto-fills and reruns the query.

### Run
```bash
streamlit run app.py
```

Ensure `.env` contains your Azure OpenAI keys.
