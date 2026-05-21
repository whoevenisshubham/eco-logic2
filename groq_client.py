import os
import json
import requests
from typing import Optional


def get_groq_api_key() -> Optional[str]:
    return os.getenv("GROQ_API_KEY")


def groq_generate(prompt: str, model: str = "llama-3.1-8b-instant", max_tokens: int = 1024) -> dict:
    """Call Groq Llama text generation endpoint using the API key in env `GROQ_API_KEY`.

    Returns the JSON response on success, raises RuntimeError on HTTP errors or missing key.
    """
    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment")

    # Use OpenAI-compatible endpoint for Groq
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=15)
    if resp.status_code != 200:
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        raise RuntimeError(f"Groq API error {resp.status_code}: {body}")

    return resp.json()


def generate_refactor(prompt: str, model: str = "llama-3.1-8b-instant", max_output_tokens: int = 1024) -> str:
    """Compatibility wrapper that returns raw text refactor similar to previous Gemini helper.

    It extracts the assistant message text from the OpenAI-compatible chat response.
    """
    resp = groq_generate(prompt, model=model, max_tokens=max_output_tokens)
    # OpenAI-compatible chat response
    try:
        choices = resp.get("choices", [])
        if choices:
            first = choices[0]
            # Chat-style
            msg = first.get("message") or first.get("delta") or {}
            if isinstance(msg, dict):
                content = msg.get("content") or msg.get("text")
                if isinstance(content, str) and content:
                    return content
            # Completion-style
            text = first.get("text")
            if isinstance(text, str) and text:
                return text
    except Exception:
        pass

    # Fallback: stringify body
    return json.dumps(resp)


def generate_code_explanation(
    original_code: str,
    refactored_code: str,
    language: str = "C++",
    objective: str = "improve runtime and energy efficiency",
    metrics_summary: str = "",
    shap_summary: str = "",
    model: str = "llama-3.1-8b-instant",
    max_output_tokens: int = 700,
) -> str:
    """Use Groq to explain how the refactor improved the code.

    The output should describe the algorithmic change, expected complexity impact,
    and the observed/predicted runtime or energy deltas.
    """

    prompt = (
        "You are a senior performance engineer and code reviewer. Explain how the refactored code improved the original code. "
        "Be concrete and technical, but write for a developer who wants to understand the change quickly. "
        "Do not generate code. Do not mention that you are an AI. "
        "Use plain language with short bullet points. "
        f"Target language: {language}. Objective: {objective}.\n\n"
        "Original code:\n```"
        f"{language.lower()}\n{original_code}\n```\n\n"
        "Refactored code:\n```"
        f"{language.lower()}\n{refactored_code}\n```\n\n"
        f"Metrics summary:\n{metrics_summary or 'No metrics provided.'}\n\n"
        f"SHAP summary:\n{shap_summary or 'No SHAP summary provided.'}\n\n"
        "Explain: what changed, why it is better, what tradeoffs remain, and which parts of the refactor most likely drove the improvement."
    )
    return generate_refactor(prompt, model=model, max_output_tokens=max_output_tokens)


if __name__ == "__main__":
    import load_env
    load_env.load()
    key = get_groq_api_key()
    print("GROQ key present:", bool(key))
    if key:
        sample_prompt = (
            "Refactor the following Python function to be more readable and efficient:\n" +
            "def add_all(a):\n    s=0\n    for i in a:\n        s+=i\n    return s\n"
        )
        try:
            # Note: default model updated to llama-3.1-8b-instant
            out = groq_generate(sample_prompt, model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
            print(json.dumps(out, indent=2))
        except Exception as e:
            print("Groq request failed:", e)
