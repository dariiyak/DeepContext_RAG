import os
import requests
import uuid

OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

DEFAULT_SYSTEM_PROMPT = (
    "Ты — ассистент RAG-системы. "
    "Отвечай только на основе переданного контекста. "
    "Если информации недостаточно, честно скажи об этом. "
    "Не придумывай факты и не добавляй сведения, которых нет в контексте. "
    "Если фрагменты противоречат друг другу, укажи на это. "
    "Отвечай по-русски, кратко, структурированно и логично."
)

def get_access_token():
    auth_key = os.getenv("GIGACHAT_AUTH_KEY")
    if not auth_key:
        raise RuntimeError("Нет GIGACHAT_AUTH_KEY в .env.")
    
    scope = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': str(uuid.uuid4()),
        'Authorization': auth_key
}
    data = {"scope": scope}
    resp = requests.post(OAUTH_URL, headers=headers, data=data, timeout=30, verify=False)
    resp.raise_for_status()

    return resp.json()["access_token"]
    
    

def answer_with_gigachat(question, context):
    token = get_access_token()

    model = os.getenv("GIGACHAT_MODEL", "GigaChat-2")

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": f"КОНТЕКСТ:\n{context}\n\nВОПРОС:\n{question}"},
        ],
        "temperature": 0.2
    }

    resp = requests.post(CHAT_URL, headers=headers, json=payload, timeout=60, verify=False)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()
