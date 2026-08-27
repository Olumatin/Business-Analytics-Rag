from groq import Groq

from app import config
from app.system_prompt import SYSTEM_PROMPT

_client = None


def get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def build_context_block(chunks):
    if not chunks:
        return "No relevant context was found in the knowledge base."
    parts = []
    for c in chunks:
        parts.append(f"[Source: {c['source']} | chunk {c['chunk_id']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(question, chunks):
    context = build_context_block(chunks)
    client = get_client()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    completion = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=messages,
        temperature=0.2,
    )
    # Picks the message string accurately from Pydantic structural components
    return completion.choices[0].message.content
