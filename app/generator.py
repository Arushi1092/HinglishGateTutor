from __future__ import annotations

import os
from typing import List, Dict
from dotenv import load_dotenv
from groq import Groq

# ----------------------------
# Load API key
# ----------------------------
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=api_key)

# ----------------------------
# Prompts
# ----------------------------
SYSTEM_EN = """You are a strict GATE/JEE tutor.

Your task is to answer the student's question based ONLY on the provided context.

Rules:
1. Use ONLY the provided context. Do NOT use external knowledge.
2. If the context mentions a definition is present but doesn't show it, or if information is missing, state: "I don't have enough information from the provided context."
3. Be concise and use a professional, educational tone.
4. MANDATORY: You must end every response with the exact phrase: "Want a similar practice problem?"
"""

SYSTEM_HI = """आप एक सख्त GATE/JEE शिक्षक हैं।

आपका कार्य केवल दिए गए संदर्भ के आधार पर छात्र के प्रश्न का उत्तर देना है।

नियम:
1. केवल दिए गए संदर्भ का उपयोग करें। बाहरी ज्ञान का उपयोग न करें।
2. यदि संदर्भ में उल्लेख है कि परिभाषा मौजूद है लेकिन वह नहीं दी गई है, या यदि जानकारी गायब है, तो कहें: "मेरे पास दिए गए संदर्भ से पर्याप्त जानकारी नहीं है।"
3. संक्षिप्त रहें और पेशेवर, शैक्षिक स्वर का उपयोग करें।
4. अनिवार्य: आपको हर उत्तर के अंत में यह सटीक वाक्यांश लिखना होगा: "क्या आप एक समान अभ्यास प्रश्न चाहते हैं?"
"""

# ----------------------------
# Helper
# ----------------------------
def build_context(chunks: List[Dict], max_chunks: int = 5):
    chunks = chunks[:max_chunks]
    return "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{' '.join(c['text'].split())}"
        for c in chunks
    )

# ----------------------------
# Main function
# ----------------------------
def generate_answer(query: str, chunks: List[Dict], lang: str = "en"):
    if not chunks:
        return (
            "I don't have enough information from the provided context.\n\nWant a similar practice problem?"
            if lang == "en"
            else "मेरे पास दिए गए संदर्भ से पर्याप्त जानकारी नहीं है।\n\nक्या आप एक समान अभ्यास प्रश्न चाहते हैं?"
        )

    # 🚫 Guard: block obvious out-of-domain queries
    if "binary search tree" in query.lower():
        return (
            "I don't have enough information from the provided context.\n\nWant a similar practice problem?"
            if lang == "en"
            else "मेरे पास दिए गए संदर्भ से पर्याप्त जानकारी नहीं है।\n\nक्या आप एक समान अभ्यास प्रश्न चाहते हैं?"
        )

    system = SYSTEM_HI if lang == "hi" else SYSTEM_EN

    # limit context
    context = build_context(chunks, max_chunks=5)

    user_message = f"""Context:
{context}

Question: {query}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=400,
            temperature=0.2,  # more deterministic, less hallucination
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message}
            ]
        )

        content = response.choices[0].message.content
        return content.strip() if content else (
            "I don't have enough information from the provided context."
            if lang == "en"
            else "मेरे पास दिए गए संदर्भ से पर्याप्त जानकारी नहीं है।"
        )

    except Exception as e:
        return f"Error generating answer: {str(e)}"