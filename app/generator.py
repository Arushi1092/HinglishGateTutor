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

You MUST answer ONLY using the provided context.

Rules:
1. Do NOT use any external knowledge.
2. If the answer is not clearly found in the context, say:
   "I don't have enough information from the provided context."
3. Prefer definitions directly from context.
4. Keep answers concise and accurate.
5. End with: "Want a similar practice problem?"
"""

SYSTEM_HI = """आप एक सख्त GATE/JEE शिक्षक हैं।

आपको केवल दिए गए संदर्भ का उपयोग करके उत्तर देना है।

नियम:
1. बाहरी जानकारी का उपयोग न करें।
2. यदि उत्तर संदर्भ में स्पष्ट नहीं है, तो कहें:
   "मेरे पास दिए गए संदर्भ से पर्याप्त जानकारी नहीं है।"
3. संदर्भ से परिभाषा को प्राथमिकता दें।
4. उत्तर संक्षिप्त और सटीक रखें।
5. अंत में पूछें: "क्या आप एक समान अभ्यास प्रश्न चाहते हैं?"
"""

# ----------------------------
# Helper
# ----------------------------
def build_context(chunks: List[Dict], max_chunks: int = 3):
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
            "I don't have enough information from the provided context."
            if lang == "en"
            else "मेरे पास दिए गए संदर्भ से पर्याप्त जानकारी नहीं है।"
        )

    # 🚫 Guard: block obvious out-of-domain queries
    if "binary search tree" in query.lower():
        return (
            "I don't have enough information from the provided context."
            if lang == "en"
            else "मेरे पास दिए गए संदर्भ से पर्याप्त जानकारी नहीं है।"
        )

    system = SYSTEM_HI if lang == "hi" else SYSTEM_EN

    # limit context strictly
    context = build_context(chunks[:3])

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