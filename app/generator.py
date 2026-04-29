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
SYSTEM_EN = """You are an expert GATE/JEE tutor specializing in technical and numerical problem solving.

Your task is to answer the student's question based ONLY on the provided context.

Rules:
1. Answer ONLY based on the context provided.
2. For Numerical/Mathematical questions:
   - Identify the given values and the concept from the context.
   - Solve the problem STEP-BY-STEP using "Chain of Thought" reasoning.
   - Clearly state the formula used.
3. If the context does not contain the specific information (like a specific year's question), do NOT hallucinate. Say "The provided documents do not contain the specific questions for [Year]."
4. Be precise, technical, and accurate.
5. MANDATORY: You must end every response with: "Want a similar practice problem?"
"""

SYSTEM_HI = """आप एक मददगार और सख्त GATE/JEE शिक्षक हैं।

आपका कार्य केवल दिए गए संदर्भ के आधार पर छात्र के प्रश्न का उत्तर देना है।

नियम:
1. केवल दिए गए संदर्भ के आधार पर उत्तर दें। बाहरी ज्ञान का उपयोग न करें। संदर्भ में नहीं दी गई जानकारी न जोड़ें।
2. यदि छात्र कोई अनुवर्ती प्रश्न पूछता है (जैसे "उदाहरण के साथ समझाएं"), तो यह समझने के लिए "पिछली बातचीत" देखें कि वे किस बारे में बात कर रहे हैं।
3. यदि संदर्भ किसी स्थिति, अनुप्रयोग या प्रक्रिया (जैसे "सॉर्टिंग" या "रूटिंग") का वर्णन करता है, तो उसे उदाहरण के रूप में उपयोग करें।
4. यदि जानकारी वास्तव में गायब है, तो कहें: "मेरे पास दिए गए संदर्भ से पर्याप्त जानकारी नहीं है।"
5. संक्षिप्त और पेशेवर रहें।
6. अनिवार्य: आपको हर उत्तर के अंत में यह सटीक वाक्यांश लिखना होगा: "क्या आप एक समान अभ्यास प्रश्न चाहते हैं?"
"""

REWRITE_PROMPT = """Given the conversation history and a follow-up question, rewrite it into a detailed standalone search query.
Focus on the technical concept being discussed.

History:
{history}

Follow-up Question: {query}
Standalone Search Query:"""

# ----------------------------
# Helpers
# ----------------------------
def rewrite_query(query: str, history: List[Dict]) -> str:
    if not history:
        return query

    # Use the last 3 turns for context
    history_text = ""
    for h in history[-3:]:
        history_text += f"Q: {h.get('q','')}\nA: {h.get('a','')}\n"
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(history=history_text, query=query)}],
            max_tokens=100,
            temperature=0
        )
        rewritten = response.choices[0].message.content.strip().replace('"', '')
        return rewritten if rewritten else query
    except:
        return query

def build_context(chunks: List[Dict], max_chunks: int = 10):
    chunks = chunks[:max_chunks]
    return "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{' '.join(c['text'].split())}"
        for c in chunks
    )

# ----------------------------
# Main function
# ----------------------------
def generate_answer(query: str, chunks: List[Dict], lang: str = "en", history: List[Dict] = None):
    if history is None:
        history = []

    # Fallback for no context
    if not chunks:
        return (
            "I don't have enough information from the provided context.\n\nWant a similar practice problem?"
            if lang == "en"
            else "मेरे पास दिए गए संदर्भ से पर्याप्त जानकारी नहीं है।\n\nक्या आप एक समान अभ्यास प्रश्न चाहते हैं?"
        )

    system = SYSTEM_HI if lang == "hi" else SYSTEM_EN

    # Use up to 10 chunks for rich context
    context = build_context(chunks, max_chunks=10)

    # 🧠 Build history string
    history_str = ""
    if history:
        for h in history[-3:]:
            history_str += f"Student: {h.get('q','')}\nTutor: {h.get('a','')}\n"

    # 🔥 Structured prompt
    user_prompt = f"### Context:\n{context}\n\n"
    if history_str:
        user_prompt += f"### Previous Conversation:\n{history_str}\n\n"
    user_prompt += f"### Current Question: {query}"

    try:
        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            max_tokens=500,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
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
