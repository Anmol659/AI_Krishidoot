import os
from fastapi import FastAPI, Form, Response
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv

# --- Import your main EcoAdvisor function ---
from backend.EcoAdvisior.EcoAdvisior import get_ai_response 

load_dotenv()
app = FastAPI()

# --- User Session Memory ---
# In app.py

# ... (keep all your imports) ...

# --- User Session Memory ---
user_sessions = {}

# --- Keywords and Choices ---
GREETINGS = ["hi", "hello", "menu", "start", "help", "hey"]
LANGUAGES = {
    "1": "English", "english": "English",
    "2": "Hindi",   "hindi": "Hindi",
    "3": "Gujarati","gujarati": "Gujarati"
}
# Add translated messages
LANGUAGE_MESSAGES = {
    "English": "Language has been set to English. How can I help you today?",
    "Hindi": "भाषा हिंदी में सेट हो गई है। मैं आपकी क्या मदद कर सकता हूँ?",
    "Gujarati": "ભાષા ગુજરાતીમાં સેટ કરવામાં આવી છે. હું તમને કેવી રીતે મદદ કરી શકું?"
}
LANGUAGE_MENU = """Welcome to AI-Krishidoot! 🌱
Please choose your language by replying with a number or name:
1. English
2. हिन्दी (Hindi)
3. ગુજરાતી (Gujarati)"""

# In your app.py file

@app.post("/api/whatsapp")
async def handle_whatsapp_message(From: str = Form(...), Body: str = Form(...)):
    """
    Handles incoming WhatsApp messages. If it's a greeting, it ONLY shows the
    language menu. Otherwise, it calls the EcoAdvisor AI.
    """
    user_number = From
    user_message = Body.lower().strip()
    reply_text = ""

    # This 'if' block handles greetings FIRST.
    if user_message in GREETINGS or user_message == "change language":
        reply_text = LANGUAGE_MENU
    
    # This 'elif' block handles the language selection.
    elif user_message in LANGUAGES:
        chosen_lang = LANGUAGES[user_message]
        user_sessions[user_number] = chosen_lang
        reply_text = LANGUAGE_MESSAGES.get(chosen_lang, LANGUAGE_MESSAGES["English"])
    
    # The AI is ONLY called in this 'else' block for all other queries.
    else:
        chosen_lang = user_sessions.get(user_number, "English")
        ai_response_text = get_ai_response(query_text=user_message, lang=chosen_lang)
        reply_text = f"{ai_response_text}\n\n---\nReply with 'menu' to change language."

    # Create and send the final reply
    twiml_response = MessagingResponse()
    twiml_response.message(reply_text)
    
    return Response(content=str(twiml_response), media_type="application/xml")
