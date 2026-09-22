from pathlib import Path
import random

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

try:
    from . import preprocess as nlp                     # uvicorn backend.main:app (from the project root)
    from .classifier import IntentClassifier
    from .training_data import TRAINING_DATA
except ImportError:
    import preprocess as nlp                            # uvicorn main:app (from inside backend/)
    from classifier import IntentClassifier
    from training_data import TRAINING_DATA

# Absolute path to the frontend folder, so the app starts correctly
# no matter which directory you launch uvicorn from.
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="HRVerse")

# CORS (important)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve ONLY the chat page (no folder is exposed publicly)
@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


# Health check (used by the hosting platform to see that the app is running)
@app.get("/health")
def health():
    return {"status": "ok"}


# Longest message we will process (a public endpoint should not accept huge inputs)
MAX_MESSAGE_LENGTH = 500


# Request model
class ChatRequest(BaseModel):
    message: str


# 🧠 INTENTS
INTENTS = [

    # GREETING
    {
        "tag": "greeting",
        "patterns": ["hi", "hello", "hey", "good morning", "good evening"],
        "responses": [
            "Hello! 👋 I'm your HR assistant. How can I help you today?",
            "Hi there! Ask me anything about HR policies.",
            "Hey! What HR query can I assist you with?"
        ]
    },

    # THANKS
    {
        "tag": "thanks",
        "patterns": ["thanks", "thank you", "thankyou", "thx"],
        "responses": [
            "You're welcome 😊",
            "Happy to help!",
            "Anytime! Let me know if you need anything else."
        ]
    },

    # 🔥 APPLY LEAVE (IMPORTANT FIRST)
    {
        "tag": "apply_leave",
        "patterns": [
            "how to apply leave",
            "apply leave",
            "leave request",
            "how do i take leave",
            "leave process",
            "i want leave"
        ],
        "responses": [
            "To apply for leave:\n1. Login to HR portal\n2. Go to Leave section\n3. Select dates\n4. Submit for manager approval",
            "You can apply leave via employee dashboard → Leave section → submit request."
        ]
    },

    # LEAVE POLICY
    {
        "tag": "leave_policy",
        "patterns": [
            "leave policy",
            "how many leaves",
            "leave details",
            "types of leave",
            "casual leave",
            "sick leave",
            "paid leave"
        ],
        "responses": [
            "Employees get:\n• 12 Casual Leaves\n• 10 Sick Leaves\n• Maternity leave as per policy.",
            "Leave policy includes casual, sick and maternity leave.\nYou can check balance in HR portal.",
            "You are eligible for different types of leaves including casual and sick leave."
        ]
    },

    # SALARY
    {
        "tag": "salary",
        "patterns": ["salary", "pay", "payroll", "salary date"],
        "responses": [
            "Salary is credited on the last working day of every month.",
            "You can check salary slips in HR portal."
        ]
    },

    # WORK FROM HOME
    {
        "tag": "wfh",
        "patterns": ["work from home", "wfh", "remote work", "can i work from home"],
        "responses": [
            "You can work from home up to 2 days/week with manager approval.",
            "WFH depends on your manager's approval and project requirements."
        ]
    },

    # WORKING HOURS
    {
        "tag": "working_hours",
        "patterns": ["working hours", "office timing", "shift timing"],
        "responses": [
            "Standard working hours are 9 AM to 6 PM.",
            "You are expected to complete 8 hours daily."
        ]
    },

    # PROBATION
    {
        "tag": "probation",
        "patterns": ["probation", "confirmation", "probation period"],
        "responses": [
            "Probation period is 6 months.\nConfirmation depends on performance review.",
            "After probation, your manager will review and confirm your role."
        ]
    },

    # HR CONTACT
    {
        "tag": "contact_hr",
        "patterns": ["contact hr", "hr email", "hr number", "how to contact hr"],
        "responses": [
            "You can contact HR via:\n📧 hr@company.com\n📞 1800-123-456\n📍 HR Office",
            "HR team is available during working hours. You can also use the internal portal."
        ]
    },

    # BENEFITS
    {
        "tag": "benefits",
        "patterns": ["benefits", "health benefits", "insurance", "medical", "health"],
        "responses": [
            "Company provides:\n• Health Insurance\n• Medical Coverage\n• Wellness Programs",
            "Employees are covered under insurance tie-ups with partner hospitals."
        ]
    },

    # ETHICS
    {
        "tag": "ethics",
        "patterns": ["ethics", "rules", "code of conduct", "company rules"],
        "responses": [
            "Employees must follow:\n• Professional behavior\n• Confidentiality\n• Ethical work practices",
            "Code of ethics ensures a respectful and professional work environment."
        ]
    },

    # LEGAL
    {
        "tag": "legal",
        "patterns": ["legal", "regulations", "laws", "company law"],
        "responses": [
            "Company follows all government regulations and legal compliance.",
            "Employees must adhere to company policies and legal guidelines."
        ]
    },

    # ABOUT THE BOT
    {
        "tag": "about_bot",
        "patterns": ["who are you", "introduce yourself", "your name", "hrverse", "are you a bot"],
        "responses": [
            "I'm HRVerse, a virtual HR assistant. I can explain HR policies such as leave, work from home, probation, salary and benefits, and tell you how to reach the HR team.",
            "Hi, I'm HRVerse 🙂 A chatbot built to answer employees' everyday HR questions: leave, working hours, benefits, probation and more.",
            "I'm HRVerse, an NLP-based HR assistant built as a B.Tech project. I work out what your question is about and reply with the matching HR policy."
        ]
    },

    # WHAT THE BOT CAN DO
    {
        "tag": "capabilities",
        "patterns": ["what can you do", "what can i ask", "help me", "help", "your features"],
        "responses": [
            "Here is what I can help with:\n• Leave policy and how to apply\n• Work from home and working hours\n• Probation and confirmation\n• Salary and payroll\n• Benefits and insurance\n• Code of conduct and legal compliance\n• How to contact HR\nTry: \"How many sick leaves do I get?\"",
            "I can answer questions about HR policies. For example:\n• \"What is the probation period?\"\n• \"Can I work from home?\"\n• \"When is salary paid?\"\n• \"How do I contact HR?\"\nYou can also tap a topic on the Home screen."
        ]
    },

    # ASKING THE BOT TO DO SOMETHING FOR THE USER
    {
        "tag": "do_for_me",
        "patterns": ["behalf", "cancel my leave", "approve my"],
        "responses": [
            "I can't submit or change anything on your behalf yet, I can only guide you. For leave: log in to the HR portal, open the Leave section and submit your request for approval.",
            "Right now I can explain the steps but not do them for you. To apply for leave:\n1. Login to HR portal\n2. Go to Leave section\n3. Submit request for approval\nActing on your behalf from chat is planned for a later version."
        ]
    },

    # PRIVATE INFORMATION (never shared)
    {
        "tag": "personal_info",
        "patterns": ["personal number", "personal phone", "home address", "personal email", "someone else salary"],
        "responses": [
            "I can't share personal contact details or private information about employees or HR staff. To reach HR, use the official channels: hr@company.com or 1800-123-456.",
            "Personal details of colleagues and HR staff are private, so I can't help with that. The official HR contact is hr@company.com, phone 1800-123-456."
        ]
    },

    # COMPLAINTS AND CONCERNS
    {
        "tag": "grievance",
        "patterns": ["complaint", "grievance", "harassment", "harassing", "bullying", "misconduct"],
        "responses": [
            "I'm sorry you're going through this. Please raise it with HR directly so it can be looked into properly: email hr@company.com, call 1800-123-456, or visit the HR office.",
            "That sounds serious, and a person should look into it. Please contact HR at hr@company.com or 1800-123-456 and explain what happened. I can't file complaints from chat yet."
        ]
    },

    # BAD LANGUAGE
    {
        "tag": "bad_language",
        "patterns": ["idiot", "stupid", "dumb", "shut up"],
        "responses": [
            "Please maintain professional communication.",
            "Let's keep the conversation respectful.",
            "I'm here to help professionally."
        ]
    }
]


# 🧠 SMART MATCHING
# 🧠 NLP engine, built once when the server starts
#   VOCAB            known words, used to fix typos ("probaton" -> "probation")
#   CLASSIFIER       machine-learning intent model trained on training_data.py
#   PATTERN_TOKENS   the original keyword patterns, kept as a safety net
INTENT_BY_TAG = {intent["tag"]: intent for intent in INTENTS}

VOCAB = nlp.build_vocabulary(
    [p for intent in INTENTS for p in intent["patterns"]] +
    [text for text, label in TRAINING_DATA if label != "out_of_scope"]
)

PATTERN_TOKENS = [
    (intent["tag"], [t for t in (nlp.preprocess(p) for p in intent["patterns"]) if t])
    for intent in INTENTS
]

CLASSIFIER = IntentClassifier(
    keyword_tokens=lambda text: nlp.preprocess(text, VOCAB),
    phrase_tokens=lambda text: nlp.normalize(text, VOCAB),
).fit(TRAINING_DATA)

# The model must be at least this sure, otherwise the keyword matcher gets a chance first.
CONFIDENCE_THRESHOLD = 0.20

# Topics the bot can suggest when it is unsure ("Did you mean ...?")
TOPIC_LABELS = {
    "apply_leave": "how to apply for leave",
    "leave_policy": "the leave policy",
    "salary": "salary and payroll",
    "wfh": "work from home",
    "working_hours": "working hours",
    "probation": "the probation period",
    "contact_hr": "how to contact HR",
    "benefits": "employee benefits",
    "ethics": "the code of conduct",
    "legal": "legal compliance",
    "grievance": "raising a complaint or concern with HR",
}
SUGGESTION_MIN_PROBABILITY = 0.10

OUT_OF_SCOPE_REPLIES = [
    "That's outside what I know. I'm here for HR questions, so try asking about leave or working hours, or type \"what can you do\".",
    "I couldn't match that to an HR topic. I can help with leave, work from home, salary, probation, benefits and contacting HR.",
    "Hmm, I don't have an answer for that. Ask me about an HR policy, or type \"what can you do\" for ideas.",
]

# Said before the answer when someone mentions being unwell
UNWELL_WORDS = {"fever", "unwell", "injured", "injury", "hospital", "headache", "infection", "migraine", "ill"}
UNWELL_PHRASES = ("i am sick", "i m sick", "feeling sick", "not feeling well", "not well")
CARING_LINES = [
    "Sorry to hear you're not feeling well, I hope you feel better soon.",
    "I'm sorry you're unwell. Take care and get some rest.",
]
CARING_INTENTS = {"apply_leave", "leave_policy", "do_for_me"}


# Privacy safety net: private details of OTHER people are never discussed, whatever the model thinks.
OTHER_PEOPLE = {"colleague", "colleagues", "coworker", "coworkers", "teammate", "teammates",
                "manager", "managers", "boss", "ceo", "friend", "friends", "someone", "somebody"}
PRIVATE_DETAILS = {"salary", "salaries", "earn", "earns", "personal", "phone", "mobile", "whatsapp", "address", "balance"}


def asks_about_others_privately(words):
    text = " ".join(words)
    others = any(w in OTHER_PEOPLE for w in words) or "team lead" in text
    return others and any(w in PRIVATE_DETAILS for w in words)


def feels_unwell(words):
    return any(w in UNWELL_WORDS for w in words) or any(p in " ".join(words) for p in UNWELL_PHRASES)


def keyword_intent(tokens):
    """Safety net: the intent whose keyword patterns match the most (whole words), or None."""
    best_tag, best_score = None, 0
    for tag, patterns in PATTERN_TOKENS:
        score = sum(1 for pattern in patterns if nlp.contains_phrase(tokens, pattern))
        if score > best_score:
            best_tag, best_score = tag, score
    return best_tag


def predict_intent(message):
    """Return (intent, confidence, method). method is "model", "keywords" or "none"."""
    if not nlp.normalize(message):
        return None, 0.0, "none"

    tag, confidence = CLASSIFIER.predict(message)
    if tag and confidence >= CONFIDENCE_THRESHOLD:
        return tag, confidence, "model"

    tag = keyword_intent(nlp.preprocess(message, VOCAB))
    if tag:
        return tag, confidence, "keywords"
    return None, confidence, "none"


def not_sure_reply(message):
    """The model is unsure: offer the closest topics instead of a fixed sentence."""
    close = [TOPIC_LABELS[tag] for tag, p in CLASSIFIER.rank(message)
             if tag in TOPIC_LABELS and p >= SUGGESTION_MIN_PROBABILITY][:2]
    if len(close) == 2:
        return f"I'm not sure I understood that. Did you mean {close[0]} or {close[1]}?"
    if close:
        return f"I'm not sure I understood that. Are you asking about {close[0]}?"
    return random.choice(OUT_OF_SCOPE_REPLIES)


def get_response(user_message):
    if asks_about_others_privately(nlp.normalize(user_message)):
        return random.choice(INTENT_BY_TAG["personal_info"]["responses"])

    tag, confidence, method = predict_intent(user_message)

    if tag == "out_of_scope":
        return random.choice(OUT_OF_SCOPE_REPLIES)
    if tag is None:
        return not_sure_reply(user_message)

    reply = random.choice(INTENT_BY_TAG[tag]["responses"])
    if tag in CARING_INTENTS and feels_unwell(nlp.normalize(user_message)):
        reply = random.choice(CARING_LINES) + "\n" + reply
        if tag != "leave_policy":
            reply += "\nChoose Sick Leave when you submit the request."
    return reply


# API
@app.post("/chat")
def chat(request: ChatRequest):
    user_message = request.message[:MAX_MESSAGE_LENGTH]      # ignore anything beyond a normal question
    response = get_response(user_message)
    return {"response": response}