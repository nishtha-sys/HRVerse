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

CLASSIFIER = IntentClassifier(lambda text: " ".join(nlp.preprocess(text, VOCAB))).fit(TRAINING_DATA)

# The model must be at least this sure, otherwise the keyword matcher gets a chance first.
CONFIDENCE_THRESHOLD = 0.35

LEAVE = nlp.stem("leave")
NOT_SURE = "I'm not sure 🤔. Try asking about leave, salary, benefits or company policies."


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
    tokens = nlp.preprocess(message, VOCAB)
    if not tokens:
        return None, 0.0, "none"

    tag, confidence = CLASSIFIER.predict(message)
    if tag and confidence >= CONFIDENCE_THRESHOLD:
        return tag, confidence, "model"

    tag = keyword_intent(tokens)
    if tag:
        return tag, confidence, "keywords"
    return None, confidence, "none"


def get_response(user_message):
    tag, confidence, method = predict_intent(user_message)

    if tag == "out_of_scope":
        return NOT_SURE
    if tag:
        return random.choice(INTENT_BY_TAG[tag]["responses"])

    # SMART FALLBACK: nothing matched, but a single key word may still help
    tokens = nlp.preprocess(user_message, VOCAB)
    if LEAVE in tokens:
        return "You can ask about leave policy or how to apply leave."
    elif nlp.stem("salary") in tokens:
        return "Try asking about salary or payroll."
    elif "hr" in tokens:
        return "You can contact HR via email or portal."
    elif "help" in tokens:
        return "I can help with leave, salary, HR contact, benefits etc."

    return NOT_SURE


# API
@app.post("/chat")
def chat(request: ChatRequest):
    user_message = request.message[:MAX_MESSAGE_LENGTH]      # ignore anything beyond a normal question
    response = get_response(user_message)
    return {"response": response}