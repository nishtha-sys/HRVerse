# HRVerse - NLP-based HR Assistant for Employee Queries

A chatbot that answers common employee questions about HR policies (leave, work from home,
probation, salary, benefits, contacting HR and more), by text or by voice.

**Live demo:** https://hrverse-je3k.onrender.com

> The demo runs on a free hosting plan. It goes to sleep when nobody has used it for about
> 15 minutes, so the first visit after a break can take up to a minute to load.

Minor project, B.Tech Computer Science & Engineering (AI), Babu Banarasi Das University, Lucknow.

## Features

- Chat interface with a home screen of topic cards, light and dark theme, works on phone and laptop
- Voice input (microphone) and read-aloud answers, in Chrome and Edge
- Understands typos and different wordings: "probaton period", "wanna take a day off"
- Introduces itself and explains what it can do ("who are you", "what can you do")
- Handles off-topic questions politely and suggests the closest topic when it is unsure ("Did you mean ...?")
- Never shares private details of employees or HR staff, and points complaints and harassment reports to HR
- Says honestly what it cannot do yet, such as applying for leave on your behalf
- One deployment serves both the web page and the API

## How it works

Every message goes through this pipeline:

1. **Text preprocessing (NLTK):** lowercase, tokenize, remove stop words, fix typos, stem
2. **Intent classification:** TF-IDF features + Logistic Regression (scikit-learn), trained on
   labelled example questions in `backend/training_data.py`
3. **Safety nets:** a keyword matcher for when the model is unsure, and a rule that blocks requests
   for other people's private details
4. **Response:** a reply for the detected intent, or the closest topics if the bot is unsure

## Results

Measured with `python -m backend.evaluate` on 156 questions (19 intents) that were **not** used for training:

| Version | Accuracy |
|---|---|
| v1 raw substring matching (original) | 39.1% |
| v2 + NLP preprocessing, keyword patterns | 44.2% |
| v3 + ML classifier (current) | 93.6% |

The earlier versions did not know the newer skills (introductions, capabilities, privacy, complaints),
so those count as misses for them. Average time per reply: about 2 ms.

The training and test questions were written by the project team, so the score is optimistic.
Cross-validation on the training data gives 82.7%, which is a more cautious estimate. Testing with
questions from real users is planned.

## Known limitations

- The bot only covers the topics listed in `training_data.py`. Questions about other HR topics
  (for example holidays or resignation) get a polite "not sure" reply until they are added.
- Answers are ready-made policy text chosen by the model. It is not a generative AI.
- It does not yet remember earlier messages, and there is no login or chat history.

## Run it on your computer

Needs Python 3.12 or newer.

```
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000 in Chrome. To see the accuracy report:

```
python -m backend.evaluate
```

## Project structure

```
backend/
  main.py            FastAPI app, intents and responses
  preprocess.py      text preprocessing (NLTK)
  classifier.py      intent classifier (TF-IDF + Logistic Regression)
  training_data.py   labelled training questions
  test_data.py       held-out test questions
  evaluate.py        accuracy report
frontend/
  index.html         the chat page (HTML, CSS, JavaScript)
hr_data.py           sample HR data for the planned employee lookup
render.yaml          hosting configuration (Render)
requirements.txt     Python packages
```

## Roadmap

- [x] Rule-based chatbot with web interface
- [x] NLP preprocessing and typo handling
- [x] Trained intent classifier with evaluation
- [x] Voice input and spoken answers
- [x] Deployed online
- [ ] Conversation memory (follow-up questions)
- [ ] Login and saved chat history
- [ ] Better voices through a text-to-speech service
- [ ] Entity recognition (leave type, and similar details)
- [ ] Employee data lookup, for example remaining leave balance
- [ ] Admin dashboard
- [ ] Answers from uploaded policy documents