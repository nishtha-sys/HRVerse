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
- Says "I'm not sure" for off-topic questions instead of guessing
- One deployment serves both the web page and the API

## How it works

Every message goes through this pipeline:

1. **Text preprocessing (NLTK):** lowercase, tokenize, remove stop words, fix typos, stem
2. **Intent classification:** TF-IDF features + Logistic Regression (scikit-learn), trained on
   labelled example questions in `backend/training_data.py`
3. **Keyword matching:** the earlier rule-based matcher, kept as a safety net when the model is unsure
4. **Response:** a reply for the detected intent is sent back to the chat page

## Results

Measured with `python -m backend.evaluate` on 114 questions that were **not** used for training:

| Version | Accuracy |
|---|---|
| v1 raw substring matching (original) | 51.8% |
| v2 + NLP preprocessing, keyword patterns | 58.8% |
| v3 + ML classifier (current) | 96.5% |

Average time per reply: under 1 ms. The training and test questions were written by the project
team, so the score is optimistic. Testing with questions from real users is planned.

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
- [ ] Entity recognition (leave type, and similar details)
- [ ] Employee data lookup, for example remaining leave balance
- [ ] Login and admin dashboard
- [ ] Answers from uploaded policy documents