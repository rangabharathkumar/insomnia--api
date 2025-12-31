# Insomnia API (alternate repo) — Health Oracle

Live docs: https://insomnia-api-lf3g.onrender.com/docs

Description

This repository contains an Insomnia prediction API implementation that is part of the Health Oracle application. The ML model has been trained, tested and deployed; this service exposes endpoints to obtain predictions. Check the live docs for the definitive list of endpoints and payloads.

Key notes

- Part of Health Oracle microservices.
- Trained and deployed ML model behind the API.
- Interactive OpenAPI docs at the /docs endpoint.

Quick Start (consume the API)

- Base URL: https://insomnia-api-lf3g.onrender.com
- Docs: https://insomnia-api-lf3g.onrender.com/docs

Example JSON request (verify exact fields in /docs):

POST /predict
{
  "age": 42,
  "gender": "male",
  "sleep_duration_hours": 6,
  "sleep_quality_score": 4,
  "exercise_freq_per_week": 1,
  "stress_level": 4
}

Example JSON response (example):
{
  "prediction": 0,
  "probability": 0.23,
  "label": "no_insomnia"
}

Local setup

1. Clone the repository
   git clone https://github.com/rangabharathkumar/insomnia--api.git
2. Create a Python virtual environment and install dependencies
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
3. Start the API locally
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Check http://localhost:8000/docs for interactive docs.

Environment/config

- Store model files, API keys, and other secrets in environment variables or a .env file.

Testing

- Run tests with pytest if available.

Using as a reference

- This repo is useful as a pattern for serving ML models in production-grade REST APIs.

Author

Ranga Bharath Kumar — https://github.com/rangabharathkumar

Live docs: https://insomnia-api-lf3g.onrender.com/docs
