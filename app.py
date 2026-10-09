
import os
from urllib.parse import urlparse

import torch
from flask import Flask, render_template, request
from playwright.sync_api import sync_playwright
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from google import genai

app = Flask(__name__)

# -----------------------------
# 1. Load the fake news model
# -----------------------------
MODEL_NAME = "hamzab/roberta-fake-news-classification"

print("Loading AI model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

print("AI model loaded successfully!")

# -----------------------------
# 2. Connect to Gemini
# -----------------------------
api_key = os.getenv("GEMINI_API_KEY")
gemini_client = None

if api_key:
    try:
        gemini_client = genai.Client(api_key=api_key)
        print("Gemini client initialized.")
    except Exception as e:
        print("Gemini initialization failed:", e)
else:
    print("Gemini API key not found.")
    print("News prediction will work without Gemini explanations.")


# -----------------------------
# 3. Extract article from URL
# -----------------------------
def extract_news_from_url(url):
    if not url:
        return "", "", "Please enter a news URL."

    try:
        parsed_url = urlparse(url.strip())

        if parsed_url.scheme not in ("http", "https"):
            return "", "", "Please enter a valid HTTP or HTTPS URL."

        if not parsed_url.netloc:
            return "", "", "The URL is invalid."

        title = ""
        article_text = ""

        # Keep all Playwright operations inside this context.
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)

            try:
                page = browser.new_page(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/130.0.0.0 Safari/537.36"
                    )
                )

                page.goto(
                    url.strip(),
                    wait_until="domcontentloaded",
                    timeout=30000
                )

                title = page.title()

                # Try to extract the main article.
                selectors = [
                    "article",
                    '[itemprop="articleBody"]',
                    "main",
                    "body"
                ]

                for selector in selectors:
                    try:
                        locator = page.locator(selector).first
                        content = locator.inner_text(timeout=5000).strip()

                        if len(content) > len(article_text):
                            article_text = content

                        if len(article_text) >= 200:
                            break

                    except Exception:
                        continue

            finally:
                # Close the browser before Playwright stops.
                try:
                    browser.close()
                except Exception:
                    pass

        if not article_text.strip():
            return title, "", (
                "Could not extract article text. "
                "Try another news URL or paste the news text directly."
            )

        return title, article_text, ""

    except Exception as e:
        print("URL extraction error:", str(e))
        return "", "", (
            "Unable to read this URL. The website may block automated "
            "access, or the page may be unavailable. Try another URL "
            "or use the News Text option."
        )


# -----------------------------
# 4. Predict fake or real
# -----------------------------
def predict_news(news_text):
    if not news_text or not news_text.strip():
        return None, None, "Please enter some news text."

    try:
        # Limit input length for the model.
        inputs = tokenizer(
            news_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(
                outputs.logits, dim=1
            )[0]

        predicted_id = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_id].item()) * 100

        # Use the model's configured class labels where available.
        id2label = getattr(model.config, "id2label", {}) or {}
        raw_label = str(
            id2label.get(
                predicted_id,
                id2label.get(str(predicted_id), "")
            )
        ).strip()

        if "fake" in raw_label.lower():
            prediction = "Fake"
        elif "real" in raw_label.lower() or "true" in raw_label.lower():
            prediction = "Real"
        else:
            # Fallback for models whose labels are named LABEL_0/LABEL_1.
            # Confirm this order against the model's documentation.
            prediction = "Fake" if predicted_id == 0 else "Real"

        return prediction, round(confidence, 2), None

    except Exception as e:
        print("Prediction error:", str(e))
        return None, None, "Prediction failed. Please try again."


# -----------------------------
# 5. Generate explanation
# -----------------------------
def generate_explanation(news_text, prediction, confidence):
    if gemini_client is None:
        return (
            "Gemini explanation is unavailable because the API key "
            "is not configured. The news prediction was produced "
            "by the machine-learning model."
        )

    prompt = f"""
You are assisting with a student fake-news-detection project.

Analyze the following news text carefully.

News text:
{news_text[:6000]}

Machine-learning model prediction: {prediction}
Model confidence: {confidence}%

Explain briefly:
1. What aspects of the text may support or weaken its credibility.
2. Whether the text contains claims that should be independently verified.
3. Why this prediction should not be treated as definitive proof.

Do not invent facts or claim that you verified external sources.
Clearly distinguish suspicious wording from verified evidence.
"""

    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.0-flash"
    ]

    for model_name in models_to_try:
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            if response.text:
                return response.text

        except Exception as e:
            print(f"Gemini model {model_name} failed:", str(e))

    return (
        "The news prediction is available, but Gemini could not "
        "generate an explanation. Check the API key, model access, "
        "internet connection, and API quota."
    )


# -----------------------------
# 6. Flask home page
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    explanation = None
    error = None
    title = ""
    extracted_text = ""
    news_text = ""
    url = ""

    if request.method == "POST":
        # Support common form field names.
        news_text = (
            request.form.get("news_text", "").strip()
            or request.form.get("text", "").strip()
        )
        url = request.form.get("url", "").strip()

        # If a URL was submitted, extract the article first.
        if url:
            title, extracted_text, extraction_error = (
                extract_news_from_url(url)
            )

            if extraction_error:
                error = extraction_error
            else:
                news_text = extracted_text

        if not error:
            if not news_text:
                error = "Please enter news text or provide a news URL."
            else:
                prediction, confidence, prediction_error = (
                    predict_news(news_text)
                )

                if prediction_error:
                    error = prediction_error
                else:
                    explanation = generate_explanation(
                        news_text,
                        prediction,
                        confidence
                    )

    # Keep these variable names available to the existing template.
    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        explanation=explanation,
        error=error,
        title=title,
        extracted_text=extracted_text,
        news_text=news_text,
        url=url,
        result=prediction,
        gemini_explanation=explanation
    )


# -----------------------------
# 7. Start Flask
# -----------------------------
if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)