# HW4 Customer Satisfaction API

## Project Overview

This project deploys a machine learning model that predicts whether a customer review is positive or negative based on order and delivery information. The model was trained using the Olist e-commerce dataset and is served through a Flask API.

The deployed model is a **HistGradientBoostingClassifier** wrapped in a preprocessing pipeline that handles numeric scaling and categorical encoding.

---

## Live API URL

https://hw4-mlops-68aw.onrender.com

---

## API Endpoints

### 1. Health Check

**GET /health**

Checks if the API and model are running.

**Response:**
```json
{
  "status": "healthy",
  "model": "loaded"

2. Single Prediction

POST /predict

Predicts customer satisfaction for one input.

Request Example:
{
  "delivery_days": 5,
  "delivery_vs_estimated": -1,
  "total_price": 100,
  "total_freight": 10,
  "n_items": 2,
  "payment_value_sum": 110,
  "payment_installments": 1,
  "mean_price": 50,
  "order_weekday": 2,
  "order_month": 6,
  "payment_type": "credit_card"
}

Response Example:

{
  "prediction": 1,
  "probability": 0.85,
  "label": "positive"
}

3. Batch Prediction

POST /predict/batch

Predicts multiple records at once.

Request Example:

[
  {
    "delivery_days": 5,
    "delivery_vs_estimated": -1,
    "total_price": 100,
    "total_freight": 10,
    "n_items": 2,
    "payment_value_sum": 110,
    "payment_installments": 1,
    "mean_price": 50,
    "order_weekday": 2,
    "order_month": 6,
    "payment_type": "credit_card"
  }
]

Response Example:

[
  {
    "prediction": 1,
    "probability": 0.85
  }
]

Input Schema
| Feature               | Type   | Description                        |
| --------------------- | ------ | ---------------------------------- |
| delivery_days         | float  | Days between purchase and delivery |
| delivery_vs_estimated | float  | Difference from estimated delivery |
| total_price           | float  | Total order price                  |
| total_freight         | float  | Shipping cost                      |
| n_items               | int    | Number of items                    |
| payment_value_sum     | float  | Total payment value                |
| payment_installments  | int    | Number of installments             |
| mean_price            | float  | Average price per item             |
| order_weekday         | int    | Day of week (0–6)                  |
| order_month           | int    | Month (1–12)                       |
| payment_type          | string | Example: credit_card, boleto       |

Local Setup
Run without Docker
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py

API runs at:
http://127.0.0.1:5000

Run with Docker
docker build -t hw4-api .
docker run -p 5001:5000 hw4-api

Access API at:
http://127.0.0.1:5001

Model Information
Model: HistGradientBoostingClassifier
Accuracy: ~0.78
F1 Score: ~0.86
AUC: ~0.70

The model performs well and remains stable under moderate data drift.

Limitations
Model may degrade under strong data drift
Limited to available dataset features
Does not capture external factors such as seasonality or promotions
Notes

This API was containerized using Docker and deployed using Render.


