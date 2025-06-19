import os
import uuid
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from django.shortcuts import render, redirect
from django.conf import settings
from django.core.mail import EmailMessage
from django.urls import reverse
from django.contrib import messages
from io import BytesIO

# ===============================================
# 1️⃣ Load and preprocess Zillow dataset (once)
# ===============================================

df = pd.read_csv('zillow_friscotx_final.csv')

# Clean the Price column (remove $ and commas)
df['Price'] = df['Price'].str.replace('[$,]', '', regex=True).astype(float)

# Extract Beds, Baths, Sqft from Details column using regex
def extract_details(details):
    match = re.findall(r'(\d+)\s\|\s\w+\s\|\s(\d+)\s\|\s\w+\s\|\s([\d,]+)', details)
    if match:
        beds = int(match[0][0])
        baths = int(match[0][1])
        sqft = int(match[0][2].replace(',', ''))
        return pd.Series([beds, baths, sqft])
    return pd.Series([None, None, None])

df[['Beds', 'Baths', 'Sqft']] = df['Details'].apply(extract_details)
df.drop(columns=['Details', 'Address'], inplace=True)
df.dropna(inplace=True)

# ===============================================
# 2️⃣ Train Linear Regression model (once)
# ===============================================

X = df[['Beds', 'Baths', 'Sqft']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ===============================================
# 3️⃣ Price + Monthly Payment Predictor Function
# ===============================================

def predict_price_and_payment(beds, baths, sqft, rate=7.0, term=30):
    pred_price = model.predict([[beds, baths, sqft]])[0]
    r = rate / 12 / 100  # Monthly interest rate
    n = term * 12  # Total number of payments
    monthly_payment = (pred_price * r * (1 + r)**n) / ((1 + r)**n - 1)
    return round(pred_price, 2), round(monthly_payment, 2)

# ===============================================
# 4️⃣ Matplotlib Graph Generator
# ===============================================

def generate_plot(sqft, predicted_price):
    # Use BytesIO to send image via email without saving to disk
    buffer = BytesIO()
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Sqft'], df['Price'], alpha=0.6, label='Zillow Listings', color='blue')
    plt.scatter(sqft, predicted_price, color='red', label='Your Prediction', s=100)
    plt.title("Sqft vs Price - Zillow Data + Your Prediction")
    plt.xlabel("Square Footage (Sqft)")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    return buffer

# ===============================================
# 5️⃣ Main Django View Function
# ===============================================

def predict_view(request):
    if request.method == 'POST':
        # 1. Get form input
        to_email = request.POST.get('to_email')
        beds = int(request.POST.get('beds'))
        baths = int(request.POST.get('baths'))
        sqft = int(request.POST.get('sqft'))

        # 2. Predict price and monthly payment
        predicted_price, monthly_payment = predict_price_and_payment(beds, baths, sqft)

        # 3. Prepare message content
        subject = '🏡 Your House Price Prediction from JD Portal'
        message = f"""
✅ Prediction Result
----------------------------
🛏 Beds: {beds}
🛁 Baths: {baths}
📐 Sqft: {sqft}

🏠 Predicted House Price: ${predicted_price}
💰 Estimated Monthly Payment: ${monthly_payment}
        """

        # 4. Generate and attach graph
        image_buffer = generate_plot(sqft, predicted_price)

        email = EmailMessage(
            subject=subject,
            body=message,
            from_email=settings.EMAIL_HOST_USER,
            to=[to_email]
        )
        email.attach('prediction_plot.png', image_buffer.read(), 'image/png')
        email.send()

        # 5. Show success message after redirect (avoids resubmission on refresh)
        messages.success(request, f"✅ Prediction results sent to {to_email}")
        return redirect(reverse('predict_form'))

    # GET request: just show the form
    return render(request, 'predictor/form.html')
