import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from django.shortcuts import render, redirect
from django.conf import settings
from django.core.mail import EmailMessage
from django.urls import reverse
from django.contrib import messages
from io import BytesIO

# 🔁 Load CSV dynamically based on dropdown selection
def load_dataset(source):
    if source == 'zillow':
        df = pd.read_csv('zillow_friscotx_final.csv')
        df['Price'] = df['Price'].str.replace('[$,]', '', regex=True).astype(float)
        df[['Beds', 'Baths', 'Sqft']] = df['Details'].str.extract(r'(\d+)\s\|\s\w+\s\|\s(\d+)\s\|\s\w+\s\|\s([\d,]+)')
        df['Sqft'] = df['Sqft'].str.replace(',', '', regex=True)
        df[['Beds', 'Baths', 'Sqft']] = df[['Beds', 'Baths', 'Sqft']].fillna(0).astype(int)
        df.drop(columns=['Details', 'Address'], inplace=True)
        df.dropna(inplace=True)
    else:
        df = pd.read_csv('realtor_processed_for_ml.csv')  # Already clean: Beds, Baths, Sqft, Price

    
    return df

# 📊 Graph Generator
def generate_plot(df, sqft, predicted_price):
    buffer = BytesIO()
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Sqft'], df['Price'], alpha=0.6, label='Listings', color='blue')
    plt.scatter(sqft, predicted_price, color='red', label='Your Prediction', s=100)
    plt.title("Sqft vs Price - Prediction")
    plt.xlabel("Square Footage (Sqft)")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    return buffer

# 💬 Prediction Logic
def predict_price_and_payment(model, beds, baths, sqft, rate=7.0, term=30):
    pred_price = model.predict([[beds, baths, sqft]])[0]
    r = rate / 12 / 100
    n = term * 12
    monthly_payment = (pred_price * r * (1 + r)**n) / ((1 + r)**n - 1)
    return round(pred_price, 2), round(monthly_payment, 2)

# 🔁 Main View
def predict_view(request):
    if request.method == 'POST':
        to_email = request.POST.get('to_email')
        beds = int(request.POST.get('beds'))
        baths = int(request.POST.get('baths'))
        sqft = int(request.POST.get('sqft'))
        source = request.POST.get('source')

        try:
            df = load_dataset(source)
            X = df[['Beds', 'Baths', 'Sqft']]
            y = df['Price']

            model = LinearRegression()
            model.fit(X, y)

            predicted_price, monthly_payment = predict_price_and_payment(model, beds, baths, sqft)

            subject = f'🏡 House Price Prediction ({source.capitalize()})'
            message = f"""
✅ Prediction Result ({source.capitalize()})
----------------------------
🛏 Beds: {beds}
🛁 Baths: {baths}
📐 Sqft: {sqft}

🏠 Predicted House Price: ${predicted_price}
💰 Estimated Monthly Payment: ${monthly_payment}
"""

            image_buffer = generate_plot(df, sqft, predicted_price)

            email = EmailMessage(subject, message, settings.EMAIL_HOST_USER, [to_email])
            email.attach('prediction_graph.png', image_buffer.read(), 'image/png')
            email.send()

            messages.success(request, f"✅ Prediction sent to {to_email} using {source.capitalize()} data.")
            return redirect(reverse('predict_form'))

        except Exception as e:
            messages.error(request, f"❌ Prediction failed: {e}")
            return redirect(reverse('predict_form'))

    return render(request, 'predictor/form.html')
