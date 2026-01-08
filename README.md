# 🏭 DS03: Smart Manufacturing – IoT-Cloud Monitoring & Predictive Intelligence

## 📌 Project Overview

Modern manufacturing plants generate massive volumes of IoT sensor data, but plant managers and operators are not data scientists. They need clear, actionable insights, not complex charts or raw metrics.

This project delivers an end-to-end Smart Manufacturing solution that combines:

* IoT sensor data

* Machine Learning for predictive maintenance

* Generative AI (Gemini) for human-friendly explanations

* Streamlit for interactive deployment

The result is an AI-powered operations assistant that translates ML outputs into simple, business-ready decisions.

## 🎯 Business Problem

Unplanned machine downtime leads to:

* High maintenance costs

* Production delays

* Equipment damage

* Safety risks

## Key Challenges:

* Difficult interpretation of sensor data

* Late detection of failures

* Limited technical expertise among operations teams

## 💡 Solution Approach

We built a Predictive Intelligence System that:

* Monitors IoT sensor data

* Detects anomalies

* Predicts maintenance needs

* Uses GenAI to explain results in plain language

## 🏗️ End-to-End Architecture

IoT Sensors

     ↓

Cloud Data Storage

     ↓

Data Processing & Feature Engineering

     ↓

ML Models

  • Predictive Maintenance
  
  • Anomaly Detection (Isolation Forest)
  
     ↓

ML Outputs (Risk + Anomaly Flags)

     ↓

GenAI (Gemini)

  • Explain risks
  
  • Recommend actions
  
     ↓

Streamlit Web App

  • Interactive chatbot interface


## 🧠 Machine Learning Models Used

| Model            | Purpose                          |
| ---------------- | -------------------------------- |
| Random Forest    | Predict maintenance requirement  |
| Isolation Forest | Detect abnormal machine behavior |
| StandardScaler   | Feature normalization            |


## 🤖 Generative AI Layer

**Google Gemini API is used to:**

* Interpret ML results

* Answer operational questions

* Provide action-oriented explanations

**Example Questions:**

* Which machines are at risk today?

* Why is this machine flagged?

* What action is recommended?

* Is immediate maintenance required?

## 📊 Key Insights & Results

* Early detection of machine failures

* Clear explanation of anomaly reasons

* Reduced dependency on technical dashboards

* Faster maintenance decision-making

## 🛠️ Streamlit Application Features

* Machine selection dropdown

* Natural language question input

* GenAI-generated explanations

* Raw ML status transparency

* Lightweight deployment (<25MB dataset)

## 📁 Project Structure

DS03-Smart-Manufacturing/

│

├── StreamlitApp/

│   ├── app.py

│   ├── smart_manufacturing_data_latest.csv.gz

│   ├── final_predictive_maintenance_model.pkl

│   ├── scaler.pkl

├── DS03_Smart_Manufacturing_IoT_Cloud_Monitoring_&_Predictive_Intelligence.ipynb

├── smart_manufacturing_data.csv

├── README.md

## 🚀 Deployment

**Run Locally**

pip install -r requirements.txt

streamlit run app.py


## Required Secrets

Create .streamlit/secrets.toml:

GEMINI_API_KEY = "your_api_key_here"

## 📈 Business Impact

* ⏱️ Reduced downtime through early alerts

* 💰 Lower maintenance costs

* 🧠 Improved decision-making

* 👷 Non-technical teams empowered by GenAI

## 🔮 Future Scope

* Real-time sensor streaming

* Multi-plant monitoring dashboard

* Maintenance scheduling automation

* LLM fine-tuning on plant-specific data

* Integration with ERP / CMMS systems

## 👨‍💻 Author

Tauseef Alam

Data Scientist | IoT | Machine Learning | Generative AI

## ⭐ Final Note

This project demonstrates how ML + GenAI + Streamlit can transform raw IoT data into business intelligence that anyone can understand.

## UI Look:

<img width="1920" height="1324" alt="Streamlit_UI" src="https://github.com/user-attachments/assets/0333aabf-918d-48ba-915d-b48e1085f9bc" />

<img width="1920" height="1327" alt="Streamlit_UI_2" src="https://github.com/user-attachments/assets/8241945f-dcb6-4823-b525-9fea191a9c05" />

