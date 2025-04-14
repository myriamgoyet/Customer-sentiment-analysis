# Sentiment and thematic analysis from restaurants' client reviews  🍟

This project focuses on **sentiment analysis of customer comments** related to McDonald's restaurants, aiming to provide actionable insights and faster customer feedback handling through data-driven automation and visualization.

---

## Problem :

### 1. **Volume of Feedback**
- **Challenge**: Thousands of online customer reviews make it hard to get a global view.
- **Solution**:  
  - Automatically collect and classify reviews by **themes** and **sentiments**.
  - Build a **dashboard** to visualize trends without reading every single review.

### 2. **Interpretation Difficulties**
- **Challenges**:
  - How to **identify top sources of customer dissatisfaction**?
  - How to **detect what satisfies customers**?
  - How to **spot misaligned ratings** (e.g. 1 star for a positive comment)?
- **Solution**:
  - Analyze comments using **keywords** and **sentiment scores**.
  - Detect **inconsistencies** between sentiment and given star ratings.

### 3. **Slow or Inappropriate Reactions**
- **Challenge**: How to respond quickly and appropriately to customer feedback?
- **Solution**:  
  - Use **LLMs** to **automatically generate** tailored responses based on sentiment.

---

## 🚀 MVP Features

1. 🔄 **Automated Review Collection** via public APIs  
2. 🧠 **Sentiment Analysis** using a pre-trained NLP model  
3. 🗂️ **Thematic Categorization** of reviews  
4. 📈 **Interactive Dashboard** for data visualization  
5. 📝 **LLM-Powered Auto-Responses** to understand and respond to positive or negative feedback.

---

## 🛠️ Tech Stack (Planned/Used)

- **Data Collection**: Python, API integrations (e.g.Yelp reviews)
- **NLP**: RoBERTa
- **Dashboard**: Streamlit
- **Auto Response Generation**: Mistral

---

## 📌 Goal

> Help McDonald's customer service and strategy teams to **understand, act on, and learn from customer feedback** at scale — without manually reading thousands of reviews.

---

## 📁 Project Structure (Example)

customer-sentiment-analysis/
│
├── data/                   
├── notebooks/              
├── app/                    
├── models/                 
├── README.md
└── requirements.txt

