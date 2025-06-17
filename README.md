# Sentiment and thematic analysis from restaurants' client reviews  🍟

This project focuses on **sentiment analysis of customer comments** related to McDonald's restaurants, aiming to provide actionable insights and faster customer feedback handling through data-driven automation and visualization.

---

## Problem :

### 1. **Volume of Feedback**
- **Challenge**: Thousands of online customer reviews make it hard to get a global view.
- **Solution**:  
  - Automatically collect and classify reviews by **sentiments**.
  - Extract main **topics** 
  - Build a **dashboard** to visualize trends without reading every single review and allow a benchmark between localisations.

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

1. 🔄 **Automated Review Collection** via public APIs, Scraping or by finding a Database already aggregated  
2. 🧠 **Sentiment Analysis** using a pre-trained NLP model  
3. 🗂️ **Thematic Categorization** of reviews by mesuring the proximity between the embeding of selected topics to the embedding of the reviews   
4. 📈 **Interactive Dashboard** for data visualization and benchmark
5. 📝 **LLM-Powered Auto-Responses** to understand and respond to positive or negative feedback.

---

## 🛠️ Tech Stack

- **Data Collection**: Database of McDonald's Store reviews found on [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/mcdonalds-store-reviews)
- **NLP**: RoBERTa
- **Topic and reviews embeding**: sentence-transformers/all-MiniLM-L6-v2
- **Dashboard**: Streamlit and Hugging Face
- **Auto Response Generation**: mistral-small-latest

---

## 📌 Goal

> Help McDonald's customer service and strategy teams to **understand, act on, and learn from customer feedback** at scale — without manually reading thousands of reviews.

---

## 📁 Project Structure

customer-sentiment-analysis/   
│                    
├── notebooks/data/  
├── app.py   
├── .streamlit/
└── README.md  

## 📊 Dashboard
👉[Click here to see Streamlit dashboard hosted on Hugging Face ![image](https://github.com/user-attachments/assets/bfeddf26-6d55-4965-93da-3e5944e677c6)](https://huggingface.co/spaces/myriamgoyet/Sentiment_Analysis)
   


## 📰 Slides of presentation
👉[Click here to see presentation on Google Slide ![image](https://github.com/user-attachments/assets/d8da5f92-f835-46d3-a896-269faaa0d744)](https://docs.google.com/presentation/d/1ebGR4GE3Pfl0D_uwOrwL8d8MGd2ShRH8U-gLwR9frOY/edit?usp=sharing)
   



