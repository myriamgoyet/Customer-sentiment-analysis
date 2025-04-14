# Pour faire tourner le fichier app.py, il faut utiliser le fichier data_avec_labels.csv dans le dossier data

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
import hashlib
import os
from dotenv import load_dotenv

# ============================== CONFIG ===================================
DATA_PATH = Path("data\data_avec_labels.csv")

# ============================== PAGE SETUP ==============================
st.set_page_config(layout="wide", page_title="Restaurant Review Dashboard", page_icon="📊")

st.markdown("""
    <div style='padding-left: 10px; padding-right: 10px;'>
        <h1>Mcdonald's Dashboard</h1>
    </div>
""", unsafe_allow_html=True)


# ============================== LABELISATION =============================
# Rappel de la liste des labels utilisées par le model
labels = [
    'hygiene', 'food quality', 'food', 'staff', 'something is missing',
    'location', 'speed of service', 'drive-thru', 'temperature of the food',
    'atmosphere', 'customer service', "temperature" , "price", "speed", "quality", "courtesy",
]

# ============================== LOAD DATA ==============================
@st.cache_data
def load_data(path: Path):
    """Load review data from a CSV file with basic preprocessing."""
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()  # Clean column names
        df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
        df["review_date"] = pd.to_datetime(df["review_date"])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


# ============================== FILTER DATA ==============================
def apply_filters(df):
    """Apply hierarchical location and date filters via the sidebar."""
    st.sidebar.header("📍 Location Filters")

    df["review_date"] = pd.to_datetime(df["review_date"])

    # === Select State ===
    state_list = sorted(df["State"].dropna().unique())
    selected_state = st.sidebar.selectbox("Select a State", ["All"] + state_list)

    # === Select City (based on State) ===
    if selected_state == "All":
        city_list = sorted(df["City"].dropna().unique())
    else:
        city_list = sorted(df[df["State"] == selected_state]["City"].dropna().unique())
    selected_city = st.sidebar.selectbox("Select a City", ["All"] + city_list)

    # === Select Restaurant Address (based on State & City) ===
    if selected_state == "All" and selected_city == "All":
        address_list = sorted(df["store_address"].dropna().unique())
    elif selected_city == "All":
        address_list = sorted(df[df["State"] == selected_state]["store_address"].dropna().unique())
    elif selected_state == "All":
        address_list = sorted(df[df["City"] == selected_city]["store_address"].dropna().unique())
    else:
        address_list = sorted(df[(df["State"] == selected_state) & (df["City"] == selected_city)]["store_address"].dropna().unique())
    selected_address = st.sidebar.selectbox("Select a Restaurant", ["All"] + address_list)

    # Update State and City based on selected address
    if selected_address != "All":
        address_info = df[df["store_address"] == selected_address].iloc[0]
        selected_state = address_info["State"]
        selected_city = address_info["City"]

    # === Date Filter ===
    st.sidebar.header("📆 Select a Period")
    min_date = df["review_date"].min()
    max_date = df["review_date"].max()

    # Apply filters
    start_date = st.sidebar.date_input(
        "Starting date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    end_date = st.sidebar.date_input(
        "End date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )

    # Protect against bad input (e.g. end before start)
    if start_date > end_date:
        st.sidebar.error("❌ End date must be after start date.")
        return pd.DataFrame()

    filtered_df = df[(df["review_date"] >= pd.to_datetime(start_date)) & (df["review_date"] <= pd.to_datetime(end_date))]

    if selected_state != "All":
        filtered_df = filtered_df[filtered_df["State"] == selected_state]
    if selected_city != "All":
        filtered_df = filtered_df[filtered_df["City"] == selected_city] 
    if selected_address != "All":
        filtered_df = filtered_df[filtered_df["store_address"] == selected_address] 

    # Store filters in session
    current_filters = {
        "state": selected_state,
        "city": selected_city,
        "address": selected_address,
        "start_date": start_date,
        "end_date": end_date
    }

    # Check if filters have changed
    if "selected_filters" not in st.session_state or st.session_state["selected_filters"] != current_filters:
        st.session_state["selected_filters"] = current_filters
        # Reset pagination state
        st.session_state.positive_start_index = 0
        st.session_state.negative_start_index = 0
        st.session_state.positive_page = 1
        st.session_state.negative_page = 1
        # Reset topic filter
        st.session_state.positive_topic = "All"
        st.session_state.negative_topic = "All"

    return filtered_df

# ============================== UI HELPERS ==============================

def render_metric(label, value, bg_color, text_color):
    """
    Render a custom metric box.
    Accepts both numbers and strings (like percentages).
    """
    st.markdown(f"""
        #### {label}
        <div style='
            background-color:{bg_color}; 
            border-radius: 60%; 
            display: flex;
            align-items: center;
            justify-content: center;
            text-align:center; 
            font-size:25px; 
            font-weight: bold;
            width: 156px;
            height: 150px;
            margin: auto;
            color:{text_color};'>
            {value if isinstance(value, str) else f"{value:,}"}
        </div>
    """, unsafe_allow_html=True)

# Prompt system
sys_prompt = """
You are a manager at McDonald's, responding to client reviews about the restaurant.
You need to be extremely polite and speak correct English.
Thank the client for their review.
If the review mentions a bad experience, apologize for it and invite the client to come again.
Limit your answer to two sentences.
"""

# LangChain pipeline
template = ChatPromptTemplate.from_messages([
    ("system", sys_prompt),
    ("user", "{text}"),
])

# MISTRAL API KEY: real key is saved in the .env
load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")
model = ChatMistralAI(model="mistral-small-latest",mistral_api_key=mistral_api_key)
parser = StrOutputParser()
chain = template | model | parser  # ✅ Cette variable "chain" est celle à passer à render_comments


# Seuil pour filtrer les labels
seuil = 0.2

def render_comments(comments, color_primary, color_secondary, chain, sentiment=""):
    """Render a list of comments in stylized boxes."""
    for index, comment in comments.items():
        if index not in filtered_df.index:
            continue
        # Récupérer les labels avec un score supérieur au seuil pour ce commentaire
        labels_above_threshold = filtered_df[labels].loc[index] > seuil
        selected_labels = filtered_df[labels].columns[labels_above_threshold].tolist()
        labels_str = " ".join([f"#{label}" for label in selected_labels])
        formatted_comment = comment.replace('\n', ' ')

        # Générer une clé unique
        hash_key = hashlib.md5(f"{index}-{comment}-{sentiment}".encode()).hexdigest()[:8]
        unique_key = f"show_reply_{hash_key}"

        # Conteneur visuel du commentaire
        st.markdown(f"<div style='background-color:{color_primary}; padding:10px; border-radius:10px; margin-bottom:10px;'>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:{color_primary};'>💬 {formatted_comment}</div>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{color_secondary}; font-weight: bold;'>{labels_str}</p>", unsafe_allow_html=True)

        # Bouton qui déclenche la génération via LLM
        if st.button("Generate an answer", key=unique_key):
            with st.spinner("Loading answer..."):
                generated_response = chain.invoke({"text": formatted_comment})

            st.markdown(f"<p style='color:gray; font-style:italic;'>🤖 {generated_response}</p>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
# ============================== MAIN APP ==============================
with st.spinner("Loading data..."):
    df = load_data(DATA_PATH)

if df.empty:
    st.warning("No data available. Please check the source file.")
    st.stop()

filtered_df = apply_filters(df)

dashboard_tab, reviews_tab = st.tabs(["📊 Overview", "📈 Review Trends"])

# ============================== NPS MAPPING HELPER ==============================

def compute_nps_value(sentiment):
    """
    Convert sentiment to NPS value:
    - 'positive' => +1 (Promoter)
    - 'neutral'  =>  0 (Passive)
    - 'negative' => -1 (Detractor)
    """
    return {'positive': 1, 'neutral': 0, 'negative': -1}.get(sentiment, 0)


sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
filtered_df["nps_value"] = filtered_df["pred_sentiment"].map(sentiment_mapping).fillna(0)


# Calcul des % par catégorie
promoters_pct = (filtered_df["nps_value"] == 1).mean() * 100
detractors_pct = (filtered_df["nps_value"] == -1).mean() * 100
passives_pct = (filtered_df["nps_value"] == 0).mean() * 100

# NPS Global
nps_score = promoters_pct - detractors_pct

### MAP NPS SCORE pour les bubbles 
df['nps_val_sentiment'] = df["pred_sentiment"].map(sentiment_mapping).fillna(0)
df["review_count"] = df.groupby("store_address")["review"].transform("count")
filtered_NPS_df = df[df["review_count"] > 100]


# ============================== METRICS ====================================================
with dashboard_tab:
    total_reviews = len(filtered_df)

    # ============================== TITLE ==============================
    ### attention, si changements, ne pas oublier de changer également dans reviews_tab
    filters = st.session_state.get("selected_filters", {})
    start = filters.get("start_date")
    end = filters.get("end_date")
    
    # Déterminez le selected_scope en fonction des filtres sélectionnés
    if filters.get("state")=="All" and filters.get("city")=="All" and filters.get("address")=="All":
        selected_scope = "All Restaurants"
    elif filters.get("state") and filters.get("city")=="All" and filters.get("address")=="All":
        selected_scope = f"All Restaurants, {filters['state']}"
    elif filters.get("state") and filters.get("city") and filters.get("address")=="All":
        selected_scope = f"All Restaurants, {filters['city']}, {filters['state']}"
    elif filters.get("address") and filters.get("city") and filters.get("state"):
        selected_scope = f"📍 {filters['address']}, {filters['city']}, {filters['state']}"
    else:
        selected_scope = "All Restaurants"

    filtered_title = f"{selected_scope} — _{start.strftime('%b %d, %Y')} to {end.strftime('%b %d, %Y')}_"
    st.markdown(filtered_title)

    # ============================== AFFICHAGE DU SCORE NPS GLOBAL ==============================

    #====== Appliquer la fonction pour créer une colonne 'nps_value'==========
    filtered_df["nps_value"] = filtered_df["pred_sentiment"].apply(compute_nps_value)

    #========= Calculer les pourcentages ===============
    promoters_pct = (filtered_df["nps_value"] == 1).mean() * 100
    detractors_pct = (filtered_df["nps_value"] == -1).mean() * 100
    passives_pct = (filtered_df["nps_value"] == 0).mean() * 100

    #========= Affichage des métriques ==================
    total_reviews = len(filtered_df)
    nps_color = "#1aa442" if nps_score > 50 else "#b36500" if nps_score > 0 else "#aa0000"
    nps_text_color = "#ffffff"

    # ============== Affichage Metrics (Total Reviews, Prom, Pass, Detra) =========================

    total_col, prom_col, passif_col, detract_col = st.columns(4)

    with total_col:
        st.markdown(f"""
            ### 📊 Total Reviews
            <div style="
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center; 
                font-size: 50px; 
                font-weight: bold;
                width: 130px;
                height: 110px;
                margin: auto;">
                {total_reviews if isinstance(total_reviews, str) else f"{total_reviews:,}"}
            </div>
        """, unsafe_allow_html=True)


    with prom_col:
        render_metric("🙂 Promoters", f"{promoters_pct:.1f}%", "#137830", "#b7f7d0")
    with passif_col:
        render_metric("😐 Passives", f"{passives_pct:.1f}%", "#b36500", "#eeeeee")
    with detract_col:
        render_metric("😠 Detractors", f"{detractors_pct:.1f}%", "#aa0000", "#ffb6b6")

    st.divider()

    # ====================== AFFICHAGE NPS SCORE AND LOCATION MAP AND WEEKLY TRENDS ==============================


    # ============= MCdonalds US map ==============

    nps_col , map_col = st.columns([2,8])

    with map_col:

        st.markdown("### 🗺️ Mcdonald's US Map")
        
        # US MAP Tooltip
        with st.expander("ℹ️ About this Map"):
            st.markdown("""
            This map shows McDonald's locations across the US. 
            - **Bubble size** = Number of reviews
            - **Color** = NPS score (green = high, red = low)
            - Hover over bubbles to explore store details.
        """)

        required_cols = {"latitude", "longitude", "City", "store_address", "pred_sentiment", "clean_reviews"}
        if required_cols.issubset(filtered_df.columns):

            # Prepare data: drop missing coordinates and compute NPS value
            location_df = filtered_df.dropna(subset=["latitude", "longitude", "store_address"])
            location_df["nps_value"] = location_df["pred_sentiment"].apply(compute_nps_value)

            # Group by store location
            map_data = location_df.groupby(["store_address","City","State","latitude", "longitude", ]).agg(
                review_count=("clean_reviews", "count"),
                nps_score=("nps_value", lambda x: (x == 1).mean() * 100 - (x == -1).mean() * 100)
            ).reset_index()

            # Filter stores with more than 100 reviews
            map_data = map_data[map_data["review_count"] > 100]

            if not map_data.empty:
                # Scale bubble size
                map_data["size_scaled"] = map_data["review_count"].apply(lambda x: min(x, 100))

                # Define fixed color scale (same visual logic every time)
                custom_nps_scale_map = [
                    [0.0, "red"],
                    [0.3, "red"],
                    [0.5, "white"],
                    [0.6, "green"],
                    [1.0, "green"]
                ]

                fig_map = px.scatter_geo(
                    map_data,
                    hover_data={
                        "nps_score": ':.1f',
                        "review_count": True,
                        "City": True,
                        "State":True
                    },
                    lat="latitude",
                    lon="longitude",
                    size="size_scaled",
                    color="nps_score",
                    color_continuous_scale=custom_nps_scale_map,
                    range_color=[-70,70],
                    hover_name="store_address",
                    size_max=15,
                    scope="usa",
                    template="plotly_dark"
                )

                fig_map.update_layout(
                    margin=dict(l=0, r=0, t=50, b=10),
                    coloraxis_colorbar=dict(
                        title="NPS Score"
                    )
                )

                st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})

                #--------- Add a caption under the map--------------------
                # Safely extract filters
                state = filters.get("state")
                city = filters.get("city")
                address = filters.get("address")
                start = filters.get("start_date")
                end = filters.get("end_date")

                # Determine the dynamic location label
                if address and address != "All":
                    location_label = f"📍 Store at {address}"
                elif city and city != "All":
                    location_label = f"🏙️ City: {city}"
                elif state and state != "All":
                    location_label = f"🗺️ State: {state}"
                else:
                    location_label = "All Restaurants (Data only available for restaurants with more than 100 reviews)"

                # Compose the full dynamic caption
                if start and end:
                    caption = (
                        f"{location_label}<br>"
                        f"📅 From {start.strftime('%b %d, %Y')} to {end.strftime('%b %d, %Y')}<br>"
                    )
                else:
                    caption = (
                        f"{location_label}<br>"
                    )

                # Display the caption under the map
                st.markdown(
                    f"<div style='text-align:center; font-size:16px; margin-top:-10px;'>{caption}</div>",
                    unsafe_allow_html=True
                )


            else:
                st.info("No location data available.")
        else:
            st.info("Incomplete location data.")

#================= NPS SCORE ==========================
    with nps_col:
        st.markdown("### 🧮 NPS Score ")
        
        #Tooltip of NPS 
        with st.expander("ℹ️ What is NPS?", expanded=False):
            st.markdown("""
                **Net Promoter Score (NPS)** measures customer loyalty by subtracting the percentage of detractors from promoters.
                
                - **Promoters** (positive): Loyal enthusiasts.
                - **Passives** (neutral): Satisfied but unenthusiastic.
                - **Detractors** (negative): Unhappy customers.

                **NPS = %Promoters - %Detractors**
            """)

        render_metric("NPS Score", f"{nps_score:.1f}", nps_color, nps_text_color)
        st.subheader(f"")
    
    st.divider()
    
# ========================== NPS by restaurant Bar Chart ============================ 
    
    # NPS Scores of restaurants' title
    st.markdown("### 🏙️ Restaurants performances")

    # NPS BAR CHART Tooltip 
    with st.expander("ℹ️ About this bar chart"):
        st.markdown("""
        This chart shows Net Promoter Score (NPS) for each store based on filtered date and location.
        
        - **Higher bars** mean better customer sentiment.
        - Hover over a bar to get restaurant's info like review count, city, and state.
    """)


    # Retrieve current filters from session stat
    filters = st.session_state.get("selected_filters", {})
    start = filters.get("start_date")
    end = filters.get("end_date")


    if "store_address" in filtered_df.columns and not filtered_df["store_address"].isna().all():
        
        # Group by restaurant and compute NPS score as the difference between the percentage of promoters and detractors
        nps_by_restaurant = (
            filtered_df.groupby("store_address")
            .agg(
                NPS=("nps_value", lambda x: (x == 1).mean() * 100 - (x == -1).mean() * 100),
                review_count=("nps_value", "count"),
                City=("City", "first"),
                State=("State", "first")
            )
            .reset_index()
            .sort_values("NPS", ascending=True)  # Sort from highest to lowest NPS
        )

        # Cap review count for visualization
        nps_by_restaurant = nps_by_restaurant[nps_by_restaurant["review_count"] > 100]
        
        # Sort after filtering
        nps_by_restaurant = nps_by_restaurant.sort_values("NPS", ascending=False)

        # Define custom color gradient (red to white to green)
        custom_nps_scale_bar = [
                [0.0, "red"], 
                [0.3, "red"],    
                [0.5, "white"],
                [0.6, "green"],  
                [1.0, "green"]  
            ]

        # BAR CHART restaurants par NPS Score
        fig_nps = px.bar(
            nps_by_restaurant,
            x="store_address",
            y="NPS",
            orientation="v",
            color="NPS",
            range_color=[-60 , 60],
            color_continuous_scale=custom_nps_scale_bar,
            hover_data={
                "NPS": ":.2f",
                "review_count": True,
                "City": True,
                "State": True
                        },
            template="plotly_dark",
            height=600
        )

        # Rotate x-axis labels to improve readability
        fig_nps.update_layout(xaxis_title="Restaurants",yaxis_title="NPS Score", xaxis_tickangle=-45 ) 
        st.plotly_chart(fig_nps, use_container_width=True )

        # Build dynamic title based on the "address" filter:
        selected_scope = "All Restaurants with more than 100 reviews" if filters.get("address") == "All" else f"📍 {filters.get('address')}"
        nps_title = f"NPS Score of {selected_scope} — {start.strftime('%b %d, %Y')} to {end.strftime('%b %d, %Y')}"

        # Display the dynamic title in the app
        st.markdown(f"""<div style='text-align:center; font-size:16px; margin-top:-10px;'>
                    {nps_title}
                    </div>""",
                    unsafe_allow_html=True)

    else:
        st.info("No available data for selected City.")

    st.divider()

    
# ============================== TOP TOPICS ==============================
    
with reviews_tab:
    # ============================== TITLE ==============================
    ### attention, si changements, ne pas oublier de changer également dans dashboard_tab
    filters_tab2 = st.session_state.get("selected_filters", {})
    start = filters_tab2.get("start_date")
    end = filters_tab2.get("end_date")
    
    # Déterminez le selected_scope en fonction des filtres sélectionnés
    if filters_tab2.get("state")=="All" and filters_tab2.get("city")=="All" and filters_tab2.get("address")=="All":
        selected_scope = "All Restaurants"
    elif filters_tab2.get("state") and filters_tab2.get("city")=="All" and filters_tab2.get("address")=="All":
        selected_scope = f"All Restaurants, {filters_tab2['state']}"
    elif filters_tab2.get("state") and filters_tab2.get("city") and filters_tab2.get("address")=="All":
        selected_scope = f"All Restaurants, {filters_tab2['city']}, {filters_tab2['state']}"
    elif filters_tab2.get("address") and filters_tab2.get("city") and filters_tab2.get("state"):
        selected_scope = f"📍 {filters_tab2['address']}, {filters_tab2['city']}, {filters_tab2['state']}"
    else:
        selected_scope = "All Restaurants"

    filtered_title = f"{selected_scope} — _{start.strftime('%b %d, %Y')} to {end.strftime('%b %d, %Y')}_"
    st.markdown(filtered_title)

    # ============================== TOPICS BAR ==============================
    with st.expander("ℹ️ What is Topic ratio ?", expanded=False):
        st.markdown("""
            **The Topic Ratio** measures the significance of a topic's frequency in positive or negative reviews relative to its overall frequency.

            It is calculated as follows:
            - Positive Topic Ratio = Count in positive reviews / Total count
            - Negative Topic Ratio = Count in negative reviews / Total count

            Example: A topic with a ratio of 0.7 in the "Most Positive Topics" graph indicates that over all reviews that mention this topic, 70% of them are positive reviews.
            """)
        
    topics_col1, topics_col2 = st.columns(2)

    # ==== Initialisation of the topic dataset
    positive_df = filtered_df[filtered_df['pred_sentiment'] == 'positive']
    top_topics_pos = (positive_df[labels] > seuil).sum()
    df_pos = top_topics_pos.reset_index()
    df_pos.columns = ['labels', 'count_positif']

    negative_df = filtered_df[filtered_df['pred_sentiment'] == 'negative']
    top_topics_neg = (negative_df[labels] > seuil).sum()
    df_neg = top_topics_neg.reset_index()
    df_neg.columns = ['labels', 'count_negatif']
    
    topic_df = pd.merge(df_pos, df_neg, on='labels', how='outer')

    # Initialiser les colonnes 'frec_positif_vs_posneg' 'frec_negatif_vs_posneg' 'frec_positif_vs_totpos' 'frec_negatif_vs_totneg'
    topic_df["frec_positif_vs_posneg"] = None
    topic_df["frec_negatif_vs_posneg"] = None
    topic_df["frec_positif_vs_totpos"] = None
    topic_df["frec_negatif_vs_totneg"] = None

    for i in topic_df.index:
        topic_df.loc[i, "frec_positif_vs_posneg"] = (topic_df.loc[i, "count_positif"]) / (topic_df.loc[i, "count_positif"] + topic_df.loc[i, "count_negatif"])
        topic_df.loc[i, "frec_negatif_vs_posneg"] = (topic_df.loc[i, "count_negatif"]) / (topic_df.loc[i, "count_positif"] + topic_df.loc[i, "count_negatif"])
        topic_df.loc[i, "frec_positif_vs_totpos"] = ((topic_df.loc[i, "count_positif"]) / (topic_df["count_positif"].sum()))*100
        topic_df.loc[i, "frec_negatif_vs_totneg"] = ((topic_df.loc[i, "count_negatif"]) / (topic_df["count_negatif"].sum()))*100

    with topics_col1:
        st.markdown("""
    <div style='padding-left: 10px; padding-right: 10px;'>
        <h4>😍 Most Positive Topics</h4>
        <i><h7>Topics with at least 5% occurrences among positive reviews and ratio above 0.50</h7></i>
    </div>
    """, unsafe_allow_html=True)

                    
        top_topics = topic_df[topic_df["frec_positif_vs_totpos"]>5][["labels", "frec_positif_vs_posneg"]].sort_values(by="frec_positif_vs_posneg", ascending=False)
        topic_to_show = top_topics[top_topics["frec_positif_vs_posneg"]>0.51]
        fig_topics = go.Figure()

        fig_topics.add_trace(go.Bar(
            x=topic_to_show["frec_positif_vs_posneg"],
            y=topic_to_show["labels"],
            orientation='h',
            marker=dict(color='green'),
            texttemplate='%{x:.2f}',  # Formater le texte pour afficher deux décimales
            textposition='auto',
            hovertemplate='%{y}: %{x} mentions<extra></extra>',
        ))

        fig_topics.update_layout(
            xaxis_title="Topic ratio",
            height=500,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20),
        )

        fig_topics.update_yaxes(autorange="reversed")  # Most frequent on top
        st.plotly_chart(fig_topics, use_container_width=True)


    with topics_col2:
        st.markdown("""
    <div style='padding-left: 10px; padding-right: 10px;'>
        <h4>🤬 Most Negative Topics</h4>
        <i><h7>Topics with at least 5% occurrences among negative reviews and ratio above 0.50</h7></i>
    </div>
    """, unsafe_allow_html=True)
        top_topics = topic_df[topic_df["frec_negatif_vs_totneg"]>5][["labels", "frec_negatif_vs_posneg"]].sort_values(by="frec_negatif_vs_posneg", ascending=False)
        topic_to_show = top_topics[top_topics["frec_negatif_vs_posneg"]>0.51]
        fig_topics = go.Figure()

        fig_topics.add_trace(go.Bar(
            x=topic_to_show["frec_negatif_vs_posneg"],
            y=topic_to_show["labels"],
            orientation='h',
            marker=dict(color='red'),
            texttemplate='%{x:.2f}',  # Formater le texte pour afficher deux décimales
            textposition='auto',
            hovertemplate='%{y}: %{x} mentions<extra></extra>',
        ))

        fig_topics.update_layout(
            xaxis_title="Topic ratio",
            height=500,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20),
        )

        fig_topics.update_yaxes(autorange="reversed")  # Most frequent on top
        st.plotly_chart(fig_topics, use_container_width=True)
    
    # ============================== TOP COMMENTS ==============================
    # Initialiser l'état de la session pour le suivi des index de départ et des pages
    if 'positive_start_index' not in st.session_state:
        st.session_state.positive_start_index = 0
    if 'negative_start_index' not in st.session_state:
        st.session_state.negative_start_index = 0
    if 'positive_page' not in st.session_state:
        st.session_state.positive_page = 1
    if 'negative_page' not in st.session_state:
        st.session_state.negative_page = 1

    comment_col1, comment_col2 = st.columns(2)

    # Palette de couleurs par sentiment
    sentiment_styles = {
        "positive": {"bg": "#b7f7d0", "text": "#ffffff", "label": "👍 Positive Comments"},
        "neutral": {"bg": "#4b4b1e", "text": "#f9eec0", "label": "😐 Neutral Comments"},
        "negative": {"bg": "#ffb6b6", "text": "#ffffff", "label": "👎 Negative Comments"},
    }

    with comment_col1:
        style = sentiment_styles["positive"]
        st.markdown(f"#### {style['label']}")

        # Ajouter un menu déroulant pour sélectionner un sujet
        selected_top_topic = st.selectbox("Select a topic to filter the good comments", options=["All"] + labels, key="positive_topic")

        # Filtrer le DataFrame en fonction du sujet sélectionné
        if selected_top_topic != "All":
            topic_filtered_df = filtered_df[filtered_df[selected_top_topic] > seuil]
        else:
            topic_filtered_df = filtered_df

        top_pos_df = topic_filtered_df[topic_filtered_df["pred_sentiment"] == "positive"].sort_values(by='RoBERTa_score', ascending=False)
        top_pos = top_pos_df["review"].iloc[st.session_state.positive_start_index:st.session_state.positive_start_index+5]
        render_comments(top_pos, style["bg"], style["text"], chain, sentiment="positive")

        # Afficher le numéro de la page
        st.write(f"Page {st.session_state.positive_page}")

        # Ajouter des boutons pour charger les commentaires précédents et suivants
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.session_state.positive_page > 1:
                if st.button("Previous", key="positive_prev_button"):
                    st.session_state.positive_start_index -= 5
                    st.session_state.positive_page -= 1
                    st.rerun()
        with col2:
            if st.session_state.positive_start_index + 5 < len(top_pos_df):
                if st.button("Next", key="positive_next_button"):
                    st.session_state.positive_start_index += 5
                    st.session_state.positive_page += 1
                    st.rerun()

    with comment_col2:
        style = sentiment_styles["negative"]
        st.markdown(f"#### {style['label']}")

        # Ajouter un menu déroulant pour sélectionner un sujet
        selected_bad_topic = st.selectbox("Select a topic to filter the bad comments", options=["All"] + labels, key="negative_topic")

        # Filtrer le DataFrame en fonction du sujet sélectionné
        if selected_bad_topic != "All":
            topic_filtered_df = filtered_df[filtered_df[selected_bad_topic] > seuil]
        else:
            topic_filtered_df = filtered_df

        top_neg_df = topic_filtered_df[topic_filtered_df["pred_sentiment"] == "negative"].sort_values(by='RoBERTa_score', ascending=False)
        top_neg = top_neg_df["review"].iloc[st.session_state.negative_start_index:st.session_state.negative_start_index+5]
        render_comments(top_neg, style["bg"], style["text"], chain, sentiment="negative")

        # Afficher le numéro de la page
        st.write(f"Page {st.session_state.negative_page}")

        # Ajouter des boutons pour charger les commentaires précédents et suivants
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.session_state.negative_page > 1:
                if st.button("Previous", key="negative_prev_button"):
                    st.session_state.negative_start_index -= 5
                    st.session_state.negative_page -= 1
                    st.rerun()
        with col2:
            if st.session_state.negative_start_index + 5 < len(top_neg_df):
                if st.button("Next", key="negative_next_button"):
                    st.session_state.negative_start_index += 5
                    st.session_state.negative_page += 1
                    st.rerun()
# ============================== EMPTY STATE ==============================
if filtered_df.empty:
    st.warning("No reviews match the selected filters. Try adjusting them.")
