FROM continuumio/miniconda3

WORKDIR /app
COPY . /app

# Create environment and install dependencies
RUN conda update -n base -c defaults conda -y && \
    conda create -n app_env python=3.10 -y && \
    conda run -n app_env pip install --upgrade pip && \
    conda run -n app_env pip install \
        streamlit pandas plotly matplotlib seaborn \
        langchain langchain-mistralai python-dotenv

# Set env vars
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 7860

# Use conda run instead of source ~/.bashrc
CMD ["conda", "run", "-n", "app_env", "streamlit", "run", "app.py"]