import streamlit as st
from PIL import Image
from model import fetch_data, influenza_train_and_predict
from model import smape
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from constants import STATE_CODE_MAPPER
import threading
import time
import requests

class KeepAlive:
    def __init__(self):
        self.running = False
        self.thread = None

    def keep_alive(self):
        while self.running:
            try:
                # Make request to your app URL
                requests.get("https://evision-python-app-gp4lssjsvbwz53dlnsem6m.streamlit.app/")
            except:
                pass
            time.sleep(60 * 10)  # Sleep for 10 minutes

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.keep_alive)
            self.thread.daemon = True
            self.thread.start()

# Initialize keep-alive in session state
if 'keep_alive' not in st.session_state:
    st.session_state.keep_alive = KeepAlive()
    st.session_state.keep_alive.start()

# Initialize session state for cache
if 'previous_predictions' not in st.session_state:
    st.session_state.previous_predictions = {}

INFLUENZA = "Influenza"

st.set_page_config(
    page_title="eVision",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

placeholder = st.empty()
predict = False
# if not predict:
if not predict:
    with placeholder.container():
        st.markdown(
            "<h1 style='text-align: center; color: grey;'>Psst! Hit Predict</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<h2 style='text-align: center; color: black;'>To see the model in action</h2>",
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        model_image = Image.open("model.png")
        with col1:
            st.write(" ")

        with col2:
            st.image(model_image)

        with col3:
            st.write(" ")


with st.sidebar:
    scu = Image.open("scu-icon.png")
    epiclab = Image.open("EpicLab-icon.png")
    cepheid = Image.open("cepheid.png")
    
    scu_height = scu.height
    epiclab_height = epiclab.height
    cepheid_height = cepheid.height
    
    max_height = max(scu_height, epiclab_height)
    padding_needed = (max_height - cepheid_height) // 2
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(scu, use_container_width=True)
        with col2:
            for _ in range(padding_needed // 10):
                st.write("")
            st.image(cepheid, use_container_width=True)
        with col3:
            st.image(epiclab, use_container_width=True)

            
    st.header("eVision")
    disease = st.selectbox(
        "**Pick disease**",
        [INFLUENZA],
        help="This input is where you select the disease that you would like to make a prediction for. The only option we currently have is influenza, but more diseases can be easily added to this framework if you would like to expand the scope of these predictions.",
    )
    if disease == INFLUENZA:
        pred_level = st.selectbox(
            "**Prediction Level**",
            ["National", "State"],
            help="This input allows you to select the level of the area you would like to predict. The options in this case are national and state, since these are the two levels that the WHO and CDC are able to provide us with. There has been work done by our team to add the ability to predict for cities since we have the Google data available for it, but there is no case data out there for us to reference for predictions.",
        )
        if pred_level == "National":
            states = None
        elif pred_level == "State":
            state_list = STATE_CODE_MAPPER.keys()
            states = st.selectbox("**Pick State**", state_list)
            # st.warning("Predictions on state level coming soon! Please select National level meanwhile.")
        terms = st.multiselect(
            "**Keywords**",
            ["cough", "flu", "tamiflu", "sore throat"],
            help="This is where you type in the keywords you would like us to extract from Google Trends",
        )
        predict = False
        with st.form(disease + "_train"):
            num_weeks = st.select_slider(
                "**Number of weeks prediction**",
                [3, 7, 14],
                help="This input allows you to dictate how many weeks ahead you would like to predict cases for. Generally, the higher amount of weeks you select, the less accurate your predictions become.",
            )
            epochs = st.slider(
                "**Number of Epochs**",
                min_value=1,
                max_value=200,
                step=1,
                help="This input is where you specify the number of epochs that you want to the machine learning model to use. Epochs are the number of iterations that a machine learning model goes through in its training. So, naturally, the higher you make this the more iterations it will go through, which improves the prediction. However, the higher number of iterations you do also increases the amount of time it will take the model to finish training.",
            )
            predict = st.form_submit_button("**Predict**")
            if predict:
                placeholder.empty()

# Create a cache key for data and model predictions
def create_cache_key(terms, pred_level, states, epochs=None, num_weeks=None):
    base_key = f"{'-'.join(sorted(terms))}__{pred_level}__{states}"
    if epochs is not None and num_weeks is not None:
        return f"{base_key}__{epochs}__{num_weeks}"
    return base_key

# Data fetching phase - using st.cache_data decorator in model.py
df = None
if disease == INFLUENZA and terms:
    # Check if we have all the required values before fetching data
    with st.spinner("Fetching the data..."):
        df = fetch_data(terms, pred_level, states)
        # The fetch_data function is now cached with @st.cache_data in model.py

# Model prediction phase - using both decorator caching and session state caching
response = None
if disease == INFLUENZA and predict and df is not None:
    # Create a unique cache key for this prediction
    cache_key = create_cache_key(terms, pred_level, states, epochs, num_weeks)
    
    # Check if we have this prediction cached in session state
    if cache_key in st.session_state.previous_predictions:
        response = st.session_state.previous_predictions[cache_key]
        st.success("Using cached prediction results")
    else:
        # If not in cache, use the cached function (which may still return quickly if parameters match)
        with st.spinner("Training the model..."):
            response = influenza_train_and_predict(df, epochs, num_weeks)
            
        # Store in session state cache for future use
        st.session_state.previous_predictions[cache_key] = response

    if response:
        st.header(f"{disease} Prediction results")

        ci = response.get("confidence_interval")
        results_df = pd.DataFrame(
            {
                "actual_data": response.get("actual_data"),
                "predictions": response.get("predictions"),
            }
        )

        results_df["week"] = range(1, len(results_df) + 1)
        results_df["predictions_upper"] = results_df["predictions"] + results_df["predictions"] * 0.01 * ci
        results_df["predictions_lower"] = results_df["predictions"] - results_df["predictions"] * 0.01 * ci
        fig = go.Figure()

        fig.add_trace(
            go.Line(name="Actual Data", x=results_df["week"], y=results_df["actual_data"])
        )

        fig.add_trace(
            go.Line(
                name="Predictions",
                x=results_df["week"],
                y=results_df["predictions"],
            )
        )

        fig.add_trace(
            go.Scatter(
                name="Upper Bound",
                x=results_df["week"],
                y=results_df["predictions_upper"],
                mode="lines",
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                name="Lower Bound",
                x=results_df["week"],
                y=results_df["predictions_lower"],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode="lines",
                fillcolor="rgba(68, 0, 0, 0.2)",
                fill="tonexty",
                showlegend=False,
            )
        )

        fig.update_layout(
            xaxis={"title": "Number of weeks"},
            yaxis={"title": "ILI Cases"},  # 'tickformat':'.2e'},
            title="<b>Influenza</b> Prediction",
            title_x=0.5,
        )

        st.plotly_chart(fig, theme=None, use_container_width=True)

        st.metric(
            "**Confidence interval**", f'{response.get("confidence_interval"):.5f}'
        )

        # Calculate SMAPE using the function already defined in model.py
        smape_value = smape(results_df["actual_data"].values, results_df["predictions"].values)

        # Display SMAPE metric
        st.metric(
            "**Error**", 
            f"{smape_value:.5f}", 
            help="Symmetric Mean Absolute Percentage Error"
        )

        history = response.get("history")
        # print(history.history["loss"])

        st.header("Epoch-Loss Graph")
        df_loss = pd.DataFrame(
            {
                "loss": history.history["loss"],
            }
        )

        df_loss["epoch"] = range(1, epochs + 1)
        fig = go.Figure()
        fig.add_trace(
            go.Line(
                name="Loss",
                x=df_loss["epoch"],
                y=df_loss["loss"],
            )
        )
        fig.update_layout(
            xaxis={"title": "Epoch"},
            yaxis={"title": "Loss"},  # 'tickformat':'.2e'},
            title="Epoch</b>VS</b>Loss",
            title_x=0.5,
        )
        st.plotly_chart(fig, theme=None, use_container_width=True)

# Add a button to clear cache if needed
with st.sidebar:
    if st.button("Clear Cache"):
        # Clear all caches
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.previous_predictions = {}
        st.success("Cache cleared successfully!")