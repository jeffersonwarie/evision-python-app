import streamlit as st
from PIL import Image
from model import fetch_data, influenza_train_and_predict
import plotly.graph_objects as go
import pandas as pd
from constants import STATE_CODE_MAPPER


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
        model_image = Image.open("evision/media/model.png")
        with col1:
            st.write(" ")

        with col2:
            st.image(model_image)

        with col3:
            st.write(" ")


with st.sidebar:
    scu = Image.open("/media/scu-icon.png")
    epiclab = Image.open("/media/EpicLab-icon.png")
    cepheid = Image.open("/media/cepheid.png")
    st.image(scu)
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.image(cepheid)
        with col2:
            st.image(epiclab)
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

# data fetching pahse:
if disease == INFLUENZA:
    # check if We have all the required values. Only start data fetching after that
    with st.spinner("Fetching the data..."):
        df = fetch_data(terms, pred_level, states)

# model prediction phase:
response = None
if disease == INFLUENZA:
    if predict:
        # placeholder.empty()
        with st.spinner("Training the model..."):
            response = influenza_train_and_predict(df, epochs, num_weeks)

        if response:
            st.header(f"{disease} Prediction results")

            ci = response.get("confidence_interval")
            df = pd.DataFrame(
                {
                    "actual_data": response.get("actual_data"),
                    "predictions": response.get("predictions"),
                }
            )

            df["week"] = range(1, len(df) + 1)
            df["predictions_upper"] = df["predictions"] + df["predictions"] * 0.01 * ci
            df["predictions_lower"] = df["predictions"] - df["predictions"] * 0.01 * ci
            fig = go.Figure()

            fig.add_trace(
                go.Line(name="Actual Data", x=df["week"], y=df["actual_data"])
            )

            fig.add_trace(
                go.Line(
                    name="Predictions",
                    x=df["week"],
                    y=df["predictions"],
                )
            )

            fig.add_trace(
                go.Scatter(
                    name="Upper Bound",
                    x=df["week"],
                    y=df["predictions_upper"],
                    mode="lines",
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    name="Lower Bound",
                    x=df["week"],
                    y=df["predictions_lower"],
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
            history = response.get("history")
            # print(history.history["loss"])

            st.header("Epoch-Loss Graph")
            df = pd.DataFrame(
                {
                    "loss": history.history["loss"],
                }
            )

            df["epoch"] = range(1, epochs + 1)
            fig = go.Figure()
            fig.add_trace(
                go.Line(
                    name="Loss",
                    x=df["epoch"],
                    y=df["loss"],
                )
            )
            fig.update_layout(
                xaxis={"title": "Epoch"},
                yaxis={"title": "Loss"},  # 'tickformat':'.2e'},
                title="Epoch</b>VS</b>Loss",
                title_x=0.5,
            )
            st.plotly_chart(fig, theme=None, use_container_width=True)



# if terms


# with st.spinner("Training the model..."):
#     if disease == 'Influenza':
#         response = influenza_train_and_predict(df, epochs, num_weeks)

# st.text("Model trained successfully !")

# disease
# # of weeks prediction
# Prediction leve: National
# Level
# Sublevel
# Keywords
# Epochs
# predict
