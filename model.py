import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import pandas as pd
from constants import (
    TRENDS_NATIONAL_CSV,
    TRENDS_STATE_CSV,
    ILINET_NATIONAL_CSV,
    ILINET_STATE_CSV,
)
from pathlib import Path
from app_logger import setup_logger
from typing import Dict, Any, List
import pandas as pd
# import psycopg2 as pg
from sqlalchemy import create_engine
from decouple import config
import os
import streamlit as st


logging = setup_logger(__name__)

DATA_DIR = str(Path("/").resolve())


def smape(a, f):
    return 1 / len(a) * np.sum(2 * np.abs(f - a) / (np.abs(a) + np.abs(f)) * 100)


@st.cache_data
def get_db_engine():
    engine = pg.connect(config('DB_URL', default=os.environ.get("DB_URL")))
    return engine

# def fetch_data_from_db(terms: List, level: str, states: str = None) -> pd.DataFrame:
#     engine = pg.connect(config('DB_URL', default=os.environ.get("DB_URL")))
#     if level == "National":
#         trends_df = pd.read_sql("select date, cough, flu, tamiflu, sore_throat, week_number, year from influenza_gtrends WHERE level='National' ORDER BY year DESC, week_number DESC LIMIT 260", con=engine)
#         ilinet_df = pd.read_sql("select * from influenza_ilinet WHERE region_type='National' ORDER BY year DESC, week DESC LIMIT 260", con=engine)
#     else:
#         trends_df = pd.read_sql(f"select date, cough, flu, tamiflu, sore_throat, week_number, year from influenza_gtrends WHERE state='{states}' ORDER BY year DESC, week_number DESC LIMIT 260", con=engine)
#         ilinet_df = pd.read_sql(f"select * from influenza_ilinet WHERE region='{states}' ORDER BY year DESC, week DESC LIMIT 260", con=engine)

#     ilinet_df.drop(
#         ilinet_df.loc[ilinet_df["ilitotal"].isnull()].index, axis=0, inplace=True
#     )
#     terms.extend(["date", "week_number", "year"])
#     terms = ['sore_throat' if term == 'sore throat' else term for term in terms]        
#     trends_df = trends_df[terms]
#     df = pd.merge(
#         trends_df,
#         ilinet_df.loc[:, ["year", "week", "ilitotal"]],
#         left_on=["year", "week_number"],
#         right_on=["year", "week"],
#         how="inner",
#     ).drop(columns=["week_number", "year", "year", "week"])
#     logging.info(f"Combined data: {df.shape}")
#     return df

def fetch_data(terms: List, level: str, states: str = None) -> pd.DataFrame:
    try:
        # return fetch_data_from_db(terms, level, states)
        raise Exception("Database connection failed")  # Add a descriptive message
    except Exception as ex:
        print("----- failed to fetch from db ------")
        print(ex)
        return fetch_data_from_csv(terms, level, states)

def fetch_data_from_csv(terms: List, level: str, states: str = None) -> pd.DataFrame:
    """
    Currently the method scrapes the data on demand based
    on the params; and returns a df.
    Once we have a db, this method should fetch data from db
    based on the specified params; and return a df.
    The call to scrapers would be replaced by calls to db
    """
    if level == "National":
        trends_df = pd.read_csv(TRENDS_NATIONAL_CSV)
        ilinet_df = pd.read_csv(ILINET_NATIONAL_CSV, skiprows=[0], na_values="X")
    else:
        trends_df = pd.read_csv(TRENDS_STATE_CSV)
        trends_df = trends_df.loc[trends_df["state"] == states, :].drop(
            columns=["state", "state_code"]
        )
        ilinet_df = pd.read_csv(ILINET_STATE_CSV, skiprows=[0], na_values="X")
        ilinet_df = ilinet_df.loc[ilinet_df["REGION"] == states, :]

    ilinet_df.drop(
        ilinet_df.loc[ilinet_df["ILITOTAL"].isnull()].index, axis=0, inplace=True
    )
    terms.extend(["date", "week_number", "year"])
    trends_df = trends_df[terms]
    df = pd.merge(
        trends_df,
        ilinet_df.loc[:, ["YEAR", "WEEK", "ILITOTAL"]],
        left_on=["year", "week_number"],
        right_on=["YEAR", "WEEK"],
        how="inner",
    ).drop(columns=["week_number", "year", "YEAR", "WEEK"])
    logging.info(f"Combined data: {df.shape}")
    df = df.rename(columns={'ILITOTAL':'ilitotal'})
    return df



# @st.cache_data
def influenza_train_and_predict(
    data: pd.DataFrame, epochs: int, predict_ahead_by: int
) -> Dict[str, Any]:
    dates = data["date"].to_list()
    data.drop(columns=["date"], inplace=True)
    data = data.astype(float)
    pred_col = "ilitotal"
    batch_size = 256
    X, y = data.iloc[:-predict_ahead_by, :].copy(deep=True), data.iloc[
        predict_ahead_by:, :
    ].loc[:, [pred_col]].copy(deep=True)

    std = y[pred_col].std(ddof=0)
    mean = y[pred_col].mean()
    y[pred_col] = (y[pred_col] - mean) / std
    X[pred_col] = (X[pred_col] - mean) / std

    train_test_split = 0.75
    trainX, testX = (
        X[: round(X.shape[0] * train_test_split)].to_numpy(),
        X[round(X.shape[0] * train_test_split) :].to_numpy(),
    )
    trainy, testy = (
        y[: round(y.shape[0] * train_test_split)].to_numpy(),
        y[round(y.shape[0] * train_test_split) :].to_numpy(),
    )
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    trainy = np.reshape(trainy, (trainy.shape[0], trainy.shape[1], 1))
    testy = np.reshape(testy, (testy.shape[0], testy.shape[1], 1))

    model1 = Sequential()
    model1.add(LSTM(327, input_shape=(trainX.shape[1], trainX.shape[2])))
    model1.add(Dropout(rate=0.1))
    model1.add(Dense(1))

    model1.compile(loss="mse", optimizer="adam")
    logging.info(f"Finished compiling the model. Starting training")

    response = {}
    response["dates"] = dates

    history = model1.fit(
        trainX, trainy, batch_size=batch_size, epochs=epochs, shuffle=False
    )
    response["history"] = history
    pred = model1.predict(testX)
    pred_unnorm = (pred * std) + mean
    testy_unnorm = (testy * std) + mean

    response["predictions"] = pred_unnorm.reshape(testy.shape[0])
    response["actual_data"] = testy_unnorm.reshape(testy.shape[0])
    ci = smape(
        pred_unnorm.reshape(testy.shape[0]), testy_unnorm.reshape(testy.shape[0])
    )
    response["confidence_interval"] = ci

    # import pickle
    # with open('response.pkl', 'wb') as file:
    #     # A new file will be created
    #     pickle.dump(response, file)

    return response


# -------------- For testing ---------------
# df = fetch_data(['flu'], 'National', None)
# resp = influenza_train_and_predict(df, 2, 3)
# print(resp.keys())
