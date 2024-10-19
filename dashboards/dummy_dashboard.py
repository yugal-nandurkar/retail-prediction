### Import Libraries and Functional modules

import streamlit as st
#st.set_page_config(page_title="Second Dashboard", page_icon=":chart_with_upwards_trend:")
import os
import datetime
import numpy as np
import lifetimes
from lifetimes import BetaGeoFitter
import pandas as pd
from lifetimes.datasets import load_transaction_data
from lifetimes.utils import summary_data_from_transaction_data, calibration_and_holdout_data
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes.plotting import plot_period_transactions, plot_calibration_purchases_vs_holdout_purchases, plot_frequency_recency_matrix, plot_probability_alive_matrix, plot_period_transactions, plot_history_alive

from rfm_board import rfm_process, mean_median_std, split
from rfm_board import baseline_model, penalizer_score, hypertuner, hypertuned_bgf_model
from rfm_board import monval_freq_correlation, gg_fitter, ggf_tuner, hypertuned_ggf_model, ggfm
from rfm_board import clvm
from rfm_board import itertwo
from rfm_board import grab_cohorts

#------------------------------------------------------------------------------------------------------------------------

st.title("RFM Cohort Dashboard")
st.write("This dashboard is is under construction.")
    
# Define the function to display the second dashboard
def dashboard():
    st.subheader("Dummy Returns")
    # Your code for Dashboard 2 goes here, using data1_input as input
    # Define the data for the second dashboard
    data2 = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100),
        'z': np.random.randn(100)
    })
    st.write(data2)

