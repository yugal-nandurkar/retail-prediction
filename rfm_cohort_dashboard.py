### Import Master Streamlit Libraries
import streamlit as st
#-------------------------------------------------------------------------------------------------------------------------

### Import Dashboard Libraries and Functional modules
import os
import csv
import datetime
from datetime import timedelta
import numpy as np
import lifetimes
from lifetimes import BetaGeoFitter
import pandas as pd
from lifetimes.datasets import load_transaction_data
from lifetimes.utils import summary_data_from_transaction_data, calibration_and_holdout_data
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from lifetimes.plotting import plot_period_transactions, plot_calibration_purchases_vs_holdout_purchases, plot_frequency_recency_matrix, plot_probability_alive_matrix, plot_period_transactions, plot_history_alive

from rfm_board import rfm_process, mean_median_std, split
from rfm_board import baseline_model, penalizer_score, hypertuner, hypertuned_bgf_model
from rfm_board import monval_freq_correlation, gg_fitter, ggf_tuner, hypertuned_ggf_model, ggfm
from rfm_board import clvm
from rfm_board import itertwo
from rfm_board import grab_cohorts

#-------------------------------------------------------------------------------------------------------------------------
### Imports For RFM Cohorts Based Dashboard and onwards
from rfm_board import thresh_prediction, met_freq_eval, cal_vs_hopu, rfm_frame, rbg
from rfm_board import frehis_plot, frequency_recency_matrix, probability_alive_matrix, cuorcohis_plot, period_transactions
from rfm_board import next_input_period_sales, customer_orders_mtone, check_cids_integrity, get_valid_cids
from rfm_board import annual_cohorts, custats
from rfm_board import plot_retention_matrix

#-------------------------------------------------------------------------------------------------------------------------

# Define dashboard
def dashboard():
    # Your code for dashboard2 goes here
    
    # Set column width to 100 pixels
    pd.set_option("display.max_colwidth", 1000)
    
    st.title("RFM Based Cohorts Dashboard")
    st.write("This dashboard is the continuation of Lifetimes Dashboard & Helps To Visualise Customer Insights.")
    
    # Load the data from the end user input (lifetimes_dashboard)
    
    # list of paths to check and remove
    retail_paths = ["C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df.csv", "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_prediction_period.py",                               "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_period.py", "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_discount_rate.py",                             "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_profit_margin.py", "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/customer_ids.csv"]  
    
    lifetimes_paths = ["C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df.csv", "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_prediction_period.py",                               "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_period.py", "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_discount_rate.py",                             "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_profit_margin.py"]          
    
    rfm_cohorts_paths = ["C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/customer_ids.csv"]
    
    #Delete Old Session (if requested)
    st.sidebar.markdown('<p style="color: red;">(Double Click Button At Discretion)</p>', unsafe_allow_html=True)
    if st.sidebar.button("New Master Retail Session "):
        for path in retail_paths:
            if any([os.path.exists(path)]):
                os.remove("C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df.csv")
                st.sidebar.success("Select The Lifetimes Dashboard From Navigation For Fresh Input Parameters")
                st.stop()        

    if all([os.path.exists(path) for path in lifetimes_paths]):
        df = pd.read_csv("C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df.csv")
        df_origins = pd.read_csv("C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_origins.csv")

        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.date
        st.sidebar.title("Welcome To Lifetimes Visuals")
        # Display dashboard
        st.subheader("Input Data")
        st.write(df)
    
        from df_prediction_period import prediction_period
        # Prompt user to input prediction period
        st.sidebar.success("Prediction Period Already Available From Lifetimes Dashboard Session. It is Global Parameter To         Suit Calculations As and When Required")        
        st.write("Enter Prediction Period in Sidebar ")
        default_value = prediction_period
        prediction_period = st.sidebar.number_input("Enter Prediction Period (in days)", value = prediction_period,                 min_value = 1)
        
        if prediction_period:
            # Save prediction_period to df_prediction_period.py
            dir_path = os.path.expanduser('~/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/')
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, 'df_prediction_period.py')
            with open(file_path, "w") as f:
                f.write(f"\nprediction_period = {int(prediction_period)} \n")
                # Show confirmation message to user
                st.sidebar.success("Prediction period updated") #saved to df_prediction_period.py")

#-----------------------------------------------------------------------------------------------------------------------                
### Lifetimes Dashboard Code:                
                
        #from df_prediction_period import prediction_period
        #if prediction_period:
            from df_period import calibration_period_end, observation_period_end
            if calibration_period_end and observation_period_end:
                last_df_invoicedate = datetime.date.fromisoformat(str(df['InvoiceDate'].iloc[-1]))

                if observation_period_end <= last_df_invoicedate and calibration_period_end < observation_period_end:
                    rfm = rfm_process(df)
                    rfm_stats = mean_median_std(rfm)
                    rfm_train_test = split(df)

                    ## baseline model
                    bgf = baseline_model(rfm_train_test)

                    #conditional expected number of purchases up to time
                    baseline_penalizer_score = penalizer_score(rfm_train_test, bgf)
                    if baseline_penalizer_score:
                        actual_freq, predicted_freq, score = baseline_penalizer_score

                    # Define Beta Geometric/Negative Binomial model's parameters
                    params = [0.0, 0.001, 0.1, 0.3683]

                    # Run Hyperparameter model and display results and get best penalizer and score
                    bgf, f_penalizer_df, f_best_penalizer_score, f_best_penalizer, pp_ts, pp_tm  =                                                   hypertuner(rfm_train_test, params, prediction_period)

                    hbgf, pp_hs, pp_hm, pph_pa = hypertuned_bgf_model(rfm_train_test, f_best_penalizer,                                         prediction_period)

#------------------------------------------------------------------------------------------------------------------------

                    mfc = monval_freq_correlation(rfm_train_test)

                    ggf = gg_fitter(rfm_train_test)

                    m_penalizer_df, m_best_penalizer_score, m_best_penalizer = ggf_tuner(rfm_train_test, ggf)

                    paov_tp, pp_tp, ptsr = hypertuned_ggf_model(rfm_train_test, m_best_penalizer)

                    mggf, ggfp, norecu, recuhe, recuprems, excoavpr, avepro = ggfm(rfm, rfm_train_test, m_best_penalizer)

#------------------------------------------------------------------------------------------------------------------------

                    annual_discount_rate_percentage, daily_discount_rate, dadirapr, wedirapr, modirapr, clv_rftt,                               clv_ptsrs, clv_lss = clvm(hbgf, mggf, rfm_train_test)

#------------------------------------------------------------------------------------------------------------------------

                    from df_profit_margin import profit_margin 
                    if profit_margin:  
                        rfm, iterbgf, iterbgf_fit, iterggf, iterclv, rfmclts= itertwo(rfm,                                                         f_best_penalizer, m_best_penalizer, daily_discount_rate)
                        rfm, chrtfm, prenxyr = grab_cohorts(df, rfm)
                        st.sidebar.write(f"Predicted Next Input Period ({prediction_period} Day(s)) Sales (Excluding New                           Customers)", prenxyr)

### Fresh RFM Cohorts Dashboard Code:
                        alive_purchasing_prediction = thresh_prediction(rfm, thresh=0.5)
                        st.subheader("Cohorts Based Alive Purchasing Prediction")
                        st.write(alive_purchasing_prediction)

                        prepusm, preneyesasm, copre, cltsasm, proalv, cltvsm, cucopre, cuprepur =                                                   next_input_period_sales(rfm)
                        st.write("Predicted Purchases ", prepusm)
                        st.write(f"Predicted Next Input Period ({prediction_period} Day(s)) sales ", preneyesasm)
                        st.write(copre)
                        st.write("Customer Lifetimes Sales (12 months)", cltsasm)
                        st.write("Number of Probable Customers Being Alive (50% Threshold)", proalv)
                        st.write("Customer Lifetimes Value ", cltvsm)
                        st.write("Probability Alive Cohort Bins", cucopre)
                        st.write("Predicted Purchases Cohort Bins",cuprepur)

#-------------------------------------------------------------------------------------------------------------------------
### Lifetimes Visuals

                        plt.style.use('seaborn-v0_8-pastel')
                        plt.rcParams.update({'font.size': 10})
                        plt.rcParams["axes.grid"] = 1
                        mpl.rcParams['lines.linewidth'] = 2
                       # st.set_option('deprecation.showPyplotGlobalUse', False)
                        # Set the backend of matplotlib to "Qt5Agg"
                        plt.switch_backend('Qt5Agg')
                        sns.set_theme(style="ticks", palette="pastel")
                        sns.set_context("poster")
                        sns.set(rc={'image.cmap': 'coolwarm'})

                        sns.set()
                        st.subheader("Frequency Histogram For Customer(Y) vs Purchases(X))")
                        frehis_fig = plt.figure(figsize=(12, 8))
                        frehis_ax = frehis_fig.add_subplot(111)
                        frehis_ax.set_xlim([0, 70])
                        frehis = frehis_plot(rfm)
                        st.pyplot(frehis_fig)

                        st.subheader("Frequency-Recency Matrix")
                        plfrecenma_fig = plt.figure(figsize=(8, 4))
                        plfrecenma_ax = plfrecenma_fig.add_subplot(111)
                        plfrecenma = frequency_recency_matrix(iterbgf)
                        st.pyplot(plfrecenma_fig)

                        st.subheader("Probability Alive Matrix")
                        plproalv_fig = plt.figure(figsize=(8, 4))
                        plproalv_ax = plproalv_fig.add_subplot(111)
                        plproalv = probability_alive_matrix(iterbgf)
                        st.pyplot(plproalv_fig)                        

                        norcu, muorpe = customer_orders_mtone(df)
                        st.write(norcu, "count of unique orders \n\n")
                        st.write(muorpe, "% of customers ordered more than once \n") 

                        st.subheader("Histogram of Customer Order Count")
                        cuorcohis_fig, cuorcohis_ax, cuorcohis = cuorcohis_plot(rfm)
                        plt.title("Histogram of Customer Order Count")
                        plt.ylabel("Number of Customers")
                        plt.xlabel("Order Count")
                        plt.xlim([0, 30])
                        plt.xticks(np.arange(0, 30, 2))
                        st.pyplot(cuorcohis_fig)

                        st.subheader("Frequency of Repeat Transactions")
                        pertr = period_transactions(iterbgf)
                        st.pyplot()

                        evatr, movaca = met_freq_eval(actual_freq, predicted_freq, rfm_train_test)
                        st.write("Metrics (MAE) Based Frequency Evaluation ", evatr)
                        st.write("Calibrated Monetary Value", movaca)

                        st.subheader("Actual Purchases in Holdout Period vs Predicted Purchases")
                        plcapuhopu = cal_vs_hopu(hbgf, rfm_train_test)
                        st.pyplot()

                        rfm_des = rfm_frame(rfm)
                        st.write(f"Description of RFM Frame", rfm_des)

                        customer_ids = check_cids_integrity(df)        

                        if customer_ids:
                            selected_cid, sp_trans, days_since_birth, cust_df_cids = get_valid_cids(customer_ids, df, rfm,                             f_best_penalizer, df_origins)
                            st.subheader(f"History Alive Plot for CustomerID: {selected_cid}")   
                            plhial_fig = plt.figure(figsize=(10, 6))
                            plhial_ax = plhial_fig.add_subplot(111)

                            # Use the best penalizer value to create the BGF model
                            #itertwo(func) 

                            #plot_history_alive(iterbgf_fit, days_since_birth, sp_trans, "InvoiceDate",                                                 ax=plhial_ax)
                            #st.pyplot(plhial_fig)

                            #itertwo(func)
                            st.subheader("Expected Conditional vs Actual Average Profit")
                            st.write(f"Expected conditional average profit: {excoavpr},  Average profit: {avepro}")

                            cohort_size, retention_matrix = annual_cohorts(df)
                            plrema_fig = plot_retention_matrix(cohort_size, retention_matrix)
                            plrema_fig.tight_layout()
                            st.pyplot(plrema_fig)

                            st.subheader(f"Best Customers (Predicted Next Input Period ({prediction_period} Day(s)) Sales                               Wise)")
                            becust = custats(rfm)
                            st.write(becust)
                            
                            st.session_state.dashboard_status = {'rfm_cohorts': True}

                            st.write("RFM Based Cohorts Dashboard is complete. You can now access the Retail Products                                   Dashboard.")                            

                            # Check if lifetimes dashboard is completed
                            if 'dashboard_status' not in st.session_state:
                                st.session_state.dashboard_status = {'rfm_cohorts': False}
                                st.write("RFM Based Cohorts Dashboard is in Progression...")                     
                                return
                            st.stop()
                            

                            

#------------------------------------------------------------------------------------------------------------------------   

#st.write(df.describe())
#st.write("This Dashboard is Under Construction")
# Else for if all([os.path.exists(path) for path in paths]):                        
    else:
        st.write("This Dashboard Activates Only Upon Entering Input Data For Analysis in Lifetimes Dashboard")
        st.stop()                        
