### Import Master Streamlit Libraries
import streamlit as st
#-------------------------------------------------------------------------------------------------------------------------

### Import Dashboard Libraries and Functional modules
st.set_page_config(page_title="Customer Lifetime Value Dashboard", page_icon=":chart_with_upwards_trend:")
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

#-------------------------------------------------------------------------------------------------------------------------

def read_and_process_file(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

        except pd.errors.EmptyDataError:
            st.error("Error: The file is empty or has no columns to parse.")
            return None
        else:
            st.write("")
            return df

# Define dashboard
def dashboard():
    # Your code for dashboard1 goes here
    st.title("Customer Lifetime Metrics Dashboard")
    st.write("This dashboard helps to visualize the customer lifetime value for a given customer and observation period.")
    
    # Reload the last session data from the end user input (lifetimes_dashboard)
    # list of paths to check and remove
    retail_paths = ["C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df.csv", "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_prediction_period.py",                               "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_period.py", "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_discount_rate.py",                             "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_profit_margin.py", "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/customer_ids.csv"]  
    
    lifetimes_paths = ["C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df.csv", "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_prediction_period.py",                               "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_period.py", "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_discount_rate.py",                             "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_profit_margin.py"]          

    #Delete Old Session (if requested)
    st.sidebar.markdown('<p style="color: red;">(Double Click Button At Discretion)</p>', unsafe_allow_html=True)
    if st.sidebar.button("New Master Retail Session"):
        for path in retail_paths:
            if any([os.path.exists(path)]):
                os.remove("C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df.csv")
                st.sidebar.success("Select The Lifetimes Dashboard From Navigation For Fresh Input Parameters")
                st.stop()  
    
    if all([os.path.exists(path) for path in lifetimes_paths]):           
        st.sidebar.title("Last Session")
        df = pd.read_csv("C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df.csv")
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.date

        # Display dashboard
        st.subheader("Input Data")
        st.write(df)

#------------------------------------------------------------------------------------------------------------------------
        from df_prediction_period import prediction_period
        if prediction_period:
            from df_period import calibration_period_end, observation_period_end
            if calibration_period_end and observation_period_end:
                last_df_invoicedate = datetime.date.fromisoformat(str(df['InvoiceDate'].iloc[-1]))
                st.write("Last Date from the provided Input is", last_df_invoicedate)
                if observation_period_end <= last_df_invoicedate and calibration_period_end < observation_period_end:

                    st.subheader("Description of Frequency, Recency, Customer Since (T) & Monetary Value")
                    rfm = rfm_process(df)
                    st.write(rfm)

                    st.subheader("Monetary Value Statistics")
                    rfm_stats = mean_median_std(rfm)
                    st.write(rfm_stats)
                    st.write(rfm.describe())

                    st.subheader("RFM Model Training: Calibration & Holdout Periods")
                    rfm_train_test = split(df)
                    st.write(rfm_train_test.head())
                    st.write((rfm_train_test.shape))

                    ## baseline model
                    bgf = baseline_model(rfm_train_test)
                    st.subheader("Basic Frequency/Recency analysis using the BG/NBD model")
                    st.write(bgf)
                    st.write("Transaction rate of the mass: r,α comes from the gamma distribution (buy process).")
                    st.write("Dropout rate of the mass: a,b comes from the beta distribution (till you die process).")
                    st.write(bgf.summary)
                    st.write("se(coef) indicates standard error coefficient")

                    st.subheader("Mean Absolute Error as Baseline Penalizer Score")
                    #conditional expected number of purchases up to time
                    baseline_penalizer_score = penalizer_score(rfm_train_test, bgf)
                    if baseline_penalizer_score:
                        actual_freq, predicted_freq, score = baseline_penalizer_score
                        st.write("Actual Average Frequency ", round(actual_freq.mean(), 4))
                        st.write("Predicted Average Frequency ", round(predicted_freq.mean(), 4))
                        st.write("MAE = ", round(score, 5))

                    st.subheader("Tuning Hyperparameters for frequency")
                    # Define Beta Geometric/Negative Binomial model's parameters
                    params = [0.0, 0.001, 0.1, 0.3683]

                    # Run Hyperparameter model and display results and get best penalizer and score
                    bgf, f_penalizer_df, f_best_penalizer_score, f_best_penalizer, pp_ts, pp_tm  =                                                   hypertuner(rfm_train_test, params, prediction_period)
                    st.write(f_penalizer_df)
                    st.write("Best Penalizer: ", f_best_penalizer)
                    st.write("Best Penalizer Score: ", f_best_penalizer_score)
                    st.write("Predicted purchases for test period ", pp_ts)
                    st.write("Test period Mean predicted purchases", pp_tm)

                    hbgf, pp_hs, pp_hm, pph_pa = hypertuned_bgf_model(rfm_train_test, f_best_penalizer,                                         prediction_period)
                    st.write(hbgf)
                    st.write(hbgf.summary)
                    st.write("Predicted purchases for holdout period ", pp_hs)
                    st.write("Holdout period mean predicted purchases", pp_hm)
                    st.write("Holdout period's predicted mean probability of customer being alive", pph_pa)

#------------------------------------------------------------------------------------------------------------------------

                    st.subheader("Estimating customer lifetime value using the Gamma-Gamma model")
                    mfc = monval_freq_correlation(rfm_train_test)
                    st.write(" Confirm no correlation between calibration's monetary value and frequency", mfc)

                    ggf = gg_fitter(rfm_train_test)
                    st.write(ggf)
                    st.write("p: shape parameter of 1st distro, q: shape parameter of 2nd distro, r: rate parameter of                         distro")
                    st.write("Gamma(p, v) defines the distribution of a customer's observed average transaction value")

                    st.subheader("Tuning Hyperparameters for monetary value")
                    m_penalizer_df, m_best_penalizer_score, m_best_penalizer = ggf_tuner(rfm_train_test, ggf)
                    st.write(m_penalizer_df)
                    st.write("best penalizer:", m_best_penalizer)
                    st.write("best penalizer score:", m_best_penalizer_score)

                    paov_tp, pp_tp, ptsr = hypertuned_ggf_model(rfm_train_test, m_best_penalizer)
                    st.write("Predicted avg order value for test period ", paov_tp)
                    st.write("Predicted purchases for test period ", pp_tp)
                    st.write("Predicted test sales revenue", ptsr)

                    mggf, ggfp, norecu, recuhe, recuprems, excoavpr, avepro = ggfm(rfm, rfm_train_test, m_best_penalizer)
                    st.subheader("Fit Monetary Value to Best GGF Model")
                    st.write(mggf)
                    st.write(ggfp)
                    st.write("Number of returning customers ", norecu)
                    st.write(recuhe)
                    st.write("Sum of Predicted Monetary Value", recuprems)

#------------------------------------------------------------------------------------------------------------------------
                    from df_discount_rate import annual_discount_rate_percentage
                    if annual_discount_rate_percentage:
                        st.subheader("Customer Lifetime Value Model")
                        annual_discount_rate_percentage, daily_discount_rate, dadirapr, wedirapr, modirapr, clv_rftt,                               clv_ptsrs, clv_lss = clvm(hbgf, mggf, rfm_train_test)
                        st.write("Scroll sideways to access all features", clv_rftt)
                        st.write("Sum of predicted test sales revenue", clv_ptsrs)
                        st.write("Lifetime Sales ", clv_lss)

#------------------------------------------------------------------------------------------------------------------------
                        st.subheader("RFM Based Cohorts")
                        from df_profit_margin import profit_margin 
                        if profit_margin:
                            rfm, iterbgf, iterbgf_fit, iterggf, iterclv, rfmclts = itertwo(rfm,                                                         f_best_penalizer, m_best_penalizer, daily_discount_rate)
                            st.write("Average Customer Lifetime Sales", rfmclts)

                            rfm, chrtfm, prenxyr = grab_cohorts(df, rfm)
                            st.write("Statistics of RFM Based Cohorts (Scroll Sideways)", chrtfm)
                            st.write("Predicted Next Input Period Sales (Excluding New Customers)", prenxyr)
                            st.write("Predicting Recurrent Revenue With New Customers Added Will Require TIME-SERIES                                   Introspection as followed in Other Available Dashboards in This Web Application")
                            st.session_state.dashboard_status = {'lifetimes': True}

                            st.write("Lifetimes Dashboard is complete. You can now access the RFM Based Cohorts                                         Dashboard.")
                            st.write("This was the quick introduction of RFM Based Cohorts Dashboard which is now available                             in the Menu")                            

                            # Check if lifetimes dashboard is completed
                            if 'dashboard_status' not in st.session_state:
                                st.session_state.dashboard_status = {'lifetimes': False}
                                st.warning("Lifetimes Dashboard is in Progression...")                     
                                return
                            st.stop()

#-----------------------------------------------------------------------------------------------------------------------   ### Else for if all([os.path.exists(path) for path in paths]):

##### NEW SESSION                     
    else: 
        df_path = "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df.csv"
        # Upload and preprocess data
        st.sidebar.title("Input Features")
        st.write("Upload Input Features in Sidebar (Expected Features are 'CustomerID', 'InvoiceDate', and 'Sales')")        
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.date

            st.sidebar.warning("Previous Data Restored. Expect previous inputs unless set otherwise above")
                
        else:
            uploaded_file = st.sidebar.file_uploader("Choose a file")
            df = read_and_process_file(uploaded_file)
            if df is not None:
                df_origins = df.copy()
                    
                df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.date
                # Save user input data to a CSV file
                df.to_csv("C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df.csv", index=False)
                
                df_origins.drop(["InvoiceType"], axis=1, inplace=True) #, ["Log_Sales"]
                df_origins.to_csv("C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_origins.csv", index=False)
                st.sidebar.success("New Data Updated")
            else:
                st.write("")
            
        if os.path.exists(df_path):
            # Check if data has correct shape
            if 'CustomerID' not in df.columns or 'InvoiceDate' not in df.columns or 'Sales' not in df.columns:
                st.warning("The uploaded file does not have the expected columns ('CustomerID', 'InvoiceDate', and/or                       'Sales'). Loading sample transaction data instead.")
                df = load_transaction_data()
                # Display dashboard
                st.subheader("Input Data")
                st.write(df)
            else:
                st.write(df)
                #st.write(df_origins)
                # Prompt user to input prediction period
                st.write("Enter Prediction Period in Sidebar ")
                prediction_period = st.sidebar.number_input("Enter Prediction Period (in days)")
                if prediction_period:
                    # Save prediction_period to df_prediction_period.py
                    dir_path = os.path.expanduser('~/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/')
                    os.makedirs(dir_path, exist_ok=True)
                    file_path = os.path.join(dir_path, 'df_prediction_period.py')
                    with open(file_path, "w") as f:
                        f.write(f"\nprediction_period = {int(prediction_period)} \n")
                        # Show confirmation message to user
                        st.sidebar.success("Prediction period updated") #saved to df_prediction_period.py")
                
                    # Prompt user to input calibration and observation periods
                    st.write("Enter Calibration Period End Date and Observation Period End Date in Sidebar")
                    st.sidebar.subheader("Set Calibration and Holdout Periods")
                    calibration_period_end = st.sidebar.date_input("Calibration Period End Date")
                    observation_period_end = st.sidebar.date_input("Observation Period End Date")   

                    last_df_invoicedate = datetime.date.fromisoformat(str(df['InvoiceDate'].iloc[-1]))
                    st.write("Last Date from the provided Input is", last_df_invoicedate)

                    if calibration_period_end > observation_period_end:
                        st.sidebar.warning("Calibration Period End Date Can't Be Later Than Observation Period End Date") 

                    if observation_period_end > last_df_invoicedate:
                        st.sidebar.warning("Observation Period End Date Can't Be Later Than Last Date from the provided                             Input")                    
                    capeobpen = calibration_period_end < observation_period_end
                    obpenladi = observation_period_end <= last_df_invoicedate

                    if not capeobpen and obpenladi:
                        st.warning("Calibration Period End Date should be earlier than Observation Period End Date,                                 Moreover Observation Period End Date should be the Last Date from the provided Input or                                     Earlier")

                    if calibration_period_end and observation_period_end and capeobpen and obpenladi:
                        # Save calibration_period_end and observation_period_end to df_period.py
                        dir_path = os.path.expanduser('~/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/')
                        os.makedirs(dir_path, exist_ok=True)
                        file_path = os.path.join(dir_path, 'df_period.py')
                        with open(file_path, "w") as f:
                            f.write("import datetime \n")
                            f.write("calibration_period_end = datetime.date({}, {}, {}) \n".format(
                            calibration_period_end.year, calibration_period_end.month, calibration_period_end.day))
                            f.write("observation_period_end = datetime.date({}, {}, {}) \n".format(
                            observation_period_end.year, observation_period_end.month, observation_period_end.day))
                            # Show confirmation message to user
                            st.sidebar.success("Calibration and observation period end dates updated") #saved to                                         df_period.py")

#------------------------------------------------------------------------------------------------------------------------
                        from df_period import calibration_period_end, observation_period_end

                        st.subheader("Description of Frequency, Recency, Customer Since (T) & Monetary Value")
                        rfm = rfm_process(df)
                        st.write(rfm)

                        st.subheader("Monetary Value Statistics")
                        rfm_stats = mean_median_std(rfm)
                        st.write(rfm_stats)
                        st.write(rfm.describe())

                        st.subheader("RFM Model Training: Calibration & Holdout Periods")
                        rfm_train_test = split(df)
                        st.write(rfm_train_test.head())
                        st.write((rfm_train_test.shape))

                        ## baseline model
                        bgf = baseline_model(rfm_train_test)
                        st.subheader("Basic Frequency/Recency analysis using the BG/NBD model")
                        st.write(bgf)
                        st.write("Transaction rate of the mass: r,α comes from the gamma distribution (buy process).")
                        st.write("Dropout rate of the mass: a,b comes from the beta distribution (till you die                                     process).")
                        st.write(bgf.summary)
                        st.write("se(coef) indicates standard error coefficient")

                        st.subheader("Mean Absolute Error as Baseline Penalizer Score")

                        #conditional expected number of purchases up to time
                        baseline_penalizer_score = penalizer_score(rfm_train_test, bgf)
                        if baseline_penalizer_score:
                            actual_freq, predicted_freq, score = baseline_penalizer_score
                            st.write("Actual Average Frequency ", round(actual_freq.mean(), 4))
                            st.write("Predicted Average Frequency ", round(predicted_freq.mean(), 4))
                            st.write("MAE = ", round(score, 5))

                        st.subheader("Tuning Hyperparameters for frequency")
                        # Define Beta Geometric/Negative Binomial model's parameters
                        params = [0.0, 0.001, 0.1, 0.3683]
                        #from df_prediction_period import prediction_period

                        # Run Hyperparameter model and display results and get best penalizer and score
                        bgf, f_penalizer_df, f_best_penalizer_score, f_best_penalizer, pp_ts, pp_tm  =                                                   hypertuner(rfm_train_test, params, prediction_period)
                        st.write(f_penalizer_df)
                        st.write("Best Penalizer: ", f_best_penalizer)
                        st.write("Best Penalizer Score: ", f_best_penalizer_score)
                        st.write("Predicted purchases for test period ", pp_ts)
                        st.write("Test period Mean predicted purchases", pp_tm)

                        hbgf, pp_hs, pp_hm, pph_pa = hypertuned_bgf_model(rfm_train_test, f_best_penalizer,                                         prediction_period)
                        st.write(hbgf)
                        st.write(hbgf.summary)
                        st.write("Predicted purchases for holdout period ", pp_hs)
                        st.write("Holdout period mean predicted purchases", pp_hm)
                        st.write("Holdout period's predicted mean probability of customer being alive", pph_pa)

#------------------------------------------------------------------------------------------------------------------------

                        st.subheader("Estimating customer lifetime value using the Gamma-Gamma model")
                        mfc = monval_freq_correlation(rfm_train_test)
                        st.write(" Confirm no correlation between calibration's monetary value and frequency", mfc)

                        ggf = gg_fitter(rfm_train_test)
                        st.write(ggf)
                        st.write("p: shape parameter of 1st distro, q: shape parameter of 2nd distro, r: rate                                       parameter of distro")
                        st.write("Gamma(p, v) defines the distribution of a customer's observed average transaction                                 value")

                        st.subheader("Tuning Hyperparameters for monetary value")
                        m_penalizer_df, m_best_penalizer_score, m_best_penalizer = ggf_tuner(rfm_train_test, ggf)
                        st.write(m_penalizer_df)
                        st.write("best penalizer:", m_best_penalizer)
                        st.write("best penalizer score:", m_best_penalizer_score)

                        paov_tp, pp_tp, ptsr = hypertuned_ggf_model(rfm_train_test, m_best_penalizer)
                        st.write("Predicted avg order value for test period ", paov_tp)
                        st.write("Predicted purchases for test period ", pp_tp)
                        st.write("Predicted test sales revenue", ptsr)

                        mggf, ggfp, norecu, recuhe, recuprems, excoavpr, avepro = ggfm(rfm, rfm_train_test,                                         m_best_penalizer)
                        st.subheader("Fit Monetary Value to Best GGF Model")
                        st.write(mggf)
                        st.write(ggfp)
                        st.write("Number of returning customers ", norecu)
                        st.write(recuhe)
                        st.write("Sum of Predicted Monetary Value", recuprems)

#------------------------------------------------------------------------------------------------------------------------

                       # Prompt user to input annual discount rate percentage
                        st.write("Enter Annual Discount Rate Percentage in Sidebar ")
                        annual_discount_rate_percentage = st.sidebar.number_input("Enter Annual Discount Rate (in                                   percentage)")
                        if annual_discount_rate_percentage:
                            # Save prediction_period to df_prediction_period.py
                            dir_path = os.path.expanduser('~/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/')
                            os.makedirs(dir_path, exist_ok=True)
                            file_path = os.path.join(dir_path, 'df_discount_rate.py')
                            with open(file_path, "w") as f:
                                f.write(f"\nannual_discount_rate_percentage = {float(annual_discount_rate_percentage)} \n") 

                            st.subheader("Customer Lifetime Value Model")
                            annual_discount_rate_percentage, daily_discount_rate, dadirapr, wedirapr, modirapr, clv_rftt,                               clv_ptsrs, clv_lss = clvm(hbgf, mggf, rfm_train_test)
                            # Show confirmation message to user
                            st.sidebar.success(f"Discount rates updated as Daily:{dadirapr}%, Weekly: {wedirapr}%,                                     Monthly: {modirapr}%, Annual: {annual_discount_rate_percentage}%")
                            #saved to df_discount_rate.py"  
                            
                            st.write("Scroll sideways to access all features", clv_rftt)
                            st.write("Sum of predicted test sales revenue", clv_ptsrs)
                            st.write("Lifetime Sales ", clv_lss)

#------------------------------------------------------------------------------------------------------------------------
                            st.subheader("RFM Based Cohorts")
                            st.write("Enter Profit Margin in Sidebar (Scroll Down)")
                            # Prompt user to input profit margin
                            profit_margin = st.sidebar.number_input("Enter Profit Margin (in percentage)")
                            if profit_margin:
                                # Save profit_margin to df_profit_margin.py
                                dir_path = os.path.expanduser('~/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/')
                                os.makedirs(dir_path, exist_ok=True)
                                file_path = os.path.join(dir_path, 'df_profit_margin.py')
                                with open(file_path, "w") as f:
                                    f.write(f"\nprofit_margin = {float(profit_margin)} \n")
                                    # Show confirmation message to user
                                    st.sidebar.success("Profit margin updated") #saved to df_profit_margin.py")

                                #from df_profit_margin import profit_margin     
                                rfm, iterbgf, iterbgf_fit, iterggf, iterclv, rfmclts = itertwo(rfm, f_best_penalizer,                                       m_best_penalizer, daily_discount_rate)
                                st.write("Average Customer Lifetime Sales", rfmclts)

                                rfm, chrtfm, prenxyr = grab_cohorts(df, rfm)
                                st.write("Statistics of RFM Based Cohorts (Scroll Sideways)", chrtfm)
                                st.write(f"Predicted Next Input Period ({prediction_period} Day(s)) Sales (Excluding New                                   Customers)", prenxyr)
                                st.write("Predicting Recurrent Revenue With New Customers Added Will Require TIME-                                         SERIES Introspection as followed in Other Available Dashboards in This Web                                                 Application")
                                st.session_state.dashboard_status = {'lifetimes': True}

                                st.write("Lifetimes Dashboard is complete. You can now access the RFM Based Cohorts                                         Dashboard.")
                                st.write("This was the quick introduction of RFM Based Cohorts Dashboard which is now                                       available in the Menu")                            

                                # Check if lifetimes dashboard is completed
                                if 'dashboard_status' not in st.session_state:
                                    st.session_state.dashboard_status = {'lifetimes': False}
                                    st.write("Lifetimes Dashboard is in Progression...")                     
                                    return
                                st.stop()

#------------------------------------------------------------------------------------------------------------------------   


     
