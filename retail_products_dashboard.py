### Import Master Streamlit Libraries
import streamlit as st
#-------------------------------------------------------------------------------------------------------------------------

### Import Dashboard Libraries and Functional modules
import os
import csv
import datetime
#from datetime import datetime
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
from rfm_board import get_pps_sum_for_date, listed_products, max_product_purchase

#-------------------------------------------------------------------------------------------------------------------------

# Define dashboard
def dashboard():
    # Your code for dashboard2 goes here
    
    # Set column width to 100 pixels
    pd.set_option("display.max_colwidth", 1000)
    
    st.title("Retail Products Dashboard")
    st.write("This dashboard is the continuation of RFM Based Cohorts Dashboard & Helps To Visualise Product Insights.")
    
    # Load the data from the end user input (lifetimes_dashboard)
    
    # list of paths to check and remove
    retail_paths = ["C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df.csv",                                                                 "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_period.py", "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_discount_rate.py",             "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_profit_margin.py", "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/customer_ids.csv"]  
    
    lifetimes_paths = ["C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df.csv",                                                             "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_prediction_period.py",                                                               "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_period.py", "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_discount_rate.py",             "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_profit_margin.py"]          
    
    rfm_cohorts_paths = ["C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/customer_ids.csv",                                                 "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/ouput.csv"]
    
    retail_product_paths = []
    
    #Delete Old Session (if requested)
    st.sidebar.markdown('<p style="color: red;">(Double Click Button At Discretion)</p>', unsafe_allow_html=True)
    if st.sidebar.button("New Master Retail Session"):
        for path in retail_paths:
            if any([os.path.exists(path)]):
                os.remove("C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df.csv")
                st.sidebar.success("Select The Lifetimes Dashboard From Navigation For Fresh Input Parameters")
                st.stop()        

    if all([os.path.exists(path) for path in lifetimes_paths]):
        df = pd.read_csv("C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df.csv")
        df_origins = pd.read_csv("C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/df_origins.csv")

        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.date
        st.sidebar.title("Retail Products Interactives")
        # Display dashboard
        st.subheader("Input Data")
        st.write(df)
    
        from df_prediction_period import prediction_period
        # Prompt user to input prediction period
        st.sidebar.success("Prediction Period Already Available From Lifetimes Dashboard Session. It is Global Parameter To             Suit Calculations As and When Required")        
        st.write("Edit Prediction Period in Sidebar ")
        default_value = prediction_period
        prediction_period = st.sidebar.number_input("Edit Prediction Period (in days)", value = prediction_period,                     min_value = 1)
        
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
                    bgf, f_penalizer_df, f_best_penalizer_score, f_best_penalizer, pp_ts, pp_tm  =                                                 hypertuner(rfm_train_test, params, prediction_period)

                    hbgf, pp_hs, pp_hm, pph_pa = hypertuned_bgf_model(rfm_train_test, f_best_penalizer,                                             prediction_period)

#------------------------------------------------------------------------------------------------------------------------

                    mfc = monval_freq_correlation(rfm_train_test)

                    ggf = gg_fitter(rfm_train_test)

                    m_penalizer_df, m_best_penalizer_score, m_best_penalizer = ggf_tuner(rfm_train_test, ggf)

                    paov_tp, pp_tp, ptsr = hypertuned_ggf_model(rfm_train_test, m_best_penalizer)

                    mggf, ggfp, norecu, recuhe, recuprems, excoavpr, avepro = ggfm(rfm, rfm_train_test, m_best_penalizer)

#------------------------------------------------------------------------------------------------------------------------

                    annual_discount_rate_percentage, daily_discount_rate, dadirapr, wedirapr, modirapr, clv_rftt,                                   clv_ptsrs, clv_lss = clvm(hbgf, mggf, rfm_train_test)

#------------------------------------------------------------------------------------------------------------------------

                    from df_profit_margin import profit_margin 
                    if profit_margin:  
                        rfm, iterbgf, iterbgf_fit, iterggf, iterclv, rfmclts= itertwo(rfm,                                                             f_best_penalizer, m_best_penalizer, daily_discount_rate)
                        rfm, chrtfm, prenxyr = grab_cohorts(df, rfm)
                        st.sidebar.write(f"Predicted Next Input Period ({prediction_period} Day(s)) Sales (Excluding New                           Customers)", prenxyr)

### RFM Cohorts Dashboard Code:
                        #alive_purchasing_prediction = thresh_prediction(rfm, thresh=0.5)
                        #prepusm, preneyesasm, copre, cltsasm, proalv, cltvsm, cucopre, cuprepur =                                                   next_input_period_sales(rfm)

#-------------------------------------------------------------------------------------------------------------------------
### Lifetimes Visuals

                        plt.style.use('seaborn-v0_8-pastel')
                        plt.rcParams.update({'font.size': 10})
                        plt.rcParams["axes.grid"] = 1
                        mpl.rcParams['lines.linewidth'] = 2
                        #st.set_option('deprecation.showPyplotGlobalUse', False)
                        # Set the backend of matplotlib to "Agg" #Qt5Agg
                        plt.switch_backend('Qt5Agg')
                        sns.set_theme(style="ticks", palette="pastel")
                        sns.set_context("poster")
                        sns.set(rc={'image.cmap': 'coolwarm'})
                        sns.set()
                        
                        """
                        #frehis = frehis_plot(rfm)    
                        #plfrecenma = frequency_recency_matrix(iterbgf)
                        #plproalv = probability_alive_matrix(iterbgf)
                        #norcu, muorpe = customer_orders_mtone(df)
                        #cuorcohis_fig, cuorcohis_ax, cuorcohis = cuorcohis_plot(rfm)
                        #pertr = period_transactions(iterbgf)
                        #evatr, movaca = met_freq_eval(actual_freq, predicted_freq, rfm_train_test)
                        #plcapuhopu = cal_vs_hopu(hbgf, rfm_train_test)
                        #rfm_des = rfm_frame(rfm)
                        #customer_ids = check_cids_integrity(df)        
                        #selected_cid, sp_trans, days_since_birth, cust_df_cids = get_valid_cids(customer_ids, df, rfm,                                 f_best_penalizer, df_origins) 
                        # Use the best penalizer value to create the BGF model
                        #itertwo(func) 
                        #plot_history_alive(iterbgf_fit, days_since_birth, sp_trans, "InvoiceDate",                                                     ax=plhial_ax)
                        #cohort_size, retention_matrix = annual_cohorts(df)
                        #plrema_fig = plot_retention_matrix(cohort_size, retention_matrix)
                        #becust = custats(rfm) """

                        rbg(rfm, f_best_penalizer, m_best_penalizer,                                                                                   daily_discount_rate, prediction_period, iterbgf, iterbgf_fit, iterggf)
                        
                        selected_month, selected_day, sales, date_input = get_pps_sum_for_date(df)
                        # Print sales for the same day and month in earlier years
                        st.write(f"Sales for {selected_month}/{selected_day} in earlier years:", sales)
                        
                        st.subheader("Products List for Retail")
                        product_count, sales_by_product = listed_products(df)
                        
                        st.subheader("RFM Frame")
                        st.write(f"Predicted Purchases for {date_input}")
                        st.write(rfm["pre_pur_on_selected_date"].sum())
                        st.write(f"Predicted Purchases for {prediction_period} Day(s)")
                        st.write(rfm["predicted_purchases"].sum())
                        st.write(f"Predicted Sales for {date_input}")
                        st.write(rfm["preneinpesa_on_selected_date"].sum())
                        st.write(f"Mean Predicted Sales by Average Customer for {date_input}")
                        st.write(rfm["preneinpesa_on_selected_date"].mean())
                        st.write(rfm)
                        
                        if product_count:
                            # Generate a bar chart showing sales by product for the top products
                            st.subheader('Sales by Product')
                            chart_data = sales_by_product.sort_values(by='Sales', ascending=False).head(product_count)
                            toprsa_fig, toprsa_ax = plt.subplots(figsize=(10,6))
                            toprsa_ax = sns.barplot(x='Description', y='Sales', data=chart_data)
                            toprsa_ax.set_xticklabels(toprsa_ax.get_xticklabels(), rotation=45,                                                             horizontalalignment='right')
                            toprsa_ax.set_title(f'Top {product_count} Products by Sales')
                            toprsa_ax.set_xlabel('Product Description')
                            toprsa_ax.set_ylabel('Sales')
                            st.pyplot(toprsa_fig)
                            
                            st.subheader("Sales by Predicted Products")
                            predicted_sales_by_product = max_product_purchase(df, rfm, product_count)
                            
                            chart_data = predicted_sales_by_product.sort_values(by='predicted_sales_by_product',                                           ascending=False).head(product_count)
                            presabypro_fig, presabypro_ax = plt.subplots(figsize=(10,6))
                            presabypro_ax = sns.barplot(x='Description', y='predicted_sales_by_product', data=chart_data)
                            presabypro_ax.set_xticklabels(presabypro_ax.get_xticklabels(), rotation=45,                                                     horizontalalignment='right')
                            presabypro_ax.set_title(f'Predicted Sales of Top {product_count} Products')
                            presabypro_ax.set_xlabel('Product Description')
                            presabypro_ax.set_ylabel('Predicted Sales by Product')
                            st.pyplot(presabypro_fig)
                                              
                            st.subheader("Test for Expected Purchase")
                            chart_data = predicted_sales_by_product.sort_values(by='predicted_sales_by_product',                                           ascending=False).head(product_count)

                            fig, ax = plt.subplots(figsize=(10,6))
                            sns.kdeplot(data=chart_data, x='total_predicted_purchases',                                                                     y='predicted_sales_by_product', fill=True, common_norm=False, palette="crest", alpha=0.5,                                       linewidth=0, ax=ax)
                            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
                            ax.set_title(f'Predicted Sales of Products')
                            ax.set_xlabel('Total Predicted Purchases')
                            ax.set_ylabel('Predicted Sales')
                            st.pyplot(fig)
                            
                            """    
                            # adjust predicted_sales_by_product based on predicted_next_input_period_sales
                            adjustment_factor = total_daily_predicted_sales / total_predicted_sales
                            predicted_sales_by_product['predicted_sales_by_product'] *= adjustment_factor
                            """
                            

#------------------------------------------------------------------------------------------------------------------------   

#st.write(list(df.columns))      st.write(list(rfm.columns))
#st.write(df.describe())
#st.write("This Dashboard is Under Construction")
# Else for if all([os.path.exists(path) for path in paths]):                        
    else:
        st.write("This Dashboard Activates Only Upon Entering Input Data For Analysis in Lifetimes Dashboard")
        st.stop()                        
