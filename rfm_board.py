### Import Libraries
import streamlit as st
import os

import csv
import datetime
#from datetime import datetime
from datetime import timedelta
import pandas as pd

import numpy as np
from functools import reduce
from sklearn import metrics

import lifetimes
from lifetimes import BetaGeoFitter
from lifetimes.datasets import load_transaction_data
from lifetimes.utils import summary_data_from_transaction_data, calibration_and_holdout_data

from lifetimes.plotting import plot_period_transactions, plot_calibration_purchases_vs_holdout_purchases, plot_frequency_recency_matrix, plot_probability_alive_matrix, plot_period_transactions, plot_history_alive

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

#------------------------------------------------------------------------------------------------------------------------

### RFM Preprocessing

def rfm_process(df):
    rfm = lifetimes.utils.summary_data_from_transaction_data(
        df,
        customer_id_col="CustomerID",
        datetime_col="InvoiceDate",
        monetary_value_col="Sales",  # repeated_transactions.groupby(customer_id_col)[monetary_value_col].mean().fillna(0)
        observation_period_end=None,
        freq="D",
        freq_multiplier=1,
    )
    mon_val_of_zero = round(
        rfm[rfm["monetary_value"] == 0]["monetary_value"].count(), 2
    )
    percent_of_zero_rfm = round((mon_val_of_zero / len(rfm)) * 100, 2)

    if percent_of_zero_rfm > 0.05:
        rfm = rfm[rfm["monetary_value"] > 0]
    return rfm

def mean_median_std(rfm):
    retail_stats = pd.DataFrame(
        {
            "mean": round(rfm["monetary_value"].mean(), 2),
            "median": round(rfm["monetary_value"].median(), 2),
            "std": round(rfm["monetary_value"].std(), 2),
            "min": round(rfm["monetary_value"].min(), 2),
            "max": round(rfm["monetary_value"].max(), 2),
        },
        index=["monetary_value"],
    )
    return retail_stats

def split(df):
    from df_period import calibration_period_end, observation_period_end
    # Get calibration and observation periods
    #calibration_period_end, observation_period_end = rfm_period()
    rfm_train_test = lifetimes.utils.calibration_and_holdout_data(
        df,
        "CustomerID",
        "InvoiceDate",
        calibration_period_end=calibration_period_end,
        observation_period_end=observation_period_end,
        monetary_value_col="Sales", freq="D", freq_multiplier=1)
    rfm_train_test = rfm_train_test[rfm_train_test["frequency_cal"] > 0]  # filter out negatives
    return rfm_train_test

#------------------------------------------------------------------------------------------------------------------------

### BGF Model

def baseline_model(rfm_train_test):
    # Fit the Beta-Geo-Frequency model on the training set
    bgf = lifetimes.BetaGeoFitter(penalizer_coef=0.0) # <-regression hyperparameter
    bgf.fit(rfm_train_test['frequency_cal'], rfm_train_test['recency_cal'], rfm_train_test['T_cal'],freq="D",                   freq_multiplier=1)
    return bgf

def penalizer_score(rfm_train_test, bgf):
    from df_prediction_period import prediction_period
    if prediction_period:
        predicted_bgf = round(bgf.conditional_expected_number_of_purchases_up_to_time(prediction_period,                           rfm_train_test["frequency_cal"], rfm_train_test["recency_cal"], rfm_train_test["T_cal"]))
        actual_freq = rfm_train_test["frequency_holdout"]
        predicted_freq = predicted_bgf
        baseline_penalizer_score = metrics.mean_absolute_error(actual_freq, predicted_freq)
        return actual_freq, predicted_freq, baseline_penalizer_score

def hypertuner(rfm_train_test, params, prediction_period):
    list_of_f = []
    for p in params:
        bgf = BetaGeoFitter(penalizer_coef=p)
        bgf.fit(
            rfm_train_test["frequency_cal"],
            rfm_train_test["recency_cal"],
            rfm_train_test["T_cal"], freq="D", freq_multiplier=1
        )
        predicted_bgf = round(
            bgf.conditional_expected_number_of_purchases_up_to_time(
                prediction_period,
                rfm_train_test["frequency_cal"],
                rfm_train_test["recency_cal"],
                rfm_train_test["T_cal"]
            )
        )
        predicted_freq = predicted_bgf
        f_df = pd.DataFrame(
            {
                "p": p,
                "Actual_Avg_Frequency": round(rfm_train_test["frequency_holdout"].mean(), 4),
                "Predicted_Avg_Frequency": round(predicted_freq.mean(), 4),
                "Average_Absolute_Error": metrics.mean_absolute_error(
                    rfm_train_test["frequency_holdout"], predicted_freq
                ),
            },
            index=[p],
        )
        list_of_f.append(f_df)
    f_penalizer_df = pd.concat(list_of_f)
    # Get best penalizer and score
    f_best_penalizer_score = f_penalizer_df["Average_Absolute_Error"].min()
    f_best_penalizer = f_penalizer_df["Average_Absolute_Error"].idxmin()
    rfm_train_test["predicted_purchases_test"] = predicted_bgf
    pp_ts = rfm_train_test["predicted_purchases_test"].sum()
    pp_tm = round(rfm_train_test["predicted_purchases_test"].mean(), 2)

    return bgf, f_penalizer_df.transpose(), f_best_penalizer_score, f_best_penalizer, pp_ts, pp_tm

def hypertuned_bgf_model(rfm_train_test, f_best_penalizer, prediction_period):
    # Fit the Beta-Geo-Frequency model on the training set
    hbgf = lifetimes.BetaGeoFitter(penalizer_coef=f_best_penalizer) # <-regression hyperparameter
    hbgf.fit(rfm_train_test['frequency_cal'], rfm_train_test['recency_cal'], rfm_train_test['T_cal'], freq="D",                 freq_multiplier=1)
    
    rfm_train_test["predicted_purchases_holdout"] = round(
    hbgf.conditional_expected_number_of_purchases_up_to_time(
        prediction_period,
        rfm_train_test["frequency_cal"],
        rfm_train_test["recency_cal"],
        rfm_train_test["T_cal"])
    )
    pp_hs = rfm_train_test["predicted_purchases_holdout"].sum()
    pp_hm = round(rfm_train_test["predicted_purchases_holdout"].mean(), 2)
    
    rfm_train_test["probability_alive"] = hbgf.conditional_probability_alive(
    rfm_train_test["frequency_cal"],
    rfm_train_test["recency_cal"],
    rfm_train_test["T_cal"])
    pph_pa = round(rfm_train_test["probability_alive"].mean(), 2) * 100

    return hbgf, pp_hs, pp_hm, pph_pa

#------------------------------------------------------------------------------------------------------------------------

### GGF Model

def monval_freq_correlation(rfm_train_test):
    # confirm no correlation
    mfc = rfm_train_test[["monetary_value_cal", "frequency_cal"]].corr()
    # insert unit test
    rfm_train_test = rfm_train_test[
        rfm_train_test["monetary_value_cal"] >= 1] #<<<< filter out negatives
    return mfc 

def gg_fitter(rfm_train_test):
    ggf = lifetimes.fitters.gamma_gamma_fitter.GammaGammaFitter(penalizer_coef=0.0001)
    ggf.fit(rfm_train_test["frequency_cal"], rfm_train_test["monetary_value_cal"], freq="D", freq_multiplier=1)
    # p: shape parameter of 1st distro, q: shape parameter of 2nd distro, r: rate parametr of distro
    # Gamma(p, v) defines the distribution of a customerâ€™s observed average transaction value
    return ggf

def ggf_tuner(rfm_train_test, ggf):
    monetary_pred = ggf.conditional_expected_average_profit(
    rfm_train_test["frequency_holdout"], rfm_train_test["monetary_value_holdout"]
    )
    
    params = [0.0, 0.0001, 0.001, 0.005, 0.01, 0.5, 1]
    list_of_m = []
    case = 0
    for p in params:
        ggf = lifetimes.fitters.gamma_gamma_fitter.GammaGammaFitter(penalizer_coef=p)
        ggf.fit(rfm_train_test["frequency_cal"], rfm_train_test["monetary_value_cal"], freq="D", freq_multiplier=1)

        predicted_m = ggf.conditional_expected_average_profit(
            rfm_train_test["frequency_holdout"], rfm_train_test["monetary_value_holdout"]
        )
    
    case +=1
    m_df = pd.DataFrame({'p'                               : p,
                        # 'Actual_Monetary_Value'   : round(rfm_train_test['monetary_value_holdout'].mean(),4),
                        # 'Predicted_Monetary_Value': round(predicted_m,4),
                        'Average_absolute_error'           :                                                                   metrics.mean_absolute_error(rfm_train_test['monetary_value_holdout'], predicted_m)
                      },index=[p])
    list_of_m.append(m_df)

    m_penalizer_df = pd.concat(list_of_m)
    m_best_penalizer_score = m_penalizer_df["Average_absolute_error"].min()
    m_best_penalizer = m_penalizer_df["Average_absolute_error"].idxmin()

    return m_penalizer_df.transpose(), m_best_penalizer_score, m_best_penalizer

def  hypertuned_ggf_model(rfm_train_test, m_best_penalizer):
    hggf = lifetimes.fitters.gamma_gamma_fitter.GammaGammaFitter(penalizer_coef=m_best_penalizer)
    hggf.fit(rfm_train_test["frequency_cal"], rfm_train_test["monetary_value_cal"], freq="D", freq_multiplier=1)

    best_monetary_pred = hggf.conditional_expected_average_profit(
    rfm_train_test["frequency_holdout"], rfm_train_test["monetary_value_holdout"])
    rfm_train_test["predicted_avg_order_value"] = best_monetary_pred
    
    #Evaluate best model against holdout duration
    rfm_train_test["predicted_test_sales_revenue"] = (
    rfm_train_test["predicted_avg_order_value"]
    * rfm_train_test["predicted_purchases_test"]
    ) / rfm_train_test["probability_alive"]
    
    paov_tp = rfm_train_test["predicted_avg_order_value"].mean()
    pp_tp = rfm_train_test["predicted_purchases_test"].sum()
    ptsr = rfm_train_test["predicted_test_sales_revenue"].sum()
    
    return paov_tp, pp_tp, ptsr

def ggfm(rfm, rfm_train_test, m_best_penalizer):
    #Fit monetary_value Best Model
    rfm = rfm.loc[rfm["monetary_value"] > 0]
    mggf = lifetimes.fitters.gamma_gamma_fitter.GammaGammaFitter(penalizer_coef=m_best_penalizer)
    mggf.fit(rfm["frequency"], rfm["monetary_value"], freq="D", freq_multiplier=1)
    
    ggfp = mggf.params_
    
    # Predict M
    returning_customers = rfm[rfm["frequency"] > 0]
    norecu = len(returning_customers)
    
    recuprem = returning_customers["predicted_m"] = mggf.conditional_expected_average_profit(
    returning_customers["frequency"], returning_customers["monetary_value"])
    recuhe = returning_customers.sort_values(by="predicted_m", ascending=False)
    recuprems = returning_customers[returning_customers.predicted_m > 0]
    recuprems = recuprems.predicted_m.sum()
    
    excoavpr = mggf.conditional_expected_average_profit(rfm_train_test["frequency_holdout"],
               rfm_train_test["monetary_value_holdout"]).sum() 
    avepro = rfm_train_test[rfm_train_test["frequency_holdout"] > 0]["monetary_value_holdout"].sum()
    
    return mggf, ggfp, norecu, recuhe, recuprems, excoavpr, avepro

#------------------------------------------------------------------------------------------------------------------------

### CLV Model

def clvm(hbgf, mggf, rfm_train_test):
    from df_prediction_period import prediction_period
    
    from df_discount_rate import annual_discount_rate_percentage
    annual_discount_rate = annual_discount_rate_percentage/100
    daily_discount_rate = np.power(1 + annual_discount_rate, 1/365) - 1
    dadirapr = daily_discount_rate * 100
    weekly_discount_rate = np.power(1 + annual_discount_rate, 1/52) - 1
    wedirapr = weekly_discount_rate * 100
    monthly_discount_rate = np.power(1 + annual_discount_rate, 1/12) - 1
    modirapr = monthly_discount_rate * 100
                       
    clv = mggf.customer_lifetime_value(
    hbgf,
    rfm_train_test["frequency_cal"],
    rfm_train_test["recency_cal"],
    rfm_train_test["T_cal"],
    rfm_train_test["monetary_value_cal"],
    time=prediction_period,  # time in months
    discount_rate = daily_discount_rate)
    
    rfm_train_test["lifetime_sales"] = clv
    clv_rftt = rfm_train_test.sort_values(by="predicted_test_sales_revenue", ascending=False)    
    clv_ptsrs = rfm_train_test[rfm_train_test.predicted_test_sales_revenue > 0]
    clv_ptsrs = clv_ptsrs.predicted_test_sales_revenue.sum()
    clv_lss = rfm_train_test[rfm_train_test.lifetime_sales > 0]
    clv_lss = round(rfm_train_test["lifetime_sales"].sum())
    return annual_discount_rate_percentage, daily_discount_rate, dadirapr, wedirapr, modirapr, clv_rftt, clv_ptsrs, clv_lss

#------------------------------------------------------------------------------------------------------------------------

### Retraining The Models - Iteration 2

def itertwo(rfm, f_best_penalizer, m_best_penalizer, daily_discount_rate):
    from df_prediction_period import prediction_period
    # RFM
    rfm_actuals = rfm
    rfm = rfm.loc[rfm.frequency > 0, :]

    rfm = rfm.loc[
        rfm.monetary_value > 0, :
    ]  # model is sensitive to negative monetary values

    # BG/NBD
    iterbgf = lifetimes.BetaGeoFitter(penalizer_coef=f_best_penalizer)
    iterbgf_fit = iterbgf.fit(rfm["frequency"], rfm["recency"], rfm["T"])
    rfm["probability_alive"] = iterbgf.conditional_probability_alive(
        frequency=rfm["frequency"], recency=rfm["recency"], T=rfm["T"]
    )
    rfm["predicted_purchases"] = iterbgf.conditional_expected_number_of_purchases_up_to_time(
        prediction_period, frequency=rfm["frequency"], recency=rfm["recency"], T=rfm["T"]
    )

    # GG
    iterggf = lifetimes.fitters.gamma_gamma_fitter.GammaGammaFitter(
        penalizer_coef=m_best_penalizer
    )
    iterggf.fit(rfm["frequency"], rfm["monetary_value"], freq="D", freq_multiplier=1)

    # CLV model
    rfm["exp_avg_order_value"] = iterggf.conditional_expected_average_profit(
        rfm["frequency"], rfm["monetary_value"]
    )

    iterclv = iterggf.customer_lifetime_value(
        iterbgf,  # the model to predict the number of future transactions
        rfm["frequency"],
        rfm["recency"],
        rfm["T"],
        rfm["monetary_value"],
        prediction_period,
        discount_rate = daily_discount_rate  # monthly discount rate ~ 12.7% annually
    )

    rfm["predicted_next_input_period_sales"] = (
    rfm["predicted_purchases"] * rfm["exp_avg_order_value"])  # /( 1- rfm['prob_alive'] )
    
    rfm = rfm[rfm["predicted_next_input_period_sales"] > 0]   ### deep analysis: apply to -ve rendering feature 
    
    rfm["CLTSales_clvb"] = iterclv
    
    from df_profit_margin import profit_margin  
    rfm["CLTV_promrgnb"] = iterclv * (profit_margin/100)

    rfmclts = rfm["CLTSales_clvb"].mean()
    
    return rfm, iterbgf, iterbgf_fit, iterggf, iterclv, rfmclts

#-----------------------------------------------------------------------------------------------------------------------

def grab_cohorts(df, rfm):
    gcohort = pd.DataFrame(df).groupby("CustomerID")["InvoiceYear"].min().reset_index()
    gcohort.columns = ["CustomerID", "CohortYear"]
    rfm = pd.merge(
        rfm, gcohort, on="CustomerID"
    )  # toggle between models_on_retail and salescsv
    chrtfm = round(rfm.describe(), 2)
    prenxyr = rfm["predicted_next_input_period_sales"].sum()
    return rfm, chrtfm, prenxyr

def thresh_prediction(rfm, thresh):
    alive_prediction = pd.DataFrame(
        rfm[rfm["probability_alive"] > thresh]
        .groupby("CohortYear")["probability_alive"]
        .sum()
    )
    purchase_prediction = pd.DataFrame(
        rfm[rfm["probability_alive"] > thresh]
        .groupby("CohortYear")["predicted_purchases"]
        .mean()
    )  
    alive_purchasing_prediction = pd.concat(
        [alive_prediction, purchase_prediction], axis=1
    )
    alive_purchasing_prediction.columns = [
        "predicted customer count",
        "predicted avg. number of purchases",
    ]
    return alive_purchasing_prediction

def next_input_period_sales(rfm):
    prepusm = round(rfm["predicted_purchases"].sum(), 2)
    preneyesasm = round(rfm["predicted_next_input_period_sales"].sum(), 2)
    cohort_prediction = (
        rfm.groupby("CohortYear")["predicted_next_input_period_sales"].sum().reset_index()
    )
    cohort_prediction.columns = ["CohortYear", "Next_Input_Period"]
    cohort_prediction["percent_of_next_input_period_sales"] = (
        cohort_prediction["Next_Input_Period"] / rfm["predicted_next_input_period_sales"].sum()
    ) * 100
    
    copre = round(cohort_prediction)
    cltsasm = rfm["CLTSales_clvb"].sum()
    proalv = round((rfm["probability_alive"] > 0.5).sum())
    CLTV_promrgnbsm = round(rfm["CLTV_promrgnb"].sum(), 2)
    
    cust_count_pred = rfm.groupby("CohortYear")["probability_alive"].value_counts(bins=3)
    # cust_count_pred =  cust_count_pred.reset_index()
    cust_count_pred.columns = ["CohortYear", "probability_alive", "CustomerCount"]
    # cust_count_pred
    cucopre = cust_count_pred.unstack(level=1)
    
    cust_prediction_purchases = rfm.groupby("CohortYear")["predicted_purchases"].value_counts(bins=3)
    # cust_prediction_purchases = cust_prediction_purchases.reset_index()
    cust_prediction_purchases.columns = ["CohortYear", "probability_of_purchase", "number_of_purchases"]
    # cust_prediction_purchases
    cuprepur = cust_prediction_purchases.unstack(level=1)
    return prepusm, preneyesasm, copre, cltsasm, proalv, CLTV_promrgnbsm, cucopre, cuprepur

#------------------------------------------------------------------------------------------------------------------------

def frehis_plot(rfm):
    frehis = rfm["frequency"].plot(kind="hist", bins=35, title="Frequency Histogram (Customer(Y) vs Purchases(X))") 
    return frehis

def frequency_recency_matrix(iterbgf):
    plfrecenma = plot_frequency_recency_matrix(iterbgf)
    return plfrecenma

def probability_alive_matrix(iterbgf):
    plproalv = plot_probability_alive_matrix(iterbgf)
    return plproalv

def customer_orders_mtone(df):
    n_orders = df.groupby(["CustomerID"])["InvoiceNo"].nunique()
    mult_orders_perc = np.sum(n_orders > 1) / df["CustomerID"].nunique()
    norcu = n_orders.count()
    muorpe = f"{100 * mult_orders_perc:.2f}"
    return norcu, muorpe

def cuorcohis_plot(rfm):
    cuorcohis_fig = plt.figure(figsize=(12, 8))
    cuorcohis_ax = cuorcohis_fig.add_subplot(111)
    cuorcohis = cuorcohis_ax.hist(x=rfm["frequency"], bins=50)
    return cuorcohis_fig, cuorcohis_ax, cuorcohis

def period_transactions(iterbgf):
    pertr = plot_period_transactions(iterbgf)
    return pertr

def met_freq_eval(actual_freq, predicted_freq, rfm_train_test):
    from df_prediction_period import prediction_period
    evaluation = pd.DataFrame(
    {
        "Prediction_Period": prediction_period,
        "Actual_Avg_Frequency": round(actual_freq.mean(), 4),
        "Predicted_Avg_Frequency": round(predicted_freq.mean(), 4),
        "Average_Absolute_Error": metrics.mean_absolute_error(
        actual_freq, predicted_freq ),},index=["bgf"])
    evatr = evaluation.transpose()
    movaca = rfm_train_test["monetary_value_cal"].describe()
    return evatr, movaca

def cal_vs_hopu(hbgf, rfm_train_test):
    hbgf.fit(rfm_train_test["frequency_cal"], rfm_train_test["recency_cal"], rfm_train_test["T_cal"])
    plcapuhopu = plot_calibration_purchases_vs_holdout_purchases(hbgf, rfm_train_test)
    return plcapuhopu

def rfm_frame(rfm):
    rfm_des = rfm.describe()
    return rfm_des

def check_cids_integrity(df):
    demo_cid = df.CustomerID[0]
    customer_ids_input = st.sidebar.text_input("Process CustomerIDs (comma-separated)", value = demo_cid,                                            key="customer_ids_input")
    try:
        if not customer_ids_input:
            st.sidebar.warning("Provide CustomerID")
            return []

            if customer_ids_input.strip() == "":
                return []
        else:
            customer_ids = list(set(map(int, map(str.strip, customer_ids_input.split(",")))))
            customer_ids = [cid for cid in customer_ids if cid in df["CustomerID"].values]
            if len(customer_ids) > 0:
                st.sidebar.success("Processing Valid CID(s)")
                return customer_ids
            else:
                customer_ids = []
                customer_ids = [cid for cid in df['CustomerID'].values if cid not in customer_ids][:1]
                st.warning("No new customer IDs to add.")
                st.sidebar.warning("Below is First Valid CustomerID For Demonstration")
                return customer_ids
    except ValueError:
        st.sidebar.warning("Invalid input provided. Please enter valid Customer IDs separated by                                   commas.")

def get_valid_cids(customer_ids, df, rfm, f_best_penalizer, df_origins):

    valid_cids = customer_ids
    
    if valid_cids:
        cust_path = "C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/customer_ids.csv"
        cust_path_exists = os.path.exists(cust_path)

        if cust_path_exists:
            cust_df = pd.read_csv(cust_path)
            cust_df_cids = set(cust_df["CustomerID"].values)

            # Remove existing IDs from cust_df_cids
            cust_df_cids = list(set(cust_df_cids) - set(valid_cids))

            # Add remaining IDs from valid_cids to the beginning of cust_df_cids
            cust_df_cids = valid_cids + cust_df_cids

            new_customer_ids_df = pd.DataFrame({"CustomerID": cust_df_cids})
            new_customer_ids_df.to_csv(cust_path, index=False)

        else:
            cust_df_cids = valid_cids
            new_customer_ids_df = pd.DataFrame({"CustomerID": cust_df_cids})
            new_customer_ids_df.to_csv(cust_path, index=False)
        
        selected_cid = st.sidebar.selectbox("Select a Customer ID", cust_df_cids, key = "selected_cid")
        gecustres = rfm[rfm["CustomerID"] == selected_cid].transpose()

        st.subheader(f"Description of RFM Frame  for CustomerID: {selected_cid}")
        gecustres.style.set_properties(**{'text-align': 'center'})
        st.write(gecustres)

        if isinstance(gecustres, pd.DataFrame):
            sp_trans = df.loc[df['CustomerID'] == int(selected_cid)]
            if sp_trans.empty:
                st.write(f"No transactions found for Customer ID {selected_cid}. Please select a different Customer                         ID.")
            else:
                first_invoice_date = min(sp_trans['InvoiceDate'])
                from df_period import observation_period_end
                days_since_birth = (observation_period_end - first_invoice_date).days
                st.write(f"Days since birth: {days_since_birth}")

                # Get best penalizer value
                best_penalizer_value = f_best_penalizer
                st.write("Value of Best Penalizer:", best_penalizer_value)
                
                #Get Last Invoice for selected_cid
                #last_invoice_date = max(sp_trans['InvoiceDate'])
                #last_df_invoicedate = datetime.date.fromisoformat(str(df['InvoiceDate'].iloc[-1]))
                fetch_origins = df_origins.loc[df_origins['CustomerID'] == int(selected_cid)]
                last_invoice = df_origins.loc[df_origins['InvoiceDate'] == max(fetch_origins['InvoiceDate'])]
                st.subheader(f"Last Invoice for CustomerID: {selected_cid} ")
                st.write(last_invoice)

                # Remove existing selected_cid from cust_df_cids
                selected_cid_set = {int(selected_cid)}
                cust_df_cids = list(set(cust_df_cids) - selected_cid_set)

                # Add selected_cid to the beginning of cust_df_cids
                selected_cid_list = [int(selected_cid)]
                cust_df_cids = selected_cid_list + cust_df_cids

                new_customer_ids_df = pd.DataFrame({"CustomerID": cust_df_cids})
                new_customer_ids_df.to_csv("C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/customer_ids.csv", index=False)            
    
                return selected_cid, sp_trans, days_since_birth, cust_df_cids

def annual_cohorts(df):
    #Cohort
    df_retail = df[['CustomerID', 'InvoiceNo', 'InvoiceYear']].drop_duplicates()
    # df['order_month'] = df['Date'].to_period('')
    df_retail['cohort'] = df_retail.groupby('CustomerID')['InvoiceYear'].transform('min')

    df_cohort = df_retail.groupby(['cohort', 'InvoiceYear']).agg(n_customers=('CustomerID',                                                 'nunique')).reset_index(drop=False)
    df_cohort['period_number'] = (df_cohort.InvoiceYear - df_cohort.cohort)

    cohort_pivot = df_cohort.pivot_table(index = 'cohort', columns = 'period_number', values = 'n_customers')
    cohort_size = cohort_pivot.iloc[:,0]
    cohort_size.columns = ['CohortYear','unique customers per cohort']

    retention_matrix = cohort_pivot.divide(cohort_size, axis=0) 
    

    return cohort_size, retention_matrix

def plot_retention_matrix(cohort_size, retention_matrix):
    with sns.axes_style("white"):
        plrema_fig, plrema_ax = plt.subplots(1, 2, figsize=(8, 6), sharey=True, gridspec_kw={"width_ratios": [1, 11]})
        # retention matrix
        sns.heatmap(retention_matrix, mask=retention_matrix.isnull(), annot=True, fmt=".0%", cmap="icefire",                       ax=plrema_ax[1])
        plrema_ax[1].set_title("Yearly Cohorts: User Retention", fontsize=16)
        plrema_ax[1].set(xlabel="# of periods", ylabel="")
        # cohort size
        cohort_size_df = pd.DataFrame(cohort_size).rename(columns={0: "cohort_size"})
        white_cmap = mcolors.ListedColormap(["White"])
        sns.heatmap(cohort_size_df, annot=True, cbar=False, fmt="g", cmap=white_cmap, ax=plrema_ax[0])
        st.subheader("User Retention Matrix By Yearly Cohorts") 
    return plrema_fig

def custats(rfm): 
    # Define the range of values for the slider
    min_value = 0
    max_value = 100
    default_value = 10
    step_size = 5

    # Display the slider widget
    cust_count = st.sidebar.slider("Select Customers Count", min_value, max_value, default_value, step_size)

    # Use the selected value in your code
    st.write(f"You selected {cust_count} Customers")
    becust = rfm.sort_values(by="predicted_next_input_period_sales", ascending=False).head(cust_count).reset_index()
    return becust

def pqr(current_input):
    # Increment the current_input by 1 and return it
    return current_input + 1

def rbg(rfm, f_best_penalizer, m_best_penalizer, daily_discount_rate, prediction_period, iterbgf, iterbgf_fit, iterggf):

    with open('C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/predicted_sales.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        #writer.writerow(['input', 'output'])
        writer.writerow(['date', 'output'])
        
        parameter_def = prediction_period
        pps_sum_values = []
        date_i_values = []
        j = int(parameter_def)
        from df_period import observation_period_end
        observation_period_end_str = observation_period_end.strftime('%Y-%m-%d')
        end_date = datetime.datetime.strptime(observation_period_end_str, '%Y-%m-%d')
        
        for i in range(0, j+1):
            parameter_pqr = pqr(i) 
            if parameter_pqr <= j:
                date_i = end_date + timedelta(days=parameter_pqr)
                date_i_values.append(date_i)
                parameter_def = parameter_pqr
                rfm["pre_pur_on_selected_date"] = iterbgf.conditional_expected_number_of_purchases_up_to_time(parameter_def,                                                  frequency=rfm["frequency"], recency=rfm["recency"], T=rfm["T"])
            
                rfm["preneinpesa_on_selected_date"] = (rfm["pre_pur_on_selected_date"] * rfm["exp_avg_order_value"])
                rfm = rfm[rfm["preneinpesa_on_selected_date"] > 0]
                pps_sum = rfm["preneinpesa_on_selected_date"].sum()
                pps_sum_values.append(pps_sum)
                
                # Calculate the differences between adjacent elements in the list n_values
                pps_sum_diff_list = [pps_sum_values[0]] + [pps_sum_values[j+1] - pps_sum_values[j] for j in                                 range(len(pps_sum_values) - 1)]
 
        for i, pps_sum in enumerate(pps_sum_diff_list):
            date_i = date_i_values[i]
            writer.writerow([date_i.strftime('%Y-%m-%d'), pps_sum])
    
def get_pps_sum_for_date(df):
    st.subheader("Generate Prediction For Selected Date")
    from df_period import observation_period_end
    observation_period_end_str = observation_period_end.strftime('%Y-%m-%d')
    observation_period_end = datetime.datetime.strptime(observation_period_end_str, '%Y-%m-%d')
    
    from df_prediction_period import prediction_period
    start_date = observation_period_end+timedelta(days=1)
    max_date = observation_period_end+timedelta(days=prediction_period)
    date_input = st.sidebar.date_input("Select a date", value=start_date,min_value=start_date,                                 max_value=max_date)
    
    with open('C:/Users/Desk/Documents/Working-Projects/0-Retail-Prediction/predicted_sales.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader) # skip header row
        for row in reader:
            row_date = datetime.datetime.strptime(row[0], '%Y-%m-%d').date()
            if row_date == date_input:
                st.write(f"Predicted Sales for {date_input} Date:", row[1])
                break
        else:
            st.write("No data found for selected date.")
    
    # Get month and day of selected date
    selected_month = date_input.month
    selected_day = date_input.day

    # Filter dataframe for the same month and day in earlier years
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['InvoiceDay'] = df['InvoiceDate'].dt.day
    same_date_df = df[(df['InvoiceMonth'] == selected_month) & (df['InvoiceDay'] == selected_day) & 
                      (df['InvoiceYear'] < date_input.year)]

    # Group the same_date_df by year and calculate sales for each year
    sales = same_date_df.groupby('InvoiceYear')['Sales'].sum()
    
    return selected_month, selected_day, sales, date_input
            
def listed_products(df):
    # Group the data by product description and calculate sales for each description
    sales_by_product = df.groupby('Description')['Sales'].sum().reset_index()

    # Merge in the active since date for each product
    products_list = pd.DataFrame(df.groupby("Description")["InvoiceDate"].min())
    products_list = products_list.reset_index()
    products_list.columns = ["Description", "ActiveSince"]
    sales_by_product = sales_by_product.merge(products_list, on="Description")
    
    st.write(products_list)
    
    # Define the range of values for the slider
    min_value = 0
    max_value = 100
    default_value = 10
    step_size = 5
    
    # Display the slider widget
    product_count = st.sidebar.slider("Select Product Count", min_value, max_value, default_value, step_size)

    # Use the selected value in your code
    st.write(f"You selected {product_count} Products")
    top_products = sales_by_product.sort_values(by='Sales', ascending=False).head(product_count).reset_index()
    
    # Print the top 10 products by sales
    st.subheader(f'Top {product_count} Products by Sales')
    st.write(top_products)
    return product_count, sales_by_product

def max_product_purchase(df, rfm, product_count):
    # Sort the rfm DataFrame by predicted next input period sales in descending order and get the top 10 customers
    top_10_customers = rfm.sort_values(by="predicted_next_input_period_sales", ascending=False).head(10)                           ["CustomerID"].tolist()
    
    default_customer = top_10_customers[0]
    
    # Create a sidebar selectbox to choose from the top 10 customers
    selected_customer = st.sidebar.selectbox("Select CustomerID:", top_10_customers,                                               index=top_10_customers.index(default_customer))
    
    # Display the selected customer's information if a customer is selected
    if selected_customer:
        #stream1
        # Merge df on rfm dataframes on CustomerID column
        dfmerfm_df = pd.merge(df, rfm[['CustomerID', 'pre_pur_on_selected_date']], on='CustomerID', how='inner')
        st.subheader(f"Retail Merged on RFM (Predicted Purchases)")
        st.write(dfmerfm_df)
        
        # Group merged dataframe by CustomerID and Description to calculate total amount spent by each customer on each                 product
        desgrosa_df = dfmerfm_df.groupby(['CustomerID', 'Description']).agg({'Sales': 'sum'})
        st.subheader(f"Aggregated on Past Sales")
        st.write(desgrosa_df)
        
        # Identify yielding customers who have made purchases before
        yielding_customers = rfm[rfm['frequency'] > 0]['CustomerID'].unique()

        # Print the products that the yielding customers are expected to purchase based on their previous purchase pattern
        for customer in yielding_customers:
            customer_df = desgrosa_df.loc[selected_customer]
            max_sales = customer_df['Sales'].max()
            expected_products = customer_df[customer_df['Sales'] == customer_df['Sales'].max()].index.values
             
        st.write(f"Customer {selected_customer} is expected to purchase {', '.join(expected_products)} based on their                   previous purchase pattern for {max_sales} Monetary Units.")
        #return merged_df, grouped_df, merged_rfm

        #Stream 2

        # calculate total predicted sales based on predicted_next_input_period_sales
        invoice_products = df.groupby(['CustomerID', 'InvoiceDate', 'InvoiceNo', 'Description', 'UnitPrice'])                           ['Quantity'].sum().reset_index()
        st.subheader("Invoice Retail Products (Grouped By Quantity)")
        st.write(invoice_products)
        
        #Merge Invoice Retail Products on RFM (for Predicted Purchases)
        daily_predicted_sales = invoice_products.merge(rfm[['CustomerID',                                                               'pre_pur_on_selected_date']], on='CustomerID')
        st.write(f"Predicted Purchases")
        st.write(rfm["pre_pur_on_selected_date"].sum())
        
        # Merge Daily Predicted Sales on RFM for Predicted Next Input Period Sales On Selected Date
        daily_predicted_sales = pd.merge(daily_predicted_sales, rfm[['CustomerID', 'preneinpesa_on_selected_date']],                   on='CustomerID', how='inner')

        # Count number of Descriptions for a given CustomerID for a given InvoiceDate as new feature                                   date_total_description_count
        daily_predicted_sales['date_total_description_count'] = daily_predicted_sales.groupby(['CustomerID',                             'InvoiceDate'])['Description'].transform('count')
        
        #st.write(daily_predicted_sales.groupby(['InvoiceDate'])['Description'].transform('sum'))        

        # Sum up the Quantity based on InvoiceDate for a given CustomerID as new feature                                               date_total_quant
        daily_predicted_sales['date_total_quant'] = daily_predicted_sales.groupby(['CustomerID',                                      'InvoiceDate'])['Quantity'].transform('sum')

        # Divide Quantity by date_total_quant and name new feature as date_fractional_quant
        daily_predicted_sales['date_fractional_quant'] = (daily_predicted_sales['Quantity'] * 100)/                                     daily_predicted_sales['date_total_quant'] 

        # Multiply date_fractional_quant by predicted_next_input_period_sales and name the new feature                                 as daily_predicted_sales
        
        daily_predicted_sales['daily_predicted_purchases_of_product'] =                                                                           daily_predicted_sales['pre_pur_on_selected_date'] *                                                                             (daily_predicted_sales['date_fractional_quant'] / 100)
        
        # Merge df on rfm dataframes on CustomerID column
        daily_predicted_sales = pd.merge(daily_predicted_sales, rfm[['CustomerID', 'exp_avg_order_value']], on='CustomerID',           how='inner')
        
        daily_predicted_sales["dapresaprod_ggf"] =                                                                                     (daily_predicted_sales['daily_predicted_purchases_of_product'] * daily_predicted_sales["exp_avg_order_value"])         
        
        dapresaprosm_ggf = daily_predicted_sales.groupby('InvoiceDate')['dapresaprod_ggf'].sum().reset_index()
        dapresaprosm_ggf = dapresaprosm_ggf.rename(columns={'dapresaprod_ggf': 'daily_predicted_sales_product_based'})
        
        daily_predicted_sales = pd.merge(daily_predicted_sales, dapresaprosm_ggf, on='InvoiceDate', how='inner')
        
        st.subheader(f"Daily Predicted Sales Merged on RFM (Predicted Next Input Period Sales On Selected Date)")
        st.write(daily_predicted_sales) 
        
        # 
        daily_predicted_sales['dapresaprod_quant'] = daily_predicted_sales['dapresaprod_ggf'] /                                         (daily_predicted_sales['daily_predicted_sales_product_based'] / 100)
        
        daily_predicted_sales['daily_predicted_sales_of_product'] = daily_predicted_sales['preneinpesa_on_selected_date'] *             (daily_predicted_sales['dapresaprod_quant'] / 100)
        
        #
        dapresaprosm = daily_predicted_sales.groupby('InvoiceDate')['daily_predicted_sales_of_product'].sum().reset_index()
        dapresaprosm = dapresaprosm.rename(columns={'daily_predicted_sales_of_product': 'preneinpesa_rectify'})
        
        daily_predicted_sales = pd.merge(daily_predicted_sales, dapresaprosm, on='InvoiceDate', how='inner')
        
        st.subheader("Daily Predicted Sales")
        st.write(daily_predicted_sales)
        
        dapresasm = daily_predicted_sales["daily_predicted_sales"].sum()
        st.write(f"Daily Predicted Sales Sum: {dapresasm}")
        
        product_count = daily_predicted_sales.groupby('Description')['CustomerID'].count().reset_index()
        product_count = product_count.rename(columns={'CustomerID': 'customer_count_of_past_purchase'})
        
        product_purchases = daily_predicted_sales.groupby('Description')['daily_predicted_sales'].sum().reset_index()
        product_purchases = product_purchases.rename(columns={'daily_predicted_sales': 'predicted_sales_by_product'})

        predicted_counts = pd.merge(product_count, product_purchases, on='Description', how='inner')
        
        product_prices = daily_predicted_sales.groupby('Description')['UnitPrice'].mean().reset_index()
        product_prices = product_prices.rename(columns={'UnitPrice': 'mean_unit_price'})

        predicted_sales_by_product = pd.merge(predicted_counts, product_prices, on='Description', how='inner')
        
        # toggle mean_unit_price with observation_end_date_unit_price for shorter prediction_period intervals
        #predicted_sales_by_product['predicted_sales_by_product'] = predicted_sales_by_product['mean_unit_price'] *                     predicted_sales_by_product['total_predicted_purchases']
        
        predicted_sales_by_product = predicted_sales_by_product.sort_values('predicted_sales_by_product',                               ascending=False)
        st.subheader("Predicted Sales By Product")
        st.write(predicted_sales_by_product)
        
        total_predicted_sales = predicted_sales_by_product["predicted_sales_by_product"].sum()
        st.subheader(f"Total Sales By Predicted Products: {total_predicted_sales:.2f}, Ideal Max Sales Day in Prediction               Window")                
        
        return predicted_sales_by_product
    
def xyz():
    
    return 