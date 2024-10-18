import streamlit as st
from multidash import MultiDash
from dashboards import lifetimes_dashboard, rfm_cohort_dashboard, retail_products_dashboard

dashboard = MultiDash()

# Add all your application here
dashboard.add_dash("Lifetimes Dashboard", lifetimes_dashboard.dashboard)
dashboard.add_dash("RFM Based Cohorts Dashboard", rfm_cohort_dashboard.dashboard)
dashboard.add_dash("Retail Products Dashboard", retail_products_dashboard.dashboard)

# The main dashboard
dashboard.run()
