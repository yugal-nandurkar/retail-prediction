Move dummy_dashboard.py, lifetimes_dashboard.py, retail_products_dashboard.py, rfm_cohort_dashboard.py to 'dashboards' folders while running the project in streamlit.

Download df and df origins database from following link and place them in root directory of retail prediction project:
https://drive.google.com/file/d/1711gkV90d9tXNolIVoFiDG7ukSpyzYWT/view?usp=sharing

conda create -n streamlit_env python=3.8

conda activate streamlit_env

streamlit run retail_dashboards.py

pip install : lifetimes, matplotlib, seaborn, scikit-learn, PyQt6, PyQt5
