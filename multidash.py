"""Frameworks for running multiple Streamlit Dashboards from a single dashboard (retail_dashboards.py).
"""
import streamlit as st

class MultiDash:
    """Framework for combining multiple streamlit dashboards.
    Usage:
        def foo():
            st.title("Hello Foo")
        def bar():
            st.title("Hello Bar")
        dashboard = Multidashboard()
        dashboard.add_dash("Foo", foo)
        dashboard.add_dash("Bar", bar)
        dashboard.run()
    It is also possible keep each dashboardlication in a separate file.
        import foo
        import bar
        dashboard = Multidashboard()
        dashboard.add_dash("Foo", foo.dashboard)
        dashboard.add_dash("Bar", bar.dashboard)
        dashboard.run()
    """
    def __init__(self):
        self.dashboards = []

    def add_dash(self, title, func):
        """Adds a new dashboard.
        Parameters
        ----------
        func:
            the python function to render this dashboard.
        title:
            title of the dashboard. dashboardears in the dropdown in the sidebar.
        """
        self.dashboards.append({
            "title": title,
            "function": func
        })

    def run(self):
        # dashboard = st.sidebar.radio(
        dashboard = st.selectbox(
            'Navigation',
            self.dashboards,
            format_func=lambda dashboard: dashboard['title'])

        dashboard['function']()
