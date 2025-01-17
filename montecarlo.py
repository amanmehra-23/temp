import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Streamlit app setup
st.title('Monte Carlo Simulation for Stock Prices')

# Layout for user inputs
col1, col2, col3 = st.columns(3)
default_days = 250
default_simulations = 1000

# Inputs
user_ticker = col1.text_input('Enter Ticker Symbol', value='AAPL').upper()
start_date = col2.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = col3.date_input("End Date", value=pd.to_datetime("today"))
days = int(st.text_input("Days to Simulate", str(default_days)))
simulations = int(st.text_input("Number of Simulations", str(default_simulations)))

# Initialize session state for simulation results
if "results" not in st.session_state:
    st.session_state["results"] = None
    st.session_state["average_predicted_close"] = None

# Fetch historical data using yfinance
if st.button("Run Simulation"):
    try:
        stock_data = yf.download(user_ticker, start=start_date, end=end_date)
        if not stock_data.empty:
            # Prepare the DataFrame
            stock_data['Returns'] = stock_data['Close'].pct_change()
            mu = stock_data['Returns'].mean()
            sigma = stock_data['Returns'].std()
            initial_price = stock_data['Close'].iloc[-1]

            # Monte Carlo simulation function
            def monte_carlo_simulation(start_price, mu, sigma, days, simulations):
                dt = 1
                results = np.zeros((days + 1, simulations))
                results[0] = start_price
                for t in range(1, days + 1):
                    shock = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt), size=simulations)
                    results[t] = results[t - 1] * np.exp(shock)
                return results

            # Running simulation
            results = monte_carlo_simulation(initial_price, mu, sigma, days, simulations)
            st.session_state["results"] = results
            st.session_state["average_predicted_close"] = np.sum(results[-1]) / simulations

            # Visualization
            fig = go.Figure()
            for i in range(simulations):
                fig.add_trace(go.Scatter(x=np.arange(days + 1), y=results[:, i], mode='lines', opacity=0.5, showlegend=False))

            # Define regions
            fig.add_vrect(x0=0, x1=30, fillcolor="red", opacity=0.3, layer="below", line_width=0, annotation_text="Unreliable", annotation_position="top left")
            fig.add_vrect(x0=30, x1=210, fillcolor="yellow", opacity=0.3, layer="below", line_width=0, annotation_text="Transition Phase", annotation_position="top left")
            fig.add_vrect(x0=210, x1=days, fillcolor="green", opacity=0.3, layer="below", line_width=0, annotation_text="Reliable", annotation_position="top left")

            # Update layout
            fig.update_layout(title=f'Monte Carlo Simulation Results for {user_ticker}', xaxis_title='Days', yaxis_title='Simulated Price')
            st.plotly_chart(fig)

            st.write(f"Average predicted close price after {days} days: {st.session_state['average_predicted_close']:.2f}")
        else:
            st.write("No data found for the given ticker and date range.")
    except Exception as e:
        st.write(f"An error occurred: {e}")

# Selling price input and advice
user_selling_price_str = st.text_input('Enter Selling Price')
if user_selling_price_str:
    try:
        user_selling_price = float(user_selling_price_str)
        if st.session_state["average_predicted_close"] is not None:
            if st.session_state["average_predicted_close"] - user_selling_price < 0:
                st.write("Selling immediately would be advisable.")
            else:
                st.write("Holding now would be advisable.")
        else:
            st.write("Run the simulation first to get the average predicted price.")
    except ValueError:
        st.write("Please enter a valid selling price.")
