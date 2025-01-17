"""
app.py

Streamlit app showcasing four portfolio strategies:
1. Sector Relative Strength
2. GTAA 13
3. Maximum Sharpe
4. Minimum Volatility

Uses PyPortfolioOpt for realistic portfolio optimization.

Note: This version reads fixed CSV files (dummy_prices.csv & ticker_sectors.csv)
      from the local directory without any file upload options.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# PyPortfolioOpt imports for real optimization
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier

# --------------------------
# 1. UTILITY: LOAD THE DATA
# --------------------------

@st.cache_data
def load_price_data(csv_file="dummy_prices.csv"):
    """
    Load price data CSV into a DataFrame.
    """
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    df = df.dropna(how="all", axis=0)  # Drop rows (dates) with all NaN
    df = df.dropna(how="all", axis=1)  # Drop columns (tickers) with all NaN
    return df

@st.cache_data
def load_sector_data(csv_file="ticker_sectors.csv"):
    """
    Load sector mapping data into a DataFrame.
    Returns a dict: {ticker: sector}
    """
    try:
        df = pd.read_csv(csv_file)
        return dict(zip(df["Ticker"], df["Sector"]))
    except:
        return {}


# -----------------------------
# 2. STRATEGIES IMPLEMENTATION
# -----------------------------

class SectorRelativeStrength:
    """
    For each sector, we compute the average return over a specified lookback window.
    We rank the sectors by return and then allocate capital to the top N sectors.
    Within a sector, we equally weight all tickers for simplicity.
    """

    def __init__(self, lookback_days=90, top_n_sectors=3, sector_map=None):
        self.lookback_days = lookback_days
        self.top_n_sectors = top_n_sectors
        self.sector_map = sector_map if sector_map is not None else {}

    def run(self, price_df):
        if price_df.shape[1] < 1:
            raise ValueError("No tickers in price_df.")
        if self.lookback_days >= len(price_df):
            raise ValueError(f"Lookback window ({self.lookback_days} days) "
                             "is too large for the price history.")
        
        # Step 1: Compute returns over lookback window
        recent_prices = price_df.tail(self.lookback_days)
        ret = recent_prices.iloc[-1] / recent_prices.iloc[0] - 1  # simple total return

        # Step 2: Group tickers by sector
        sector_returns = {}
        for ticker, r in ret.items():
            sector = self.sector_map.get(ticker, "Unknown")
            if sector not in sector_returns:
                sector_returns[sector] = []
            sector_returns[sector].append(r)

        # Step 3: Average return by sector
        avg_sector_ret = {
            s: np.mean(vals) for s, vals in sector_returns.items() if len(vals) > 0
        }

        # Step 4: Rank sectors and pick top N
        sorted_sectors = sorted(avg_sector_ret, key=avg_sector_ret.get, reverse=True)
        top_sectors = sorted_sectors[: self.top_n_sectors]

        # Step 5: Allocate equally among tickers in top sectors
        allocations = {t: 0.0 for t in price_df.columns}
        tickers_in_top_sectors = []
        for t in ret.index:
            sec = self.sector_map.get(t, "Unknown")
            if sec in top_sectors:
                tickers_in_top_sectors.append(t)

        if len(tickers_in_top_sectors) == 0:
            # Edge case: No top sector or no data => all zero
            return pd.DataFrame({"Allocation": allocations, "RecentReturn": ret})
        
        weight = 1.0 / len(tickers_in_top_sectors)
        for t in tickers_in_top_sectors:
            allocations[t] = weight

        # Combine results in a DataFrame
        df_output = pd.DataFrame({"Allocation": allocations, "RecentReturn": ret})
        return df_output


class GTAA13:
    """
    Implementation of a basic GTAA concept: Compare price to a moving average (e.g. 10-month).
    If above, hold; if below, hold 0. Then equally weight all that pass the signal.
    """

    def __init__(self, ma_months=10):
        self.ma_months = ma_months

    def run(self, price_df):
        if price_df.shape[1] < 1:
            raise ValueError("No tickers in price_df.")
        # Convert months to an approximate daily lookback (assume ~21 trading days per month)
        ma_window = self.ma_months * 21
        if ma_window >= len(price_df):
            raise ValueError(f"Moving average window ({ma_window} days) too large.")
        
        # Calculate the moving average
        rolling_ma = price_df.rolling(window=ma_window).mean()
        current_prices = price_df.iloc[-1]
        current_ma = rolling_ma.iloc[-1]

        # Signal: 1 if price > MA, else 0
        signals = (current_prices > current_ma).astype(int)
        n_signals = signals.sum()

        allocations = {}
        if n_signals == 0:
            # Edge case: if everything is below its MA, no allocation
            allocations = {t: 0.0 for t in price_df.columns}
        else:
            weight = 1.0 / n_signals
            for t in price_df.columns:
                allocations[t] = weight if signals[t] == 1 else 0.0

        # Return allocations as a DataFrame
        df_output = pd.DataFrame({"Allocation": allocations})
        df_output["Signal"] = signals
        return df_output


class MaxSharpe:
    """
    Realistic Maximum Sharpe Ratio using PyPortfolioOpt.
    """

    def __init__(self, rf_rate=0.02):
        self.rf_rate = rf_rate

    def run(self, price_df):
        if price_df.shape[1] < 2:
            # Edge case: Need at least two assets to do meaningful optimization
            raise ValueError("Need at least 2 tickers for Max Sharpe optimization.")
        # Calculate returns matrix (daily returns)
        returns = price_df.pct_change().dropna()
        if returns.empty:
            raise ValueError("Not enough data to calculate returns for Max Sharpe.")

        # Estimate expected returns and sample covariance
        mu = mean_historical_return(price_df, frequency=252)  # annualized
        S = CovarianceShrinkage(price_df).ledoit_wolf()

        # Create Efficient Frontier
        ef = EfficientFrontier(mu, S)
        # Maximize the Sharpe ratio
        ef.max_sharpe(risk_free_rate=self.rf_rate)
        weights = ef.clean_weights()

        # Convert to DataFrame
        df_output = pd.DataFrame.from_dict(weights, orient="index", columns=["Allocation"])
        df_output["Allocation"] = df_output["Allocation"].round(4)
        return df_output


class MinVolatility:
    """
    Minimum Volatility strategy using PyPortfolioOpt.
    """

    def run(self, price_df):
        if price_df.shape[1] < 2:
            # Edge case: With 1 ticker, min-vol is trivially 100% that ticker
            # but let's handle it gracefully
            allocations = {price_df.columns[0]: 1.0}
            return pd.DataFrame({"Allocation": allocations})
        
        # Calculate returns matrix
        returns = price_df.pct_change().dropna()
        if returns.empty:
            raise ValueError("Not enough data to calculate returns for Min Volatility.")

        mu = mean_historical_return(price_df, frequency=252)  # annualized
        S = CovarianceShrinkage(price_df).ledoit_wolf()

        ef = EfficientFrontier(mu, S)
        ef.min_volatility()
        weights = ef.clean_weights()

        df_output = pd.DataFrame.from_dict(weights, orient="index", columns=["Allocation"])
        df_output["Allocation"] = df_output["Allocation"].round(4)
        return df_output


# --------------------
# 3. STREAMLIT APP UI
# --------------------
def main():
    st.title("Portfolio Optimization Strategies")

    # 1) Load the data from fixed CSVs (no upload options)
    price_csv_file = "dummy_prices.csv"
    sector_csv_file = "ticker_sectors.csv"

   
      # Strategy Descriptions
    st.markdown("""
    ## Strategy Descriptions
    **1. Sector Relative Strength:**  
    This strategy ranks sectors based on their recent performance over a specified lookback period. 
    It invests equally in the top N performing sectors, with equal allocation among the tickers within those sectors.

    **2. GTAA 13 (Global Tactical Asset Allocation):**  
    This strategy uses a moving average (e.g., 10-month) to decide whether to hold an asset. 
    If the price is above its moving average, the asset is included in the portfolio with equal weight among all qualified assets.

    **3. Maximum Sharpe Ratio:**  
    This strategy finds the optimal portfolio allocation that maximizes the Sharpe ratio. 
    It uses historical returns and covariance of the assets to construct a portfolio that provides the best risk-adjusted return.

    **4. Minimum Volatility:**  
    This strategy constructs a portfolio that minimizes the overall volatility. 
    It is suitable for risk-averse investors seeking stable returns with minimal risk.
    """)

   
    # Load Price Data
    try:
        price_df = load_price_data(price_csv_file)
    except Exception as e:
        st.error(f"Error loading price data from {price_csv_file}: {e}")
        return

    # Load Sector Data
    sector_map = load_sector_data(sector_csv_file)

    # Show data preview
    st.subheader("Preview of Price Data")
    st.write(price_df.tail())

    # Show a chart of the price history
    fig_price = px.line(price_df, x=price_df.index, y=price_df.columns, title="Historical Prices")
    st.plotly_chart(fig_price, use_container_width=True)

    # 2) Strategy Selection & Parameters
    st.sidebar.header("Strategy Selection")
    strategies = ["Sector Relative Strength", "GTAA 13", "Maximum Sharpe", "Minimum Volatility"]
    selected_strategy = st.sidebar.selectbox("Choose a Strategy", strategies)

    # 3) Show relevant input fields for each strategy
    if selected_strategy == "Sector Relative Strength":
        lookback_days = st.sidebar.number_input("Lookback Window (days)", min_value=30, max_value=365, value=90)
        top_n = st.sidebar.number_input("Top N Sectors", min_value=1, max_value=10, value=3)

        if st.sidebar.button("Run Strategy"):
            srs = SectorRelativeStrength(
                lookback_days=lookback_days,
                top_n_sectors=top_n,
                sector_map=sector_map
            )
            try:
                result_df = srs.run(price_df)
                st.subheader("Results - Sector Relative Strength")
                st.dataframe(result_df)

                allocs_srs = result_df["Allocation"]
                fig_alloc = px.bar(
                    allocs_srs, 
                    x=allocs_srs.index, 
                    y=allocs_srs.values,
                    title="Proposed Portfolio Allocation"
                )
                st.plotly_chart(fig_alloc, use_container_width=True)

            except Exception as e:
                st.error(f"Strategy Error: {e}")

    elif selected_strategy == "GTAA 13":
        ma_months = st.sidebar.number_input("Moving Average (months)", min_value=1, max_value=24, value=10)

        if st.sidebar.button("Run Strategy"):
            gtaa = GTAA13(ma_months=ma_months)
            try:
                result_df = gtaa.run(price_df)
                st.subheader("Results - GTAA 13")
                st.dataframe(result_df)

                fig_alloc = px.bar(
                    result_df["Allocation"], 
                    x=result_df.index, 
                    y=result_df["Allocation"],
                    title="Proposed Portfolio Allocation (GTAA 13)"
                )
                st.plotly_chart(fig_alloc, use_container_width=True)

            except Exception as e:
                st.error(f"Strategy Error: {e}")

    elif selected_strategy == "Maximum Sharpe":
        rf_rate = st.sidebar.number_input("Risk-Free Rate (decimal)", min_value=0.0, max_value=0.10, value=0.02, step=0.01)

        if st.sidebar.button("Run Strategy"):
            ms = MaxSharpe(rf_rate=rf_rate)
            try:
                result_df = ms.run(price_df)
                st.subheader("Results - Maximum Sharpe")
                st.dataframe(result_df)

                fig_alloc = px.bar(
                    result_df["Allocation"], 
                    x=result_df.index, 
                    y=result_df["Allocation"],
                    title="Proposed Portfolio Allocation (Max Sharpe)"
                )
                st.plotly_chart(fig_alloc, use_container_width=True)

            except Exception as e:
                st.error(f"Strategy Error: {e}")

    else:  # "Minimum Volatility"
        if st.sidebar.button("Run Strategy"):
            mv = MinVolatility()
            try:
                result_df = mv.run(price_df)
                st.subheader("Results - Minimum Volatility")
                st.dataframe(result_df)

                fig_alloc = px.bar(
                    result_df["Allocation"], 
                    x=result_df.index, 
                    y=result_df["Allocation"],
                    title="Proposed Portfolio Allocation (Min Vol)"
                )
                st.plotly_chart(fig_alloc, use_container_width=True)

            except Exception as e:
                st.error(f"Strategy Error: {e}")


if __name__ == "__main__":
    main()
