import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
import statsmodels.api as sm

#############################
# 1) MUTUAL FUND NAME -> TICKER MAPPING
#############################
MUTUAL_FUND_MAP = {
    "Vanguard 500 Index Admiral Shares": "VFIAX",
    "Fidelity 500 Index Fund": "FXAIX",
    "Schwab S&P 500 Index Fund": "SWPPX",
    "T. Rowe Price Blue Chip Growth": "TRBCX",
    "Vanguard Total Stock Market Index": "VTSAX",
}
AVAILABLE_FUNDS = list(MUTUAL_FUND_MAP.keys())

# Use SPY as a proxy for the S&P 500
SP500_TICKER = "SPY"

#############################
# 2) HELPER FUNCTIONS
#############################
def fetch_data(ticker, start, end):
    """Fetch daily Close price data from yfinance for the specified ticker."""
    return yf.download(ticker, start=start, end=end, progress=False)

def compute_performance_stats(returns, freq=252):
    """
    Compute basic performance stats for a daily returns series:
    - Annualized Return
    - Annualized Volatility
    - Sharpe Ratio (assuming a 0% risk-free rate for simplicity)
    """
    if returns.empty:
        return {
            "Annualized Return": np.nan,
            "Annualized Volatility": np.nan,
            "Sharpe Ratio": np.nan,
        }
    ann_return = (1 + returns.mean()) ** freq - 1
    ann_vol = returns.std() * np.sqrt(freq)
    if ann_vol == 0:
        sharpe = np.nan
    else:
        sharpe = ann_return / ann_vol
    
    return {
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
    }

def run_regression(dependent_returns, independent_returns):
    """
    Runs a linear regression using Statsmodels:
       dependent_returns = alpha + beta * independent_returns + error
    Returns the fitted model results, plus alpha, beta.
    """
    df_reg = pd.DataFrame({
        "dep": dependent_returns,
        "indep": independent_returns
    }).dropna()
    if df_reg.empty:
        return None, np.nan, np.nan
    
    X = sm.add_constant(df_reg["indep"])  # add intercept
    y = df_reg["dep"]
    model = sm.OLS(y, X).fit()
    
    alpha = model.params["const"]
    beta = model.params["indep"]
    
    return model, alpha, beta

def compute_drawdown(returns):
    """
    Computes drawdown series from a daily returns series.
    Drawdown is (current_cum_value / running_max) - 1
    """
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    return drawdown

def compute_monthly_returns(returns):
    """
    Resample daily returns to monthly returns:
    monthly_return = product of (1 + daily_returns) - 1 for each month
    """
    # Convert daily returns to monthly returns
    monthly = (1 + returns).resample("M").prod() - 1
    return monthly

def create_monthly_heatmap(returns, label="Fund"):
    """
    Create a heatmap of monthly returns, with years as rows and months as columns.
    Returns a Plotly Figure.
    """
    # Convert the daily returns to monthly
    monthly_returns = compute_monthly_returns(returns)
    if monthly_returns.empty:
        # Return an empty figure if no data
        fig = go.Figure()
        fig.update_layout(title=f"Monthly Returns Heatmap - {label} (No Data)")
        return fig
    
    # Create a DataFrame with columns for Year and Month
    monthly_df = monthly_returns.to_frame(name="monthly_ret")
    monthly_df["Year"] = monthly_df.index.year
    monthly_df["Month"] = monthly_df.index.month
    
    # Pivot table => rows=Year, cols=Month, values=monthly_ret
    pivot_data = monthly_df.pivot(index="Year", columns="Month", values="monthly_ret")
    
    # Sort the rows so the most recent year is on top (optional)
    pivot_data = pivot_data.sort_index(ascending=False)
    
    # Create the heatmap
    fig = px.imshow(
        pivot_data * 100,  # convert to percentages
        color_continuous_scale="RdBu",
        labels=dict(color="Return (%)"),
        title=f"Monthly Returns Heatmap - {label}",
        aspect="auto"
    )
    
    # Fix the x-axis labels to show month names
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(1, 13)),
        ticktext=month_names
    )
    
    return fig

#############################
# 3) STREAMLIT APP
#############################
def main():
    st.title("Mutual Fund vs. S&P 500 Tearsheet")

    st.sidebar.header("Input Parameters")
    
    # 1) Mutual fund selection
    fund_name = st.sidebar.selectbox("Select Mutual Fund", AVAILABLE_FUNDS)
    # 2) Map the selected name to a ticker
    fund_ticker = MUTUAL_FUND_MAP[fund_name]
    
    # 3) Date range
    default_start = date(2018, 1, 1)
    default_end = date.today()
    
    start_date = st.sidebar.date_input("Start Date", default_start)
    end_date = st.sidebar.date_input("End Date", default_end)
    
    # 4) Button to run the analysis
    if st.sidebar.button("Generate Tearsheet"):
        #############################
        # A) Data Fetch & Processing
        #############################
        fund_df = fetch_data(fund_ticker, start_date, end_date)
        spy_df = fetch_data(SP500_TICKER, start_date, end_date)
        
        if fund_df.empty or spy_df.empty:
            st.error("No data found for the given date range. Check the inputs.")
            return
        
        fund_df = fund_df[['Close']].rename(columns={"Close": "FundPrice"})
        spy_df = spy_df[['Close']].rename(columns={"Close": "SP500Price"})
        
        combined_df = fund_df.join(spy_df, how='inner')  # join on date index
        if combined_df.empty:
            st.error("No overlapping data between fund and S&P 500. Adjust date range.")
            return
        
        combined_df["Fund"] = combined_df["FundPrice"].pct_change()
        combined_df["SP500"] = combined_df["SP500Price"].pct_change()
        combined_df.dropna(inplace=True)
        
        if combined_df.empty:
            st.error("No valid daily returns after merging. Adjust date range.")
            return
        
        #############################
        # B) Basic Tearsheet Charts
        #############################
        st.subheader(f"Tearsheet for {fund_name} ({fund_ticker}) vs. S&P 500 (SPY)")
        
        # 1) Performance Stats
        fund_stats = compute_performance_stats(combined_df["Fund"])
        sp500_stats = compute_performance_stats(combined_df["SP500"])
        
        # 2) Regression (alpha, beta)
        model, alpha, beta = run_regression(combined_df["Fund"], combined_df["SP500"])
        
        # 3) Cumulative Returns Chart (Benchmark Comparison)
        cum_data = (1 + combined_df[["Fund", "SP500"]]).cumprod()
        fig_cum = px.line(
            cum_data,
            x=cum_data.index,
            y=["Fund", "SP500"],
            title="Cumulative Returns"
        )
        st.plotly_chart(fig_cum, use_container_width=True)
        
        # Show Stats side by side
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Fund Performance Stats**")
            for k, v in fund_stats.items():
                st.write(f"{k}: {v:.2%}")
        with col2:
            st.markdown("**S&P 500 (SPY) Performance Stats**")
            for k, v in sp500_stats.items():
                st.write(f"{k}: {v:.2%}")
        
        # Regression results
        st.write("---")
        st.markdown("**Regression Analysis**")
        if model is None:
            st.warning("Not enough data for regression.")
        else:
            st.write(f"Alpha: {alpha:.4%}")
            st.write(f"Beta : {beta:.2f}")
            with st.expander("See detailed regression summary"):
                st.text(model.summary())
        
        #########################################
        # C) NEW VISUALIZATIONS
        #########################################
        
        # 1.2 Drawdown Chart
        with st.expander("Drawdown Chart"):
            # Compute drawdowns for Fund
            drawdown_fund = compute_drawdown(combined_df["Fund"])
            drawdown_sp500 = compute_drawdown(combined_df["SP500"])
            
            df_drawdown = pd.DataFrame({
                "Fund Drawdown": drawdown_fund,
                "SP500 Drawdown": drawdown_sp500
            })
            
            fig_dd = px.line(
                df_drawdown,
                x=df_drawdown.index,
                y=["Fund Drawdown", "SP500 Drawdown"],
                title="Drawdown Over Time"
            )
            # Format y-axis as percentages
            fig_dd.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_dd, use_container_width=True)
            st.markdown("Drawdown measures the percentage drop from the most recent peak.")
        
        # 1.3 Monthly Returns Heatmap
        with st.expander("Monthly & Annual Returns Heatmap"):
            # Fund Heatmap
            fig_fund_heatmap = create_monthly_heatmap(combined_df["Fund"], label="Fund")
            st.plotly_chart(fig_fund_heatmap, use_container_width=True)
            
            # SP500 Heatmap
            fig_sp500_heatmap = create_monthly_heatmap(combined_df["SP500"], label="S&P 500")
            st.plotly_chart(fig_sp500_heatmap, use_container_width=True)
        
        # 1.5 Return Distribution Histogram
        with st.expander("Return Distribution Histogram"):
            # Plot daily returns distribution for Fund and SP500
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=combined_df["Fund"] * 100,
                opacity=0.5,
                name="Fund",
                nbinsx=50
            ))
            fig_hist.add_trace(go.Histogram(
                x=combined_df["SP500"] * 100,
                opacity=0.5,
                name="S&P 500",
                nbinsx=50
            ))
            fig_hist.update_layout(
                barmode='overlay',
                title="Distribution of Daily Returns (%)",
                xaxis_title="Daily Return (%)",
                yaxis_title="Frequency"
            )
            fig_hist.update_traces(marker_line_width=0.5, marker_line_color="white")
            st.plotly_chart(fig_hist, use_container_width=True)
            st.markdown("A histogram of daily returns helps visualize skewness and kurtosis.")
        
        # 1.6 (Already covered) Benchmark Comparison
        # The main cumulative returns chart is effectively a benchmark comparison.
        # If you want an additional chart comparing difference, you can add it here.
        
    else:
        st.write("Select a mutual fund, pick a date range, and click **Generate Tearsheet**.")

if __name__ == "__main__":
    main()
