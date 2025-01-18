import streamlit as st
import pandas as pd
import datetime

@st.cache_data
def load_regulatory_data(csv_path="regulatory_data.csv"):
    """
    Loads cross-border regulatory data from a CSV.
    Adjust or extend this function if reading from a DB or API.
    """
    df = pd.read_csv(csv_path)
    # Optionally parse dates
    if "Last_Updated" in df.columns:
        df["Last_Updated"] = pd.to_datetime(df["Last_Updated"], errors="coerce")
    return df

def main():
    st.set_page_config(page_title="Cross-Border Compliance", layout="wide")
    st.title("Cross-Border Compliance Portal")

    # --- Detailed Explanation of Cross-Border Compliance ---
    st.markdown("""
    Cross-border compliance refers to the adherence of organizations, financial institutions,
    and individuals to international regulations and laws that govern cross-border transactions,
    data exchanges, and other operations across multiple jurisdictions. 
    Ensuring compliance typically involves:
    
    - **Anti-Money Laundering (AML)** thresholds and reporting requirements  
    - **Data Privacy** regulations (e.g., GDPR, local privacy laws)  
    - **Taxation rules** and cross-border remittance limits  
    - **Licensing** and operational permits for specific industries  
    - **Sanctions** or restricted-party checks
    
    Failing to comply with these requirements can lead to penalties, legal repercussions,
    and reputational damage. This portal provides insights into various countries' regulations
    and helps you quickly check AML thresholds to determine if further due diligence
    or reporting is required for a given cross-border transaction.
    """)

    # --- Load the CSV data ---
    try:
        regulatory_df = load_regulatory_data("regulatory_data.csv")
    except FileNotFoundError:
        st.error("Could not find 'regulatory_data.csv'. Please ensure it's in the same directory as this app.")
        return

    if regulatory_df.empty:
        st.warning("The regulatory CSV is empty. Please populate 'regulatory_data.csv' with real data.")
        return

    # --- REGULATORY INSIGHTS SECTION ---
    st.header("Regulatory Insights")
    st.write("""
    Filter regulatory data by a specific country or date range to see the most up-to-date
    thresholds and relevant information.
    """)

    all_countries = ["All"] + list(regulatory_df["Country"].unique())
    selected_country = st.selectbox("Select a Country", all_countries)

    # Optional date filter
    st.write("Filter by Last Updated Date")
    min_date = st.date_input("Start Date (YYYY-MM-DD)", datetime.date(2022, 1, 1))
    max_date = st.date_input("End Date (YYYY-MM-DD)", datetime.date.today())

    # Filter logic
    filtered_df = regulatory_df.copy()
    if selected_country != "All":
        filtered_df = filtered_df[filtered_df["Country"] == selected_country]

    if "Last_Updated" in filtered_df.columns:
        # Convert min_date, max_date to datetime for comparison
        min_dt = pd.to_datetime(min_date)
        max_dt = pd.to_datetime(max_date)
        filtered_df = filtered_df[
            (filtered_df["Last_Updated"] >= min_dt) & (filtered_df["Last_Updated"] <= max_dt)
        ]

    if filtered_df.empty:
        st.warning("No regulatory data matches the filters.")
    else:
        st.subheader(f"Regulatory Data ({len(filtered_df)} records)")
        st.dataframe(filtered_df.reset_index(drop=True))

        # Example: Provide a download button for the filtered data
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data",
            data=csv_data,
            file_name="filtered_regulatory_data.csv",
            mime="text/csv",
        )

    # --- COMPLIANCE CHECKER SECTION ---
    st.header("Compliance Checker")
    st.write("""
    Enter details about your cross-border transaction to see if it exceeds
    a given country's AML threshold (from our data).
    """)

    transaction_value = st.number_input("Transaction Value (USD)", min_value=0, step=1000)
    country_from = st.text_input("Origin Country", "USA")
    country_to = st.text_input("Destination Country", "UK")

    if st.button("Run Check"):
        # Find row(s) in the CSV for the destination country
        row = regulatory_df[regulatory_df["Country"].str.upper() == country_to.upper()]

        if row.empty:
            st.error(f"No regulatory data found for destination: {country_to}")
        else:
            aml_threshold = row["AML_Threshold"].values[0]
            st.write(f"**AML Threshold for {country_to}** is {aml_threshold} USD.")

            if transaction_value > aml_threshold:
                st.warning("Transaction exceeds the AML threshold. Additional checks required.")
            else:
                st.success("Transaction is below AML threshold. Likely compliant.")

if __name__ == "__main__":
    main()
