import os
import glob
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Contextual Bandits Dashboard", layout="wide")

st.title("🎯 Netflix-style Personalization — Contextual Bandits Dashboard")
st.write("Load experiment CSVs from the **assets/** folder and explore CTR and regret across policies.")

assets_dir = st.sidebar.text_input("Assets directory", "assets")

csv_files = sorted(glob.glob(os.path.join(assets_dir, "results_*.csv")))
if not csv_files:
    st.warning("No results found. Run `python run_bandits.py` to generate CSVs in assets/.")
else:
    dataframes = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dataframes, ignore_index=True)
    policies = sorted(df["policy"].unique())
    sel_policies = st.multiselect("Select policies", policies, default=policies)
    df = df[df.policy.isin(sel_policies)]

    # CTR plot
    fig_ctr = px.line(df, x="step", y="ctr", color="policy", title="CTR over Time")
    st.plotly_chart(fig_ctr, use_container_width=True)

    # Regret plot
    fig_regret = px.line(df, x="step", y="regret", color="policy", title="Cumulative Regret (lower is better)")
    st.plotly_chart(fig_regret, use_container_width=True)

    # Summary table
    summary = df.groupby("policy").agg(avg_ctr=("click", "mean"),
                                       cumulative_reward=("click", "sum"),
                                       final_regret=("regret", "last")).reset_index()
    st.subheader("Summary")
    st.dataframe(summary)

    # Download summary
    st.download_button("Download summary.csv",
                       data=summary.to_csv(index=False).encode("utf-8"),
                       file_name="summary.csv",
                       mime="text/csv")
