# personalization_loyalty_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier

# Custom background styling
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background: linear-gradient(to right, #fdfbfb, #ebedee);
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background: #f8f9fa;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    transactions = pd.read_csv("transactions.csv", parse_dates=['transaction_date'])
    offers = pd.read_csv("offers.csv", parse_dates=['timestamp'])
    loyalty = pd.read_csv("loyalty.csv")
    return transactions, offers, loyalty

transactions, offers, loyalty = load_data()

# Merge loyalty info
transactions = transactions.merge(loyalty, on='customer_id', how='left')
offers = offers.merge(loyalty, on='customer_id', how='left')

# Sidebar Navigation
st.sidebar.title("🔍 Loyalty Optimization Tool")
section = st.sidebar.radio("Go to", ["Customer Segmentation", "Recommendations", "Loyalty Analysis", "Simulated Campaign"])

# --- Section 1: Customer Segmentation --- #
if section == "Customer Segmentation":
    st.title("📊 Customer Segmentation")

    snapshot_date = transactions['transaction_date'].max() + pd.Timedelta(days=1)
    rfm = transactions.groupby('customer_id').agg({
        'transaction_date': lambda x: (snapshot_date - x.max()).days,
        'customer_id': 'count',
        'amount': 'sum'
    }).rename(columns={'transaction_date': 'Recency', 'customer_id': 'Frequency', 'amount': 'Monetary'})

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Segment'] = kmeans.fit_predict(rfm_scaled)

    st.write("Segmented Customers:")
    st.dataframe(rfm.reset_index().head())

    fig = px.scatter(rfm, x='Recency', y='Monetary', color=rfm['Segment'].astype(str), title="Customer Segments")
    st.plotly_chart(fig)

# --- Section 2: Recommendations --- #
elif section == "Recommendations":
    st.title("🤖 Personalized Offer Recommendations")

    # Create a pivot table for offers
    pivot = offers.pivot_table(index='customer_id', columns='offer_id', values='redeemed', fill_value=0)
    
    # Compute similarity between customers
    similarity = cosine_similarity(pivot)
    similarity_df = pd.DataFrame(similarity, index=pivot.index, columns=pivot.index)

    selected_customer = st.selectbox("Select a Customer", pivot.index)

    # Find top similar customers
    sim_scores = similarity_df[selected_customer].sort_values(ascending=False)[1:6]
    top_customers = sim_scores.index

    # Aggregate offer scores from top customers
    recommended_scores = pivot.loc[top_customers].mean().sort_values(ascending=False)

    st.subheader("Top 5 Recommended Offers (Based on Similar Customers)")
    for offer_id, score in recommended_scores.head(5).items():
        st.write(f"Offer {offer_id} - estimated interest score: {score:.2f}")

# --- Section 3: Loyalty Program Analysis --- #
elif section == "Loyalty Analysis":
    st.title("🎁 Loyalty Program Analysis")

    loyalty_metrics = transactions.groupby('loyalty_member').agg({
        'amount': 'mean',
        'customer_id': 'count'
    }).rename(columns={'amount': 'Avg Spend', 'customer_id': 'Num Transactions'})

    st.write("Comparison of Loyalty vs Non-Loyalty Members")
    st.dataframe(loyalty_metrics)

    fig = px.bar(loyalty_metrics.reset_index(), x='loyalty_member', y='Avg Spend', title="Average Spend by Loyalty Status")
    st.plotly_chart(fig)

# --- Section 4: Simulated Campaign --- #
elif section == "Simulated Campaign":
    st.title("📬 Offer Campaign Simulator")

    snapshot_date = transactions['transaction_date'].max() + pd.Timedelta(days=1)
    rfm = transactions.groupby('customer_id').agg({
        'transaction_date': lambda x: (snapshot_date - x.max()).days,
        'customer_id': 'count',
        'amount': 'sum'
    }).rename(columns={'transaction_date': 'Recency', 'customer_id': 'Frequency', 'amount': 'Monetary'})
    rfm = rfm.reset_index().merge(offers.groupby('customer_id')['redeemed'].mean().reset_index(), on='customer_id', how='left')
    rfm['redeemed'] = rfm['redeemed'].fillna(0)

    model = RandomForestClassifier()
    model.fit(rfm[['Recency', 'Frequency', 'Monetary']], rfm['redeemed'])

    st.write("Simulate campaign for synthetic customer...")
    recency = st.slider("Recency (days since last purchase)", 1, 365, 60)
    frequency = st.slider("Frequency (no. of purchases)", 1, 50, 10)
    monetary = st.slider("Monetary (total spend)", 10, 2000, 300)

    prob = model.predict_proba([[recency, frequency, monetary]])[0][1]
    st.metric("Predicted Redemption Probability", f"{prob:.2%}")
