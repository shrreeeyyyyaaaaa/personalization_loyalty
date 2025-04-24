# personalization_loyalty_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

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

    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(offers[['customer_id', 'offer_id', 'redeemed']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    algo = SVD()
    algo.fit(trainset)

    customer_list = offers['customer_id'].unique().tolist()
    selected_customer = st.selectbox("Select a Customer", customer_list)
    offer_list = offers['offer_id'].unique()
    recommendations = [(offer, algo.predict(selected_customer, offer).est) for offer in offer_list]
    top_offers = sorted(recommendations, key=lambda x: x[1], reverse=True)[:5]

    st.subheader("Top 5 Recommended Offers")
    for offer, score in top_offers:
        st.write(f"Offer {offer} with score {score:.2f}")

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

    from sklearn.ensemble import RandomForestClassifier

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
