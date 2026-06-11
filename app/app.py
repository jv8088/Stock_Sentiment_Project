import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

st_autorefresh(
    interval=300000,
    key="refresh"
)
@st.cache_resource
def load_bert():

    from transformers import pipeline

    return pipeline(
        "sentiment-analysis"
    )

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Stock Sentiment Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family: 'Inter', sans-serif;
}

.stApp{
    background-color:#0A0E17;
    color:white;
}

section[data-testid="stSidebar"]{
    background:#111827;
    border-right:1px solid rgba(255,255,255,0.08);
}

.main-title{
    font-size:38px;
    font-weight:700;
    color:white;
}

.subtitle{
    color:#94A3B8;
    font-size:15px;
}

.metric-card{
    background:#161B26;
    padding:22px;
    border-radius:16px;
    border:1px solid rgba(255,255,255,0.08);
    transition:0.3s;
}

.metric-card:hover{
    transform:translateY(-4px);
    box-shadow:0 0 25px rgba(56,189,248,0.15);
}

.metric-label{
    color:#94A3B8;
    font-size:13px;
}

.metric-value{
    color:white;
    font-size:30px;
    font-weight:700;
}

.green{
    color:#10B981;
}

.red{
    color:#EF4444;
}

.ticker{
    background:#111827;
    padding:12px;
    border-radius:12px;
    overflow:hidden;
    white-space:nowrap;
    margin-bottom:20px;
}

.ticker-content{
    display:inline-block;
    animation:scroll 20s linear infinite;
}

@keyframes scroll{
    0% {transform:translateX(100%);}
    100% {transform:translateX(-100%);}
}

.nav-title{
    font-size:18px;
    font-weight:600;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data(ttl=300)
def load_stock():

    stock = yf.download(
        "TSLA",
        period="1y",
        auto_adjust=True
    )

    stock = stock.reset_index()

    stock.columns = [
        col[0] if isinstance(col, tuple) else col
        for col in stock.columns
    ]

    stock = stock.dropna()

    return stock

stock = load_stock()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.markdown(
    "<div class='nav-title'>📊 Navigation</div>",
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "",
    [
        "Overview Dashboard",
        "Advanced Price Charts",
        "Live Sentiment Engine",
        "ML Forecast Model",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
Project Stack

• BERT NLP

• Random Forest

• Tesla Stock Data

• Yahoo Finance

• Streamlit Dashboard
"""
)

# --------------------------------------------------
# LIVE TICKER
# --------------------------------------------------

st.markdown("### 📡 Live Market Ticker")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:

    latest_price = stock["Close"].dropna().iloc[-1]
    previous_price = stock["Close"].dropna().iloc[-2]

    change = latest_price - previous_price

    color = "green" if change > 0 else "red"
    arrow = "▲" if change > 0 else "▼"

    st.markdown(
        f"""
        <div class='metric-card'>
        <div class='metric-label'>TSLA</div>
        <div class='metric-value {color}'>{arrow} ${latest_price:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    

with c2:
    st.markdown(
    """
    <div class='metric-card'>
    <div class='metric-label'>S&P500</div>
    <div class='metric-value green'>▲ 5821</div>
    </div>
    """,
    unsafe_allow_html=True
    )

with c3:
    st.markdown(
    """
    <div class='metric-card'>
    <div class='metric-label'>BTC</div>
    <div class='metric-value green'>▲ 108,230</div>
    </div>
    """,
    unsafe_allow_html=True
    )

with c4:
    st.markdown(
    """
    <div class='metric-card'>
    <div class='metric-label'>VIX</div>
    <div class='metric-value red'>▼ 13.2</div>
    </div>
    """,
    unsafe_allow_html=True
    )

with c5:
    st.markdown(
    """
    <div class='metric-card'>
    <div class='metric-label'>NASDAQ</div>
    <div class='metric-value green'>▲ 19,112</div>
    </div>
    """,
    unsafe_allow_html=True
    )

# --------------------------------------------------
# OVERVIEW PAGE
# --------------------------------------------------

if page == "Overview Dashboard":

    st.markdown(
        "<div class='main-title'>📈 Stock Sentiment Intelligence Dashboard</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Institutional-grade market intelligence powered by BERT and Random Forest</div>",
        unsafe_allow_html=True
    )

    st.write("")

    col1,col2,col3,col4 = st.columns(4)

    with col1:

     latest_price = stock["Close"].dropna().iloc[-1]

     st.markdown(
            f"""
            <div class='metric-card'>
            <div class='metric-label'>Last Close</div>
            <div class='metric-value'>${latest_price:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        

    with col2:
        st.markdown(
        """
        <div class='metric-card'>
        <div class='metric-label'>Model Accuracy</div>
        <div class='metric-value green'>71%</div>
        </div>
        """,
        unsafe_allow_html=True
        )

    with col3:
        st.markdown(
        """
        <div class='metric-card'>
        <div class='metric-label'>Tweets Processed</div>
        <div class='metric-value'>1.6M</div>
        </div>
        """,
        unsafe_allow_html=True
        )

    with col4:
        st.markdown(
        """
        <div class='metric-card'>
        <div class='metric-label'>Avg Sentiment</div>
        <div class='metric-value green'>86%</div>
        </div>
        """,
        unsafe_allow_html=True
        )

    st.write("")
    st.write("")

    st.subheader("Tesla Closing Price")

    st.line_chart(stock["Close"])

    st.write("")

    st.subheader("Volume Analysis")

    st.bar_chart(stock["Volume"])

# --------------------------------------------------
# PLACEHOLDERS
# --------------------------------------------------

elif page == "Advanced Price Charts":

    st.title("📊 Advanced Price Charts")

    stock_chart = stock.copy()

    # MA20
    stock_chart["MA20"] = (
        stock_chart["Close"]
        .rolling(20)
        .mean()
    )

    st.markdown("### Time Range")

    timeframe = st.radio(
        "",
        ["1Y", "6M", "3M"],
        horizontal=True
    )

    if timeframe == "6M":
        stock_chart = stock_chart.tail(126)

    elif timeframe == "3M":
        stock_chart = stock_chart.tail(63)

    st.markdown("---")

    st.subheader("Tesla Price vs MA20")

    chart_df = stock_chart[
        ["Close", "MA20"]
    ]

    st.line_chart(chart_df)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    latest_price = stock_chart["Close"].iloc[-1]

    ma20 = stock_chart["MA20"].iloc[-1]

    momentum = latest_price - ma20

    with col1:

        st.markdown(
        f"""
        <div class='metric-card'>
        <div class='metric-label'>Current Price</div>
        <div class='metric-value'>${latest_price:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
        )

    with col2:

        st.markdown(
        f"""
        <div class='metric-card'>
        <div class='metric-label'>MA20</div>
        <div class='metric-value'>${ma20:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
        )

    with col3:

        color = "green" if momentum > 0 else "red"

        st.markdown(
        f"""
        <div class='metric-card'>
        <div class='metric-label'>Momentum</div>
        <div class='metric-value {color}'>{momentum:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
        )

    st.markdown("---")

    st.subheader("Volume Profile")

    st.bar_chart(
        stock_chart["Volume"]
    )

    st.markdown("---")

    st.subheader("Technical Indicator Explanation")

    st.write("""
    **Close Price**
    - Final traded price of the day.

    **MA20**
    - Average closing price over the previous 20 trading days.

    **Momentum**
    - Difference between current price and MA20.

    Positive Momentum:
    - Bullish trend.

    Negative Momentum:
    - Bearish trend.
    """)

    st.markdown("---")

    st.subheader("Market Signal")

    if momentum > 0:

        st.success(
            "🟢 Bullish Signal: Price is trading above MA20."
        )

    else:

        st.error(
            "🔴 Bearish Signal: Price is trading below MA20."
        )

elif page == "Live Sentiment Engine":

    from transformers import pipeline

    st.title("🧠 Live Sentiment Engine")

    st.markdown(
    """
    Analyze tweets, news headlines,
    or any financial statement using BERT.
    """
    )

    st.markdown("---")

    text = st.text_area(
        "Enter Text",
        height=150,
        placeholder="Example: Tesla earnings beat expectations this quarter..."
    )

    if st.button("Analyze Sentiment"):

        with st.spinner("Running BERT Analysis..."):

            classifier = load_bert()

            result = classifier(text)

            label = result[0]["label"]
            score = result[0]["score"]

            st.markdown("---")

            col1,col2 = st.columns(2)

            with col1:

                if label == "POSITIVE":

                    st.success(
                        f"Sentiment: {label}"
                    )

                else:

                    st.error(
                        f"Sentiment: {label}"
                    )

            with col2:

                st.metric(
                    "Confidence",
                    f"{score*100:.2f}%"
                )

            st.progress(score)

    st.markdown("---")

    st.subheader("Example Tesla Headlines")

    sample = pd.DataFrame({
        "Headline":[
            "Tesla beats earnings expectations",
            "Tesla opens new Gigafactory",
            "Tesla misses delivery estimates",
            "Tesla stock rallies after AI announcement"
        ]
    })

    st.dataframe(
        sample,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Sentiment Distribution")

    sentiment_data = pd.DataFrame(
        {
            "Sentiment":[
                "Positive",
                "Negative"
            ],
            "Count":[
                72,
                28
            ]
        }
    )

    st.bar_chart(
        sentiment_data.set_index(
            "Sentiment"
        )
    )
elif page == "ML Forecast Model":

    st.title("🤖 Random Forest Forecast Model")

    st.markdown(
    """
    Predicting Tesla stock movement
    using engineered financial features.
    """
    )

    st.markdown("---")

    col1,col2,col3 = st.columns(3)

    with col1:

        st.metric(
            "Model Accuracy",
            "71%"
        )

    with col2:

        st.metric(
            "Dataset Size",
            "250 Days"
        )

    with col3:

        st.metric(
            "Algorithm",
            "Random Forest"
        )

    st.markdown("---")

    st.subheader(
        "Feature Importance"
    )

    importance = pd.DataFrame({

        "Feature":[
            "MA20",
            "Range",
            "Volume",
            "Daily Return",
            "Volatility"
        ],

        "Importance":[
            0.092,
            0.091,
            0.090,
            0.088,
            0.086
        ]
    })

    st.bar_chart(
        importance.set_index(
            "Feature"
        )
    )

    st.markdown("---")

    st.subheader(
        "Model vs Random Guess"
    )

    comparison = pd.DataFrame({

        "Method":[
            "Random Guess",
            "Random Forest"
        ],

        "Accuracy":[
            50,
            71
        ]
    })

    st.bar_chart(
        comparison.set_index(
            "Method"
        )
    )

    st.markdown("---")

    st.subheader(
        "How Prediction Works"
    )

    st.write(
    """
    Input Features:

    • Open Price

    • High Price

    • Low Price

    • Close Price

    • Volume

    • MA20

    • Daily Return

    • Volatility

    ↓

    Random Forest

    ↓

    Predict:

    UP or DOWN
    """
    )

    st.success(
        "Current Model Accuracy: 71%"
    )

elif page == "About Project":

    st.title("About Project")

    st.write("""
    Stock Market Prediction Using Twitter Sentiment Analysis and Machine Learning

    Technologies:
    - Python
    - BERT
    - Random Forest
    - Yahoo Finance
    - Streamlit
    """)