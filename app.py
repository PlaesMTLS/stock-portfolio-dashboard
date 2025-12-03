import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from supabase import create_client, Client
import numpy as np

# Supabase Configuration
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Page Configuration
st.set_page_config(page_title="Stock Portfolio Dashboard", layout="wide", page_icon="ðŸ“ˆ")

# Timeframe configurations
TIMEFRAMES = {
    "1D": {"period": "1d", "interval": "5m", "days": 1},
    "1W": {"period": "5d", "interval": "30m", "days": 5},
    "1M": {"period": "1mo", "interval": "1d", "days": 30},
    "3M": {"period": "3mo", "interval": "1d", "days": 90},
    "6M": {"period": "6mo", "interval": "1d", "days": 180},
    "YTD": {"period": "ytd", "interval": "1d", "days": None},
    "1Y": {"period": "1y", "interval": "1d", "days": 365},
    "5Y": {"period": "5y", "interval": "1wk", "days": 1825}
}

SECTOR_ETFS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
    "Communication": "XLC"
}

MARKET_INDICES = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW": "^DJI",
    "VIX": "^VIX"
}

# Sidebar Navigation
st.sidebar.title("ðŸ“Š Navigation")
page = st.sidebar.radio("Go to", [
    "Portfolio Overview",
    "Stock Deep Dive",
    "Sector Analysis",
    "Market Context",
    "News & Sentiment"
])

# Utility Functions
@st.cache_data(ttl=300)
def get_stock_data(ticker, period, interval):
    """Fetch stock data from yfinance"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        return data
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_portfolio_holdings():
    """Fetch portfolio holdings from Supabase"""
    try:
        response = supabase.table("portfolio_holdings").select("*").execute()
        return pd.DataFrame(response.data)
    except:
        return pd.DataFrame()

def get_news_by_timeframe(timeframe_days, category=None, ticker=None):
    """Fetch news filtered by timeframe from Supabase"""
    try:
        cutoff_date = datetime.now() - timedelta(days=timeframe_days if timeframe_days else 365)
        query = supabase.table("news_items").select("*, sentiment_analysis(*)").gte("published_date", cutoff_date.isoformat())
        
        if category:
            query = query.eq("category", category)
        if ticker:
            query = query.contains("relevant_tickers", [ticker])
        
        response = query.order("published_date", desc=True).limit(50).execute()
        return pd.DataFrame(response.data)
    except:
        return pd.DataFrame()

def calculate_performance(data):
    """Calculate performance metrics"""
    if len(data) < 2:
        return 0, 0
    start_price = data['Close'].iloc[0]
    end_price = data['Close'].iloc[-1]
    change = end_price - start_price
    change_pct = (change / start_price) * 100
    return change, change_pct

def create_price_chart(data, ticker, timeframe):
    """Create interactive price chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add moving averages for longer timeframes
    if len(data) > 20:
        ma20 = data['Close'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=ma20,
            mode='lines',
            name='MA20',
            line=dict(color='orange', width=1, dash='dash')
        ))
    
    if len(data) > 50:
        ma50 = data['Close'].rolling(window=50).mean()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=ma50,
            mode='lines',
            name='MA50',
            line=dict(color='red', width=1, dash='dash')
        ))
    
    fig.update_layout(
        title=f"{ticker} - {timeframe}",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=400
    )
    
    return fig

def get_sentiment_color(score):
    """Return color based on sentiment score"""
    if score > 0.2:
        return "ðŸŸ¢"
    elif score < -0.2:
        return "ðŸ”´"
    else:
        return "ðŸŸ¡"

# PAGE 1: PORTFOLIO OVERVIEW
if page == "Portfolio Overview":
    st.title("ðŸ“ˆ Portfolio Overview")
    
    # Timeframe Selector
    col1, col2 = st.columns([1, 4])
    with col1:
        selected_timeframe = st.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=3)
    
    timeframe_config = TIMEFRAMES[selected_timeframe]
    
    # Fetch Portfolio Holdings
    holdings = get_portfolio_holdings()
    
    if holdings.empty:
        st.warning("No portfolio holdings found. Please add stocks to your portfolio in Supabase.")
        st.info("Add holdings to the 'portfolio_holdings' table with columns: ticker, shares, cost_basis, sector")
    else:
        # Calculate Portfolio Metrics
        portfolio_value = 0
        portfolio_cost = 0
        stock_performances = []
        
        for _, holding in holdings.iterrows():
            ticker = holding['ticker']
            shares = holding['shares']
            cost_basis = holding['cost_basis']
            
            data = get_stock_data(ticker, timeframe_config['period'], timeframe_config['interval'])
            
            if not data.empty:
                current_price = data['Close'].iloc[-1]
                position_value = current_price * shares
                position_cost = cost_basis * shares
                
                portfolio_value += position_value
                portfolio_cost += position_cost
                
                change, change_pct = calculate_performance(data)
                
                stock_performances.append({
                    'Ticker': ticker,
                    'Shares': shares,
                    'Current Price': f"${current_price:.2f}",
                    'Position Value': f"${position_value:.2f}",
                    'Cost Basis': f"${cost_basis:.2f}",
                    'Total Cost': f"${position_cost:.2f}",
                    'Gain/Loss': f"${position_value - position_cost:.2f}",
                    'Return %': f"{((position_value - position_cost) / position_cost * 100):.2f}%",
                    f'{selected_timeframe} Change': f"{change_pct:.2f}%",
                    'Sector': holding.get('sector', 'Unknown')
                })
        
        # Portfolio Summary Cards
        total_return = portfolio_value - portfolio_cost
        total_return_pct = (total_return / portfolio_cost * 100) if portfolio_cost > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Portfolio Value", f"${portfolio_value:,.2f}")
        with col2:
            st.metric("Total Cost", f"${portfolio_cost:,.2f}")
        with col3:
            st.metric("Total Gain/Loss", f"${total_return:,.2f}", f"{total_return_pct:.2f}%")
        with col4:
            st.metric("Number of Holdings", len(holdings))
        
        # Portfolio Performance Table
        st.subheader("ðŸ“Š Holdings Performance")
        perf_df = pd.DataFrame(stock_performances)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
        
        # Sector Allocation
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ðŸ¥§ Sector Allocation")
            sector_values = {}
            for _, holding in holdings.iterrows():
                sector = holding.get('sector', 'Unknown')
                ticker = holding['ticker']
                shares = holding['shares']
                
                data = get_stock_data(ticker, "1d", "1d")
                if not data.empty:
                    current_price = data['Close'].iloc[-1]
                    sector_values[sector] = sector_values.get(sector, 0) + (current_price * shares)
            
            sector_df = pd.DataFrame(list(sector_values.items()), columns=['Sector', 'Value'])
            fig = px.pie(sector_df, values='Value', names='Sector', title='Portfolio by Sector')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("ðŸ“ˆ Portfolio Value Chart")
            # Aggregate portfolio performance over timeframe
            portfolio_history = None
            for _, holding in holdings.iterrows():
                ticker = holding['ticker']
                shares = holding['shares']
                data = get_stock_data(ticker, timeframe_config['period'], timeframe_config['interval'])
                
                if not data.empty:
                    position_value = data['Close'] * shares
                    if portfolio_history is None:
                        portfolio_history = position_value
                    else:
                        portfolio_history = portfolio_history.add(position_value, fill_value=0)
            
            if portfolio_history is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=portfolio_history.index, y=portfolio_history.values, 
                                        mode='lines', fill='tozeroy', name='Portfolio Value'))
                fig.update_layout(title=f"Portfolio Value - {selected_timeframe}", 
                                xaxis_title="Date", yaxis_title="Value ($)", height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Recent News Feed
        st.subheader("ðŸ“° Recent News & Sentiment")
        days = timeframe_config.get('days', 365)
        news_df = get_news_by_timeframe(days)
        
        if not news_df.empty:
            for _, news in news_df.head(10).iterrows():
                sentiment_data = news.get('sentiment_analysis', [{}])[0] if news.get('sentiment_analysis') else {}
                sentiment_score = sentiment_data.get('sentiment_score', 0)
                sentiment_label = sentiment_data.get('sentiment_label', 'neutral')
                
                with st.expander(f"{get_sentiment_color(sentiment_score)} {news['title'][:100]}..."):
                    st.write(f"**Source:** {news.get('source', 'Unknown')}")
                    st.write(f"**Published:** {news.get('published_date', 'Unknown')}")
                    st.write(f"**Category:** {news.get('category', 'Unknown')}")
                    st.write(f"**Sentiment:** {sentiment_label.upper()} ({sentiment_score:.2f})")
                    st.write(f"**Summary:** {news.get('summary', 'No summary available')}")
                    if news.get('url'):
                        st.write(f"[Read More]({news['url']})")
        else:
            st.info("No news available for selected timeframe")

# PAGE 2: STOCK DEEP DIVE
elif page == "Stock Deep Dive":
    st.title("ðŸ” Stock Deep Dive")
    
    holdings = get_portfolio_holdings()
    
    if holdings.empty:
        st.warning("No portfolio holdings found.")
    else:
        # Stock Selector
        selected_stock = st.selectbox("Select Stock", holdings['ticker'].tolist())
        
        # Multi-timeframe view
        st.subheader(f"ðŸ“Š {selected_stock} - Multi-Timeframe Analysis")
        
        tabs = st.tabs(list(TIMEFRAMES.keys()))
        
        for i, (tf_name, tf_config) in enumerate(TIMEFRAMES.items()):
            with tabs[i]:
                data = get_stock_data(selected_stock, tf_config['period'], tf_config['interval'])
                
                if not data.empty:
                    change, change_pct = calculate_performance(data)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
                    with col2:
                        st.metric("Change", f"${change:.2f}", f"{change_pct:.2f}%")
                    with col3:
                        st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
                    
                    # Price Chart
                    fig = create_price_chart(data, selected_stock, tf_name)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Technical Indicators
                    if len(data) > 14:
                        st.subheader("Technical Indicators")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # RSI
                            delta = data['Close'].diff()
                            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                            rs = gain / loss
                            rsi = 100 - (100 / (1 + rs))
                            st.metric("RSI (14)", f"{rsi.iloc[-1]:.2f}")
                        
                        with col2:
                            # Volatility
                            volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
                            st.metric("Volatility (Annualized)", f"{volatility:.2f}%")
                else:
                    st.error(f"Unable to fetch data for {selected_stock}")
        
        # Stock-Specific News
        st.subheader(f"ðŸ“° News for {selected_stock}")
        news_df = get_news_by_timeframe(30, ticker=selected_stock)
        
        if not news_df.empty:
            for _, news in news_df.iterrows():
                sentiment_data = news.get('sentiment_analysis', [{}])[0] if news.get('sentiment_analysis') else {}
                sentiment_score = sentiment_data.get('sentiment_score', 0)
                sentiment_label = sentiment_data.get('sentiment_label', 'neutral')
                
                with st.expander(f"{get_sentiment_color(sentiment_score)} {news['title']}"):
                    st.write(f"**Published:** {news.get('published_date', 'Unknown')}")
                    st.write(f"**Sentiment:** {sentiment_label.upper()} ({sentiment_score:.2f})")
                    st.write(f"**Summary:** {news.get('summary', 'No summary available')}")
                    if news.get('url'):
                        st.write(f"[Read More]({news['url']})")
        else:
            st.info(f"No recent news found for {selected_stock}")

# PAGE 3: SECTOR ANALYSIS
elif page == "Sector Analysis":
    st.title("ðŸ­ Sector Analysis")
    
    selected_timeframe = st.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=3)
    timeframe_config = TIMEFRAMES[selected_timeframe]
    
    # Fetch Sector ETF Performance
    sector_performance = {}
    
    for sector, etf in SECTOR_ETFS.items():
        data = get_stock_data(etf, timeframe_config['period'], timeframe_config['interval'])
        if not data.empty:
            change, change_pct = calculate_performance(data)
            sector_performance[sector] = {
                'ETF': etf,
                'Change %': change_pct,
                'Current Price': data['Close'].iloc[-1]
            }
    
    # Sector Heatmap
    st.subheader(f"ðŸ”¥ Sector Performance Heatmap - {selected_timeframe}")
    
    sector_df = pd.DataFrame(sector_performance).T.reset_index()
    sector_df.columns = ['Sector', 'ETF', 'Change %', 'Current Price']
    sector_df = sector_df.sort_values('Change %', ascending=False)
    
    fig = px.bar(sector_df, x='Sector', y='Change %', color='Change %',
                 color_continuous_scale=['red', 'yellow', 'green'],
                 title=f'Sector Performance - {selected_timeframe}')
    st.plotly_chart(fig, use_container_width=True)
    
    # Sector Performance Table
    st.dataframe(sector_df, use_container_width=True, hide_index=True)
    
    # Your Holdings vs Sector
    st.subheader("ðŸ“Š Your Holdings vs Sector Benchmarks")
    holdings = get_portfolio_holdings()
    
    if not holdings.empty:
        comparison_data = []
        
        for sector in holdings['sector'].unique():
            if pd.isna(sector) or sector == 'Unknown':
                continue
            
            sector_stocks = holdings[holdings['sector'] == sector]
            sector_etf = SECTOR_ETFS.get(sector)
            
            if sector_etf:
                etf_data = get_stock_data(sector_etf, timeframe_config['period'], timeframe_config['interval'])
                _, etf_change_pct = calculate_performance(etf_data)
                
                # Calculate average performance of holdings in this sector
                holdings_performance = []
                for _, holding in sector_stocks.iterrows():
                    stock_data = get_stock_data(holding['ticker'], timeframe_config['period'], 
                                               timeframe_config['interval'])
                    if not stock_data.empty:
                        _, change_pct = calculate_performance(stock_data)
                        holdings_performance.append(change_pct)
                
                if holdings_performance:
                    avg_holdings_perf = np.mean(holdings_performance)
                    comparison_data.append({
                        'Sector': sector,
                        'Your Holdings Avg %': avg_holdings_perf,
                        'Sector ETF %': etf_change_pct,
                        'Outperformance': avg_holdings_perf - etf_change_pct
                    })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Your Holdings', x=comp_df['Sector'], 
                                y=comp_df['Your Holdings Avg %']))
            fig.add_trace(go.Bar(name='Sector ETF', x=comp_df['Sector'], 
                                y=comp_df['Sector ETF %']))
            fig.update_layout(title='Your Holdings vs Sector Benchmarks', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    # Sector News
    st.subheader("ðŸ“° Sector News")
    selected_sector = st.selectbox("Select Sector", list(SECTOR_ETFS.keys()))
    
    news_df = get_news_by_timeframe(timeframe_config.get('days', 365), category='sector')
    
    if not news_df.empty:
        # Filter by sector (simplified - in practice, you'd tag news with sectors)
        for _, news in news_df.head(10).iterrows():
            sentiment_data = news.get('sentiment_analysis', [{}])[0] if news.get('sentiment_analysis') else {}
            sentiment_score = sentiment_data.get('sentiment_score', 0)
            
            with st.expander(f"{get_sentiment_color(sentiment_score)} {news['title']}"):
                st.write(f"**Summary:** {news.get('summary', 'No summary available')}")
                if news.get('url'):
                    st.write(f"[Read More]({news['url']})")

# PAGE 4: MARKET CONTEXT
elif page == "Market Context":
    st.title("ðŸŒ Market Context")
    
    selected_timeframe = st.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=3)
    timeframe_config = TIMEFRAMES[selected_timeframe]
    
    # Major Indices Performance
    st.subheader(f"ðŸ“ˆ Major Indices - {selected_timeframe}")
    
    indices_data = {}
    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]
    
    for i, (name, symbol) in enumerate(MARKET_INDICES.items()):
        data = get_stock_data(symbol, timeframe_config['period'], timeframe_config['interval'])
        if not data.empty:
            change, change_pct = calculate_performance(data)
            current_price = data['Close'].iloc[-1]
            indices_data[name] = data
            
            with cols[i]:
                st.metric(name, f"{current_price:.2f}", f"{change_pct:.2f}%")
    
    # Market Indices Charts
    for name, data in indices_data.items():
        fig = create_price_chart(data, name, selected_timeframe)
        st.plotly_chart(fig, use_container_width=True)
    
    # Market Breadth Indicators
    st.subheader("ðŸ“Š Market Breadth")
    col1, col2 = st.columns(2)
    
    with col1:
        # VIX Analysis
        vix_data = get_stock_data("^VIX", timeframe_config['period'], timeframe_config['interval'])
        if not vix_data.empty:
            current_vix = vix_data['Close'].iloc[-1]
            if current_vix < 15:
                vix_sentiment = "Low Volatility (Complacent)"
            elif current_vix < 20:
                vix_sentiment = "Normal Volatility"
            elif current_vix < 30:
                vix_sentiment = "Elevated Volatility (Caution)"
            else:
                vix_sentiment = "High Volatility (Fear)"
            
            st.metric("VIX Fear Gauge", f"{current_vix:.2f}", vix_sentiment)
    
    with col2:
        # Market Trend
        sp500_data = get_stock_data("^GSPC", timeframe_config['period'], timeframe_config['interval'])
        if not sp500_data.empty and len(sp500_data) > 50:
            ma50 = sp500_data['Close'].rolling(window=50).mean().iloc[-1]
            current_price = sp500_data['Close'].iloc[-1]
            trend = "Bullish" if current_price > ma50 else "Bearish"
            st.metric("S&P 500 Trend", trend, f"Price vs MA50: {((current_price/ma50 - 1) * 100):.2f}%")
    
    # Market-Wide News
    st.subheader("ðŸ“° Market News & Sentiment")
    news_df = get_news_by_timeframe(timeframe_config.get('days', 365), category='market')
    
    if not news_df.empty:
        # Aggregate sentiment
        sentiment_scores = []
        for _, news in news_df.iterrows():
            sentiment_data = news.get('sentiment_analysis', [{}])[0] if news.get('sentiment_analysis') else {}
            sentiment_score = sentiment_data.get('sentiment_score', 0)
            if sentiment_score != 0:
                sentiment_scores.append(sentiment_score)
        
        if sentiment_scores:
            avg_sentiment = np.mean(sentiment_scores)
            sentiment_label = "Bullish" if avg_sentiment > 0.1 else "Bearish" if avg_sentiment < -0.1 else "Neutral"
            st.metric("Overall Market Sentiment", sentiment_label, f"Score: {avg_sentiment:.2f}")
        
        # Display news
        for _, news in news_df.head(15).iterrows():
            sentiment_data = news.get('sentiment_analysis', [{}])[0] if news.get('sentiment_analysis') else {}
            sentiment_score = sentiment_data.get('sentiment_score', 0)
            sentiment_label = sentiment_data.get('sentiment_label', 'neutral')
            
            with st.expander(f"{get_sentiment_color(sentiment_score)} {news['title']}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Published:** {news.get('published_date', 'Unknown')}")
                    st.write(f"**Source:** {news.get('source', 'Unknown')}")
                    st.write(f"**Category:** {news.get('category', 'Unknown')}")
                with col2:
                    st.write(f"**Sentiment:** {sentiment_label.upper()}")
                    st.write(f"**Score:** {sentiment_score:.2f}")
                    confidence = sentiment_data.get('confidence_score', 0)
                    st.write(f"**Confidence:** {confidence:.2f}")
                
                st.write(f"**Summary:** {news.get('summary', 'No summary available')}")
                
                if news.get('relevant_tickers'):
                    st.write(f"**Tickers:** {', '.join(news.get('relevant_tickers', []))}")
                
                if news.get('url'):
                    st.write(f"[Read Full Article]({news['url']})")
    else:
        st.info("No news available for the selected filters")

        with st.expander(f"{get_sentiment_color(sentiment_score)} {news['title']}"):
            st.write(f"**Published:** {news.get('published_date', 'Unknown')}")
            st.write(f"**Sentiment:** {sentiment_label.upper()} ({sentiment_score:.2f})")
            st.write(f"**Summary:** {news.get('summary', 'No summary available')}")
            if news.get('url'):
                st.write(f"[Read More]({news['url']})")
    else:
        st.info("No market news available")

# PAGE 5: NEWS & SENTIMENT HUB
elif page == "News & Sentiment":
    st.title("ðŸ“° News & Sentiment Hub")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_timeframe = st.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=3)
    with col2:
        category_filter = st.selectbox("Category", ["All", "stock", "sector", "market", "macro"])
    with col3:
        sentiment_filter = st.selectbox("Sentiment", ["All", "positive", "negative", "neutral"])
    
    timeframe_config = TIMEFRAMES[selected_timeframe]
    days = timeframe_config.get('days', 365)
    
    # Fetch News
    category = None if category_filter == "All" else category_filter
    news_df = get_news_by_timeframe(days, category=category)
    
    if not news_df.empty:
        # Filter by sentiment if needed
        if sentiment_filter != "All":
            filtered_news = []
            for _, news in news_df.iterrows():
                sentiment_data = news.get('sentiment_analysis', [{}])[0] if news.get('sentiment_analysis') else {}
                sentiment_label = sentiment_data.get('sentiment_label', 'neutral')
                if sentiment_label == sentiment_filter:
                    filtered_news.append(news)
            news_df = pd.DataFrame(filtered_news)
        
        # Sentiment Trend Over Time
        st.subheader("ðŸ“ˆ Sentiment Trend")
        sentiment_timeline = []
        for _, news in news_df.iterrows():
            sentiment_data = news.get('sentiment_analysis', [{}])[0] if news.get('sentiment_analysis') else {}
            sentiment_score = sentiment_data.get('sentiment_score', 0)
            pub_date = news.get('published_date')
            if pub_date and sentiment_score != 0:
                sentiment_timeline.append({
                    'Date': pd.to_datetime(pub_date),
                    'Sentiment': sentiment_score
                })
        
        if sentiment_timeline:
            sent_df = pd.DataFrame(sentiment_timeline).sort_values('Date')
            sent_df['MA7'] = sent_df['Sentiment'].rolling(window=7, min_periods=1).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sent_df['Date'], y=sent_df['Sentiment'], 
                                    mode='markers', name='Sentiment Score', opacity=0.5))
            fig.add_trace(go.Scatter(x=sent_df['Date'], y=sent_df['MA7'], 
                                    mode='lines', name='7-Day MA', line=dict(color='red', width=2)))
            fig.update_layout(title='Sentiment Trend Over Time', xaxis_title='Date', 
                            yaxis_title='Sentiment Score', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            for _, news in news_df.iterrows():
                sentiment_data = news.get('sentiment_analysis', [{}])[0] if news.get('sentiment_analysis') else {}
                sentiment_label = sentiment_data.get('sentiment_label', 'neutral')
                sentiment_counts[sentiment_label] += 1
            
            fig = px.pie(values=list(sentiment_counts.values()), names=list(sentiment_counts.keys()),
                        title='Sentiment Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Category Distribution
            category_counts = news_df['category'].value_counts().to_dict()
            fig = px.pie(values=list(category_counts.values()), names=list(category_counts.keys()),
                        title='News by Category')
            st.plotly_chart(fig, use_container_width=True)
        
# News Feed
        st.subheader(f"ðŸ“‹ News Feed ({len(news_df)} articles)")
        
        for _, news in news_df.iterrows():
            sentiment_data = news.get('sentiment_analysis', [{}])[0] if news.get('sentiment_analysis') else {}
            sentiment_score = sentiment_data.get('sentiment_score', 0)
            sentiment_label = sentiment_data.get('sentiment_label', 'neutral')
            
            with st.expander(f"{get_sentiment_color(sentiment_score)} {news['title']}"):
                st.write(f"**Published:** {news.get('published_date', 'Unknown')}")
                st.write(f"**Sentiment:** {sentiment_label.upper()} ({sentiment_score:.2f})")
                st.write(f"**Summary:** {news.get('summary', 'No summary available')}")
                if news.get('url'):
                    st.write(f"[Read More]({news['url']})")
    else:
        st.info("No news available for the selected filters")
		
# Footer
st.sidebar.markdown("---")
st.sidebar.info("ðŸ’¡ **Tip:** Data refreshes every 5 minutes. Use the refresh button in your browser to get the latest updates.")
