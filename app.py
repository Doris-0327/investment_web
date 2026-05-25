import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress

# =========================
# 頁面設定
# =========================

st.set_page_config(
    page_title="股票績效分析平台",
    layout="wide"
)

st.title("📈 股票績效分析平台")

# =========================
# 使用者輸入
# =========================

col1, col2, col3 = st.columns(3)

with col1:
    tickers_input = st.text_input(
        "股票代號（逗號分隔）",
        "2330.TW,2454.TW,AAPL"
    )

with col2:
    start_date = st.date_input(
        "開始日期",
        pd.to_datetime("2020-01-01")
    )

with col3:
    end_date = st.date_input(
        "結束日期",
        pd.to_datetime("today")
    )

risk_free_rate = st.number_input(
    "無風險利率 (%)",
    value=1.5,
    step=0.1
)

initial_money = st.number_input(
    "初始資金",
    value=100000
)

# =========================
# 計算函數
# =========================

def calculate_metrics(prices, rf_rate=0.015):

    daily_returns = prices.pct_change().dropna()

    cumulative_return = (
        prices.iloc[-1] / prices.iloc[0]
    ) - 1

    annual_return = (
        (1 + cumulative_return)
        ** (252 / len(daily_returns))
    ) - 1

    volatility = (
        daily_returns.std() * np.sqrt(252)
    )

    sharpe_ratio = (
        annual_return - rf_rate
    ) / volatility

    # 最大回撤
    cumulative = (1 + daily_returns).cumprod()

    rolling_max = cumulative.cummax()

    drawdown = (
        cumulative - rolling_max
    ) / rolling_max

    max_drawdown = drawdown.min()

    # Sortino Ratio
    downside = daily_returns[daily_returns < 0]

    downside_std = downside.std() * np.sqrt(252)

    sortino_ratio = (
        annual_return - rf_rate
    ) / downside_std

    # Calmar Ratio
    calmar_ratio = annual_return / abs(max_drawdown)

    return {
        "總報酬率": cumulative_return,
        "年化報酬率": annual_return,
        "波動率": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "最大回撤": max_drawdown,
        "Sortino Ratio": sortino_ratio,
        "Calmar Ratio": calmar_ratio
    }

# =========================
# 開始分析
# =========================

if st.button("開始分析"):

    tickers = [
        t.strip()
        for t in tickers_input.split(",")
    ]

    benchmark = "^TWII"

    benchmark_data = yf.download(
        benchmark,
        start=start_date,
        end=end_date
    )["Close"]

    benchmark_returns = benchmark_data.pct_change().dropna()

    all_results = []

    fig = go.Figure()

    for ticker in tickers:

        try:

            data = yf.download(
                ticker,
                start=start_date,
                end=end_date
            )

            prices = data["Close"]

            metrics = calculate_metrics(
                prices,
                risk_free_rate / 100
            )

            # Beta / Alpha
            stock_returns = prices.pct_change().dropna()

            aligned = pd.concat(
                [stock_returns, benchmark_returns],
                axis=1
            ).dropna()

            slope, intercept, _, _, _ = linregress(
                aligned.iloc[:,1],
                aligned.iloc[:,0]
            )

            beta = slope
            alpha = intercept * 252

            metrics["Beta"] = beta
            metrics["Alpha"] = alpha

            metrics["股票"] = ticker

            all_results.append(metrics)

            # 累積報酬圖
            cumulative = (
                prices / prices.iloc[0]
            )

            fig.add_trace(
                go.Scatter(
                    x=cumulative.index,
                    y=cumulative.values,
                    mode="lines",
                    name=ticker
                )
            )

        except Exception as e:
            st.error(f"{ticker} 發生錯誤：{e}")

    # =========================
    # DataFrame
    # =========================

    result_df = pd.DataFrame(all_results)

    percentage_cols = [
        "總報酬率",
        "年化報酬率",
        "波動率",
        "最大回撤",
        "Alpha"
    ]

    for col in percentage_cols:
        result_df[col] = (
            result_df[col] * 100
        ).round(2).astype(str) + "%"

    numeric_cols = [
        "Sharpe Ratio",
        "Sortino Ratio",
        "Calmar Ratio",
        "Beta"
    ]

    for col in numeric_cols:
        result_df[col] = result_df[col].round(3)

    # =========================
    # 顯示結果
    # =========================

    st.subheader("📊 分析結果")

    st.dataframe(
        result_df,
        use_container_width=True
    )

    # =========================
    # 圖表
    # =========================

    fig.update_layout(
        title="累積報酬率",
        xaxis_title="日期",
        yaxis_title="累積報酬倍數",
        template="plotly_white",
        height=600
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =========================
    # 投資組合模擬
    # =========================

    st.subheader("💰 長期持有資產變化")

    for ticker in tickers:

        data = yf.download(
            ticker,
            start=start_date,
            end=end_date
        )

        prices = data["Close"]

        shares = initial_money / prices.iloc[0]

        final_value = shares * prices.iloc[-1]

        total_profit = final_value - initial_money

        st.write(f"""
        ### {ticker}

        - 初始資金：{initial_money:,.0f}
        - 最終資產：{final_value:,.0f}
        - 總獲利：{total_profit:,.0f}
        """)