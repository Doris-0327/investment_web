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

    if len(daily_returns) == 0:
        return None

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

    # 避免除以0
    if volatility == 0:
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = (
            annual_return - rf_rate
        ) / volatility

    # =========================
    # 最大回撤
    # =========================

    cumulative = (1 + daily_returns).cumprod()

    rolling_max = cumulative.cummax()

    drawdown = (
        cumulative - rolling_max
    ) / rolling_max

    max_drawdown = drawdown.min()

    # =========================
    # Sortino Ratio
    # =========================

    downside = daily_returns[daily_returns < 0]

    downside_std = downside.std() * np.sqrt(252)

    if downside_std == 0 or np.isnan(downside_std):
        sortino_ratio = np.nan
    else:
        sortino_ratio = (
            annual_return - rf_rate
        ) / downside_std

    # =========================
    # Calmar Ratio
    # =========================

    if max_drawdown == 0:
        calmar_ratio = np.nan
    else:
        calmar_ratio = annual_return / abs(max_drawdown)

    return {
        "總報酬率": float(cumulative_return),
        "年化報酬率": float(annual_return),
        "波動率": float(volatility),
        "Sharpe Ratio": float(sharpe_ratio),
        "最大回撤": float(max_drawdown),
        "Sortino Ratio": float(sortino_ratio),
        "Calmar Ratio": float(calmar_ratio)
    }

# =========================
# 開始分析
# =========================

if st.button("開始分析"):

    tickers = [
        t.strip()
        for t in tickers_input.split(",")
        if t.strip() != ""
    ]

    benchmark = "^TWII"

    # =========================
    # Benchmark
    # =========================

    benchmark_df = yf.download(
        benchmark,
        start=start_date,
        end=end_date,
        auto_adjust=True
    )

    if benchmark_df.empty:
        st.error("Benchmark 無法下載")
        st.stop()

    benchmark_data = benchmark_df["Close"]

    benchmark_returns = benchmark_data.pct_change().dropna()

    all_results = []

    fig = go.Figure()

    # =========================
    # 每支股票分析
    # =========================

    for ticker in tickers:

        try:

            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True
            )

            # 無資料
            if data.empty:
                st.warning(f"{ticker} 無資料")
                continue

            prices = data["Close"]

            if prices.empty:
                st.warning(f"{ticker} 價格資料為空")
                continue

            # =========================
            # 計算指標
            # =========================

            metrics = calculate_metrics(
                prices,
                risk_free_rate / 100
            )

            if metrics is None:
                st.warning(f"{ticker} 無法計算")
                continue

            # =========================
            # Beta / Alpha
            # =========================

            stock_returns = prices.pct_change().dropna()

            aligned = pd.concat(
                [stock_returns, benchmark_returns],
                axis=1
            ).dropna()

            if len(aligned) > 10:

                slope, intercept, _, _, _ = linregress(
                    aligned.iloc[:, 1],
                    aligned.iloc[:, 0]
                )

                beta = float(slope)
                alpha = float(intercept * 252)

            else:
                beta = np.nan
                alpha = np.nan

            metrics["Beta"] = beta
            metrics["Alpha"] = alpha

            metrics["股票"] = ticker

            all_results.append(metrics)

            # =========================
            # 累積報酬圖
            # =========================

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

            st.error(f"{ticker} 發生錯誤：{str(e)}")

    # =========================
    # 無結果
    # =========================

    if len(all_results) == 0:
        st.error("沒有可分析資料")
        st.stop()

    # =========================
    # DataFrame
    # =========================

    result_df = pd.DataFrame(all_results)

    # =========================
    # 百分比欄位
    # =========================

    percentage_cols = [
        "總報酬率",
        "年化報酬率",
        "波動率",
        "最大回撤",
        "Alpha"
    ]

    for col in percentage_cols:

        result_df[col] = pd.to_numeric(
            result_df[col],
            errors="coerce"
        )

        result_df[col] = (
            result_df[col] * 100
        ).round(2).astype(str) + "%"

    # =========================
    # 數值欄位
    # =========================

    numeric_cols = [
        "Sharpe Ratio",
        "Sortino Ratio",
        "Calmar Ratio",
        "Beta"
    ]

    for col in numeric_cols:

        result_df[col] = pd.to_numeric(
            result_df[col],
            errors="coerce"
        )

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
    # 圖表設定
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
    # 長期持有模擬
    # =========================

    st.subheader("💰 長期持有資產變化")

    for ticker in tickers:

        try:

            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True
            )

            if data.empty:
                continue

            prices = data["Close"]

            if prices.empty:
                continue

            shares = initial_money / prices.iloc[0]

            final_value = shares * prices.iloc[-1]

            total_profit = final_value - initial_money

            total_return = (
                final_value / initial_money - 1
            ) * 100

            st.markdown(f"""
            ### {ticker}

            - 初始資金：{initial_money:,.0f}
            - 最終資產：{final_value:,.0f}
            - 總獲利：{total_profit:,.0f}
            - 投資報酬率：{total_return:.2f}%
            """)

        except Exception as e:

            st.warning(f"{ticker} 長期持有模擬失敗")
