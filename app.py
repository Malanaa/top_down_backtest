from __future__ import annotations

from datetime import date, datetime, timedelta
from io import BytesIO
from typing import Dict, Iterable, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from openpyxl.utils import get_column_letter


# ============================================================
# Top Down Fun Backtest
# Streamlit app using yfinance
# ============================================================

st.set_page_config(page_title="top down fun backtest", layout="wide")
st.title("top down fun backtest")
st.caption("Interactive portfolio vs benchmark backtest using yfinance")


# -------------------------
# Default inputs
# -------------------------
DEFAULT_PORTFOLIO = pd.DataFrame(
    {
        "Ticker": [
            "VGT",
            "XLY",
            "XLC",
            "SCHH",
            "XLE",
            "XLB",
            "XLU",
            "XLI",
            "XLF",
            "XLV",
            "XLP",
            "SCHO",
            "SCHP",
            "VCSH",
            "HYDB",
            "VMBS",
            "BKLN",
            "IAU",
            "SIVR",
            "CASHX",
        ],
        "Weight": [
            16.47,
            3.70,
            5.98,
            1.14,
            3.99,
            2.20,
            1.42,
            3.24,
            8.12,
            6.59,
            5.08,
            7.50,
            9.00,
            3.00,
            3.00,
            6.00,
            1.50,
            5.50,
            2.00,
            4.57,
        ],
    }
)

DEFAULT_BENCHMARK = pd.DataFrame(
    {
        "Ticker": ["SPY", "AGG"],
        "Weight": [70.0, 30.0],
    }
)

DEFAULT_START_DATE = date(2019, 7, 1)
DEFAULT_END_DATE = date.today()
DEFAULT_INITIAL_VALUE = 1.0
DEFAULT_CASH_TICKER = "CASHX"
DEFAULT_CASH_PRICE = 1.0
DEFAULT_PORTFOLIO_REBALANCE = "none"
DEFAULT_BENCHMARK_REBALANCE = "daily"
DEFAULT_BENCHMARK_NAME = "70% S&P 500 / 30% US Agg (SPY/AGG proxy)"


# -------------------------
# Helpers
# -------------------------
def clean_weights_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    if "Ticker" not in out.columns or "Weight" not in out.columns:
        raise ValueError("Each editable table must have Ticker and Weight columns.")

    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
    out["Weight"] = pd.to_numeric(out["Weight"], errors="coerce")
    out = out.replace({"Ticker": {"": np.nan, "NAN": np.nan}})
    out = out.dropna(subset=["Ticker", "Weight"])
    out = out.loc[out["Ticker"] != ""]
    out = out.groupby("Ticker", as_index=False, sort=False)["Weight"].sum()

    if out.empty:
        raise ValueError("Please provide at least one ticker with a weight.")

    if np.isclose(out["Weight"].sum(), 0.0):
        raise ValueError("Weights sum to zero. Please enter non-zero weights.")

    return out


def normalize_weight_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    total = out["Weight"].sum()
    out["Normalized Weight"] = out["Weight"] / total
    out["Normalized Weight %"] = out["Normalized Weight"] * 100
    return out


def extract_close(downloaded: pd.DataFrame, expected_tickers: Iterable[str] | None = None) -> pd.DataFrame:
    if downloaded.empty:
        raise ValueError("Downloaded data is empty.")

    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Close" not in downloaded.columns.get_level_values(0):
            raise ValueError("Could not find 'Close' in downloaded data.")
        close = downloaded["Close"].copy()
    else:
        if "Close" not in downloaded.columns:
            raise ValueError("Could not find 'Close' in downloaded data.")
        close = downloaded[["Close"]].copy()
        if expected_tickers is None:
            expected_tickers = ["Asset"]
        close.columns = list(expected_tickers)[:1]

    if isinstance(close, pd.Series):
        close = close.to_frame()

    if expected_tickers is not None:
        close = close.reindex(columns=list(expected_tickers))

    close = close.sort_index().dropna(how="all")
    return close


def download_prices(tickers: list[str], start_date: date, end_date: date) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date + timedelta(days=1),
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    return extract_close(raw, expected_tickers=tickers)


def build_portfolio_prices_with_constant_cash(
    tickers: list[str],
    start_date: date,
    end_date: date,
    cash_ticker: str,
    cash_price: float,
) -> pd.DataFrame:
    tickers = [t.upper().strip() for t in tickers]
    cash_ticker = cash_ticker.upper().strip()
    non_cash_tickers = [t for t in tickers if t != cash_ticker]

    if non_cash_tickers:
        prices_non_cash = download_prices(non_cash_tickers, start_date, end_date)
    else:
        idx = pd.date_range(start=start_date, end=end_date, freq="B")
        if len(idx) == 0:
            idx = pd.DatetimeIndex([pd.Timestamp(start_date)])
        prices_non_cash = pd.DataFrame(index=idx)

    if prices_non_cash.empty and cash_ticker not in tickers:
        raise ValueError("No price data returned for the selected portfolio tickers.")

    if prices_non_cash.empty:
        idx = pd.date_range(start=start_date, end=end_date, freq="B")
        if len(idx) == 0:
            idx = pd.DatetimeIndex([pd.Timestamp(start_date)])
        prices_non_cash = pd.DataFrame(index=idx)

    prices = prices_non_cash.copy()
    if cash_ticker in tickers:
        prices[cash_ticker] = float(cash_price)

    prices = prices.reindex(columns=tickers).sort_index().dropna(how="all")
    return prices


def filter_investable_universe(prices: pd.DataFrame, weights_series: pd.Series) -> Dict[str, object]:
    if prices.empty:
        raise ValueError("Price matrix is empty after download.")

    first_backtest_date = prices.dropna(how="all").index.min()
    if pd.isna(first_backtest_date):
        raise ValueError("Could not determine a valid first backtest date.")

    start_snapshot = prices.loc[first_backtest_date]
    valid_tickers = start_snapshot[start_snapshot.notna()].index.tolist()
    removed_tickers = start_snapshot[start_snapshot.isna()].index.tolist()

    if not valid_tickers:
        raise ValueError("No assets had valid data on the first backtest date.")

    investable_prices = prices[valid_tickers].copy().loc[first_backtest_date:]
    investable_prices = investable_prices.ffill().dropna(how="all")

    investable_weights = weights_series.reindex(valid_tickers).fillna(0.0)
    removed_weights = weights_series.reindex(removed_tickers).fillna(0.0)
    removed_weight_total = float(removed_weights.sum())

    if np.isclose(investable_weights.sum(), 0.0):
        raise ValueError("All surviving weights sum to zero after filtering.")

    investable_weights = investable_weights / investable_weights.sum()
    investable_weights.name = "Reweighted"

    removed_summary = pd.DataFrame(
        {
            "Removed Ticker": removed_tickers,
            "Original Weight": [weights_series.get(t, 0.0) for t in removed_tickers],
        }
    )
    if not removed_summary.empty:
        removed_summary = removed_summary.set_index("Removed Ticker")

    return {
        "prices": investable_prices,
        "weights": investable_weights,
        "removed_summary": removed_summary,
        "removed_weight_total": removed_weight_total,
        "first_backtest_date": first_backtest_date,
        "valid_tickers": valid_tickers,
        "removed_tickers": removed_tickers,
    }


def backtest_portfolio(
    prices: pd.DataFrame,
    weights: np.ndarray,
    rebalance: str = "none",
    initial_value: float = 1.0,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    asset_returns = prices.pct_change().dropna()

    if rebalance == "daily":
        portfolio_returns = asset_returns.mul(weights, axis=1).sum(axis=1)
        portfolio_growth = initial_value * (1 + portfolio_returns).cumprod()
        portfolio_growth = pd.concat(
            [pd.Series([initial_value], index=[prices.index[0]]), portfolio_growth]
        ).sort_index()
        portfolio_returns = portfolio_growth.pct_change().dropna()
    elif rebalance == "none":
        normalized_prices = prices / prices.iloc[0]
        portfolio_growth = initial_value * normalized_prices.mul(weights, axis=1).sum(axis=1)
        portfolio_returns = portfolio_growth.pct_change().dropna()
    else:
        raise ValueError("rebalance must be either 'none' or 'daily'.")

    portfolio_growth.name = "Growth of $1"
    portfolio_returns.name = "Returns"
    return portfolio_growth, portfolio_returns, asset_returns


def compute_drawdown(growth_series: pd.Series) -> pd.Series:
    dd = growth_series / growth_series.cummax() - 1
    dd.name = "Drawdown"
    return dd


def annualized_return_from_growth(growth: pd.Series) -> float:
    n_days = (growth.index[-1] - growth.index[0]).days
    years = n_days / 365.25 if n_days > 0 else np.nan
    if pd.notna(years) and years > 0:
        return (growth.iloc[-1] / growth.iloc[0]) ** (1 / years) - 1
    return np.nan


def performance_stats(growth: pd.Series, returns: pd.Series) -> pd.Series:
    total_return = growth.iloc[-1] / growth.iloc[0] - 1
    cagr = annualized_return_from_growth(growth)
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / ann_vol if ann_vol and ann_vol > 0 else np.nan
    drawdown = compute_drawdown(growth)
    max_drawdown = drawdown.min()
    calmar = cagr / abs(max_drawdown) if pd.notna(cagr) and max_drawdown < 0 else np.nan

    return pd.Series(
        {
            "Final Value": growth.iloc[-1],
            "Total Return": total_return,
            "CAGR": cagr,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio (rf=0)": sharpe,
            "Max Drawdown": max_drawdown,
            "Calmar Ratio": calmar,
        }
    )


def relative_stats(
    portfolio_growth: pd.Series,
    benchmark_growth: pd.Series,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.Series:
    active_returns = portfolio_returns - benchmark_returns
    tracking_error = active_returns.std() * np.sqrt(252)
    information_ratio = (active_returns.mean() * 252) / tracking_error if tracking_error and tracking_error > 0 else np.nan
    correlation = portfolio_returns.corr(benchmark_returns)

    bench_var = benchmark_returns.var()
    beta = portfolio_returns.cov(benchmark_returns) / bench_var if bench_var and bench_var > 0 else np.nan

    port_cagr = annualized_return_from_growth(portfolio_growth)
    bench_cagr = annualized_return_from_growth(benchmark_growth)
    excess_cagr = port_cagr - bench_cagr if pd.notna(port_cagr) and pd.notna(bench_cagr) else np.nan

    return pd.Series(
        {
            "Excess CAGR": excess_cagr,
            "Tracking Error": tracking_error,
            "Information Ratio": information_ratio,
            "Correlation": correlation,
            "Beta vs Benchmark": beta,
        }
    )


def rolling_total_return(returns: pd.Series, window: int = 252) -> pd.Series:
    return (1 + returns).rolling(window).apply(np.prod, raw=True) - 1


def format_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pct_rows = {
        "Total Return",
        "CAGR",
        "Annualized Volatility",
        "Max Drawdown",
        "Excess CAGR",
        "Tracking Error",
    }

    for idx in out.index:
        if idx in pct_rows:
            out.loc[idx] = out.loc[idx].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
        elif idx == "Final Value":
            out.loc[idx] = out.loc[idx].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
        else:
            out.loc[idx] = out.loc[idx].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    return out


def make_line_chart(series_map: Dict[str, pd.Series], title: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, series in series_map.items():
        ax.plot(series.index, series.values, linewidth=2.2, label=label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend()
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.tight_layout()
    return fig


def make_drawdown_chart(series_map: Dict[str, pd.Series], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, series in series_map.items():
        ax.plot(series.index, series.values, linewidth=2.0, label=label)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0%}"))
    ax.grid(True, alpha=0.3)
    ax.legend()
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.tight_layout()
    return fig


def make_heatmap(corr: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(7, len(corr.columns) * 0.55), max(6, len(corr.index) * 0.45)))
    im = ax.imshow(corr.values, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title(title)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            text = "" if pd.isna(value) else f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def sheet_autofit(writer, sheet_name: str, df: pd.DataFrame) -> None:
    ws = writer.sheets[sheet_name]
    for idx, col in enumerate(df.columns, start=1):
        values = [str(col)] + [str(v) for v in df[col].tolist()]
        width = min(max(len(v) for v in values) + 2, 40)
        ws.column_dimensions[get_column_letter(idx)].width = width
    ws.freeze_panes = "B2"


def to_excel_bytes(dataframes: Dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in dataframes.items():
            safe_name = sheet_name[:31]
            export_df = df.copy()
            if isinstance(export_df.index, pd.DatetimeIndex):
                export_df = export_df.copy()
                export_df.index.name = export_df.index.name or "Date"
            export_df.to_excel(writer, sheet_name=safe_name)
            sheet_autofit(writer, safe_name, export_df.reset_index())
    output.seek(0)
    return output.read()


# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("Inputs")

    start_date = st.date_input("Start date", value=DEFAULT_START_DATE)
    end_date = st.date_input("End date", value=DEFAULT_END_DATE)

    initial_value = st.number_input("Initial value", min_value=0.01, value=float(DEFAULT_INITIAL_VALUE), step=0.1)
    portfolio_rebalance = st.selectbox(
        "Portfolio rebalance",
        options=["none", "daily"],
        index=["none", "daily"].index(DEFAULT_PORTFOLIO_REBALANCE),
    )
    benchmark_rebalance = st.selectbox(
        "Benchmark rebalance",
        options=["daily", "none"],
        index=["daily", "none"].index(DEFAULT_BENCHMARK_REBALANCE),
    )

    cash_ticker = st.text_input("Cash ticker label", value=DEFAULT_CASH_TICKER).upper().strip()
    cash_price = st.number_input("Cash constant price", min_value=0.0, value=float(DEFAULT_CASH_PRICE), step=0.01)
    benchmark_name = st.text_input("Benchmark display name", value=DEFAULT_BENCHMARK_NAME)

    st.subheader("Portfolio weights")
    portfolio_editor = st.data_editor(
        DEFAULT_PORTFOLIO,
        use_container_width=True,
        num_rows="dynamic",
        key="portfolio_editor",
    )

    st.subheader("Benchmark weights")
    benchmark_editor = st.data_editor(
        DEFAULT_BENCHMARK,
        use_container_width=True,
        num_rows="dynamic",
        key="benchmark_editor",
    )


# -------------------------
# Validation and run
# -------------------------
if end_date < start_date:
    st.error("End date must be on or after the start date.")
    st.stop()

if end_date == start_date:
    st.warning("Using the same start and end date usually will not produce a meaningful backtest. Pick a wider range if you want returns and correlations.")

try:
    portfolio_table = clean_weights_table(pd.DataFrame(portfolio_editor))
    benchmark_table = clean_weights_table(pd.DataFrame(benchmark_editor))
except Exception as exc:
    st.error(str(exc))
    st.stop()

portfolio_table = normalize_weight_table(portfolio_table)
benchmark_table = normalize_weight_table(benchmark_table)

weights_series = pd.Series(
    portfolio_table["Normalized Weight"].values,
    index=portfolio_table["Ticker"].tolist(),
    name="Original Weight",
)
benchmark_weight_series = pd.Series(
    benchmark_table["Normalized Weight"].values,
    index=benchmark_table["Ticker"].tolist(),
    name="Benchmark Weight",
)

with st.expander("Normalized inputs", expanded=False):
    left, right = st.columns(2)
    with left:
        st.write("Portfolio")
        st.dataframe(
            portfolio_table[["Ticker", "Weight", "Normalized Weight %"]].rename(columns={"Weight": "Input Weight", "Normalized Weight %": "Normalized Weight %"}),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.write("Benchmark")
        st.dataframe(
            benchmark_table[["Ticker", "Weight", "Normalized Weight %"]].rename(columns={"Weight": "Input Weight", "Normalized Weight %": "Normalized Weight %"}),
            use_container_width=True,
            hide_index=True,
        )

try:
    with st.spinner("Downloading prices and running backtest..."):
        prices_raw = build_portfolio_prices_with_constant_cash(
            tickers=weights_series.index.tolist(),
            start_date=start_date,
            end_date=end_date,
            cash_ticker=cash_ticker,
            cash_price=cash_price,
        )

        universe = filter_investable_universe(prices_raw, weights_series)
        prices = universe["prices"]
        investable_weights = universe["weights"]
        removed_summary = universe["removed_summary"]
        removed_weight_total = universe["removed_weight_total"]
        first_backtest_date = universe["first_backtest_date"]

        benchmark_prices_raw = download_prices(benchmark_weight_series.index.tolist(), start_date, end_date)
        benchmark_prices_raw = benchmark_prices_raw.loc[benchmark_prices_raw.index >= first_backtest_date].ffill().dropna(how="all")
        if benchmark_prices_raw.empty:
            raise ValueError("Benchmark price data is empty after cleaning.")

        benchmark_universe = filter_investable_universe(benchmark_prices_raw, benchmark_weight_series)
        benchmark_prices = benchmark_universe["prices"]
        benchmark_weight_series = benchmark_universe["weights"]

        portfolio_growth, portfolio_returns, asset_returns = backtest_portfolio(
            prices=prices,
            weights=investable_weights.values,
            rebalance=portfolio_rebalance,
            initial_value=initial_value,
        )

        benchmark_growth, benchmark_returns, benchmark_component_returns = backtest_portfolio(
            prices=benchmark_prices,
            weights=benchmark_weight_series.values,
            rebalance=benchmark_rebalance,
            initial_value=initial_value,
        )

        combined_growth = pd.concat(
            [portfolio_growth.rename("Portfolio"), benchmark_growth.rename(benchmark_name)],
            axis=1,
            join="inner",
        ).dropna()
        if combined_growth.empty:
            raise ValueError("No overlapping portfolio and benchmark history after alignment.")

        combined_returns = combined_growth.pct_change().dropna()
        portfolio_growth = combined_growth["Portfolio"]
        benchmark_growth = combined_growth[benchmark_name]
        portfolio_returns = combined_returns["Portfolio"]
        benchmark_returns = combined_returns[benchmark_name]

        portfolio_drawdown = compute_drawdown(portfolio_growth)
        benchmark_drawdown = compute_drawdown(benchmark_growth)

        portfolio_stats = performance_stats(portfolio_growth, portfolio_returns)
        benchmark_stats = performance_stats(benchmark_growth, benchmark_returns)
        active_stats = relative_stats(portfolio_growth, benchmark_growth, portfolio_returns, benchmark_returns)

        stats_df = pd.concat(
            [portfolio_stats.rename("Portfolio"), benchmark_stats.rename(benchmark_name)],
            axis=1,
        )
        active_stats_df = active_stats.to_frame(name="Portfolio vs Benchmark")

        benchmark_component_nav = (
            benchmark_prices.reindex(combined_growth.index).ffill() /
            benchmark_prices.reindex(combined_growth.index).ffill().iloc[0]
        ) * initial_value

        corr_price_frame = pd.concat(
            [prices.drop(columns=[cash_ticker], errors="ignore"), benchmark_prices],
            axis=1,
        )
        corr_price_frame = corr_price_frame.loc[:, ~corr_price_frame.columns.duplicated()].ffill().dropna(how="all")
        corr_returns = corr_price_frame.pct_change().dropna(how="all")
        corr_returns = corr_returns.loc[:, corr_returns.std() > 0]
        if corr_returns.empty:
            correlation_matrix = pd.DataFrame()
        else:
            correlation_matrix = corr_returns.corr().sort_index(axis=0).sort_index(axis=1)

        cumulative_relative = (portfolio_growth / benchmark_growth) - 1
        rolling_1y_port_return = rolling_total_return(portfolio_returns, window=252)
        rolling_1y_bench_return = rolling_total_return(benchmark_returns, window=252)
        rolling_63d_tracking_error = (portfolio_returns - benchmark_returns).rolling(63).std() * np.sqrt(252)

        weight_comparison = pd.concat(
            [weights_series.rename("Original Weight"), investable_weights.rename("Reweighted")],
            axis=1,
        ).fillna(0.0)

        excel_payload = {
            "portfolio_input_weights": portfolio_table.set_index("Ticker")[["Weight", "Normalized Weight", "Normalized Weight %"]],
            "benchmark_input_weights": benchmark_table.set_index("Ticker")[["Weight", "Normalized Weight", "Normalized Weight %"]],
            "price_history": prices_raw,
            "investable_prices": prices,
            "benchmark_prices": benchmark_prices,
            "benchmark_weight_comparison": pd.concat([benchmark_table.set_index("Ticker")["Normalized Weight"].rename("Original Weight"), benchmark_weight_series.rename("Reweighted")], axis=1).fillna(0.0),
            "growth_of_dollar": combined_growth,
            "portfolio_returns": portfolio_returns.to_frame("Portfolio"),
            "benchmark_returns": benchmark_returns.to_frame(benchmark_name),
            "portfolio_drawdown": portfolio_drawdown.to_frame("Portfolio"),
            "benchmark_drawdown": benchmark_drawdown.to_frame(benchmark_name),
            "absolute_stats": stats_df,
            "relative_stats": active_stats_df,
            "weight_comparison": weight_comparison,
            "benchmark_component_nav": benchmark_component_nav,
            "cumulative_relative": cumulative_relative.to_frame("Cumulative Excess Return"),
            "rolling_1y_returns": pd.concat(
                [rolling_1y_port_return.rename("Portfolio"), rolling_1y_bench_return.rename(benchmark_name)],
                axis=1,
            ),
            "rolling_tracking_error": rolling_63d_tracking_error.to_frame("63D Tracking Error"),
        }
        if not correlation_matrix.empty:
            excel_payload["correlation_matrix"] = correlation_matrix
        if isinstance(removed_summary, pd.DataFrame) and not removed_summary.empty:
            excel_payload["removed_assets"] = removed_summary

        excel_bytes = to_excel_bytes(excel_payload)

except Exception as exc:
    st.error(f"Backtest failed: {exc}")
    st.stop()


# -------------------------
# Summary metrics
# -------------------------
st.subheader("Summary")
metric_cols = st.columns(5)
metric_cols[0].metric("Portfolio final value", f"${portfolio_growth.iloc[-1]:,.2f}")
metric_cols[1].metric("Benchmark final value", f"${benchmark_growth.iloc[-1]:,.2f}")
metric_cols[2].metric("Portfolio CAGR", f"{portfolio_stats['CAGR']:.2%}" if pd.notna(portfolio_stats["CAGR"]) else "")
metric_cols[3].metric("Benchmark CAGR", f"{benchmark_stats['CAGR']:.2%}" if pd.notna(benchmark_stats["CAGR"]) else "")
metric_cols[4].metric("Excess CAGR", f"{active_stats['Excess CAGR']:.2%}" if pd.notna(active_stats["Excess CAGR"]) else "")

st.caption(
    f"Requested start date: {start_date} | Effective first backtest date: {first_backtest_date.date()} | "
    f"Portfolio rebalance: {portfolio_rebalance} | Benchmark rebalance: {benchmark_rebalance}"
)

if isinstance(removed_summary, pd.DataFrame) and not removed_summary.empty:
    st.info(
        f"Assets without data on the first backtest date were removed and {removed_weight_total:.2%} of weight was redistributed across investable assets."
    )


# -------------------------
# Benchmark first
# -------------------------
st.subheader("Benchmark growth of $1")
benchmark_fig = make_line_chart(
    {col: benchmark_component_nav[col] for col in benchmark_component_nav.columns},
    title="Benchmark components — growth of $1",
    ylabel="Value",
)
st.pyplot(benchmark_fig, use_container_width=True)
plt.close(benchmark_fig)


# -------------------------
# Main comparison charts
# -------------------------
left, right = st.columns(2)
with left:
    growth_fig = make_line_chart(
        {"Portfolio": portfolio_growth, benchmark_name: benchmark_growth},
        title="Portfolio vs benchmark — growth of $1",
        ylabel="Value",
    )
    st.pyplot(growth_fig, use_container_width=True)
    plt.close(growth_fig)

with right:
    dd_fig = make_drawdown_chart(
        {"Portfolio": portfolio_drawdown, benchmark_name: benchmark_drawdown},
        title="Drawdown comparison",
    )
    st.pyplot(dd_fig, use_container_width=True)
    plt.close(dd_fig)


# -------------------------
# Stats tables
# -------------------------
st.subheader("Performance tables")
st.dataframe(format_stats_table(stats_df), use_container_width=True)
st.dataframe(format_stats_table(active_stats_df), use_container_width=True)


# -------------------------
# Correlation matrix
# -------------------------
st.subheader("Correlation matrix of index funds")
if correlation_matrix.empty:
    st.warning("Not enough non-constant return history to compute a correlation matrix.")
else:
    corr_fig = make_heatmap(correlation_matrix, "Correlation matrix of portfolio and benchmark funds")
    st.pyplot(corr_fig, use_container_width=True)
    plt.close(corr_fig)
    st.dataframe(correlation_matrix.style.format("{:.2f}"), use_container_width=True)


# -------------------------
# Detail tables
# -------------------------
with st.expander("Detailed analysis tables", expanded=False):
    tabs = st.tabs(
        [
            "Growth of $1",
            "Weight comparison",
            "Removed assets",
            "Prices",
            "Returns",
        ]
    )

    with tabs[0]:
        st.dataframe(combined_growth, use_container_width=True)
    with tabs[1]:
        st.dataframe(weight_comparison, use_container_width=True)
    with tabs[2]:
        if isinstance(removed_summary, pd.DataFrame) and not removed_summary.empty:
            st.dataframe(removed_summary, use_container_width=True)
        else:
            st.write("No assets were removed.")
    with tabs[3]:
        st.dataframe(prices, use_container_width=True)
    with tabs[4]:
        st.dataframe(
            pd.concat(
                [portfolio_returns.rename("Portfolio"), benchmark_returns.rename(benchmark_name)],
                axis=1,
            ),
            use_container_width=True,
        )


# -------------------------
# Export
# -------------------------
st.subheader("Download")
st.download_button(
    label="Download Excel analysis",
    data=excel_bytes,
    file_name=f"top_down_fun_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.caption("The Excel export includes inputs, cleaned prices, benchmark data, growth of $1, drawdowns, stats, rolling analytics, and the correlation matrix.")
