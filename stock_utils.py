from yfinance import Ticker
from typing import Optional


def calculate_mean_pe_ratio(ticker: str, period: str = "1y") -> float:
    yticker = Ticker(ticker)

    # Load the stock's closing price data for the past year
    data = yticker.history(period=period)

    # Get the income statement and shares outstanding data
    income_statement = yticker.income_stmt
    shares_outstanding = yticker.get_shares_full()

    # Extract net income and calculate EPS (Earnings Per Share)
    # Ensure to get the latest net income; if it's annual, use the latest year
    net_income = income_statement.loc["Net Income"].iloc[0]  # Latest Net Income
    eps = net_income / shares_outstanding.iloc[0]  # EPS

    # Calculate the daily P/E ratio by dividing each close price by the EPS
    pe_ratios = data["Close"] / eps

    # Calculate the average P/E ratio over
    return pe_ratios.mean()


def calculate_intrinsic_value(ticker: str, expected_growth: float) -> Optional[float]:
    """
    Calculates the intrinsic value of a stock using the P/E (Price to Earnings) indicator.
    """

    def get_future_eps(eps: float, growth_rate: float, years: int) -> float:
        return eps * (1 + growth_rate) ** years

    def get_present_day_stock_price(
        future_stock_price: float, growth_rate: float, years: int
    ) -> float:
        return future_stock_price / (1 + growth_rate) ** years

    yticker = Ticker(ticker)

    try:
        growth_estimate_5_years = yticker.growth_estimates.loc["+5y", "stock"]
    except KeyError:
        return

    eps = yticker.info["trailingEps"]

    future_eps = get_future_eps(eps, growth_estimate_5_years, 5)

    average_pe_ratio = calculate_mean_pe_ratio(ticker)

    future_stock_price = future_eps * average_pe_ratio

    return get_present_day_stock_price(future_stock_price, expected_growth, 5)


def parse_stock_tickers(unparsed_tickers: str) -> list[str]:
    return [x.strip().upper() for x in unparsed_tickers.split(",")]
