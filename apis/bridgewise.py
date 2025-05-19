import requests
from typing import Dict
import argparse
from requests import HTTPError


class BridgewiseAPI:
    API_URL = "https://api.bridgewise.com/v2"

    def __init__(self, key: str):
        raise DeprecationWarning("This API is deprecated!")
        self.key = key

    def _make_api_request(
        self, endpoint: str, params: Dict[str, str]
    ) -> requests.Response:

        headers = {
            "authority": "api.bridgewise.com",
            "accept": "application/json, text/plain, */*",
            "api-key": self.key,
            "origin": "https://bridgewise.com",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        }

        return requests.get(
            f"{self.API_URL}/{endpoint}", headers=headers, params=params
        )

    def get_companies_from_scanner(
        self,
        amount: int = 20000,
        last_n_days: int = 365,
        score: bool = True,
        report: bool = True,
    ) -> Dict[str, str]:

        params = {
            "n": amount,
            "by": "market_cap",
            "last_n_days": last_n_days,
            "score": str(score).lower(),
            "report": str(report).lower(),
        }

        api_response = self._make_api_request("scanner", params)

        if api_response.status_code != 200:
            raise HTTPError(
                f"Error: Got status code {api_response.status_code} from the request"
            )

        return api_response.json()

    @staticmethod
    def get_ticker_ai_score(scanned_companies, ticker: str) -> int:

        sorted_metadata = sort_by_company(scanned_companies["data"]["metadata"])
        sorted_score = sort_by_company(scanned_companies["data"]["score"])

        for index, company in enumerate(sorted_metadata):
            if ticker.upper() == company["ticker_symbol"]:
                return sorted_score[index]["final_assessment"]


def sort_by_company(data):
    return sorted(data, key=lambda x: x["company_id"])


def print_results(ticker: str, score: int):
    print(f"\nBridgewise AI Score for {ticker.upper()}: {score}")


def main():
    parser = argparse.ArgumentParser("Bridgewise")

    parser.add_argument("api", type=str, help="API Key")

    parser.add_argument("tickers", type=str, nargs="+", help="Ticker of a stock")

    args = parser.parse_args()

    bridgewise = BridgewiseAPI(args.api)

    for ticker in args.tickers:
        score = bridgewise.get_ticker_ai_score(ticker)
        print_results(ticker, score)


if __name__ == "__main__":
    main()
