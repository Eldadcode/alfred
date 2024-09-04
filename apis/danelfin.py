from typing import Dict, Optional
import argparse
import requests
from requests import HTTPError
import json
from dataclasses import dataclass
from contextlib import suppress


@dataclass
class DanelfinScores:
    general: int
    sentiment: int
    technical: int
    fundamental: int


class DanelfinAPI:
    API_URL = "https://api2.danelfin.com"

    def __init__(self, key: str):
        self.key = key

    def _make_api_request(
        self, endpoint: str, params: Dict[str, str]
    ) -> requests.Response:
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://danelfin.com",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        }

        params["X-API-KEY"] = self.key

        return requests.get(
            f"{self.API_URL}/{endpoint}", params=params, headers=headers
        )

    def get_ticker_score_graphs(self, ticker: str) -> Dict[str, str]:
        params = {"ticker": ticker, "market": "usa"}

        api_response = self._make_api_request("tickerscoregraphs", params)

        if api_response.status_code != 200:
            raise HTTPError(
                f"Error: Got status code {api_response.status_code} from the request"
            )

        return json.loads(api_response.json())

    @staticmethod
    def _get_score_from_graph(graph: Dict[str, str]) -> Optional[int]:
        with suppress(ValueError):
            return int(graph.split(";")[-1].split(",")[-1].rstrip("]"))

    def get_ticker_ai_scores(self, ticker: str) -> DanelfinScores:

        score_from_api = self.get_ticker_score_graphs(ticker)

        scores = DanelfinScores(0, 0, 0, 0)
        for score in ("general", "fundamental", "technical", "sentiment"):

            graph = score_from_api[f"ss_graph_{score}"]
            setattr(scores, score, self._get_score_from_graph(graph))
        return scores


def print_results(ticker: str, results: Dict[str, int]):
    print(f"\nDanelfin AI Score for {ticker.upper()}:")
    for result, score in results.items():
        print(f"{result.title()}: {score}")


def main():
    parser = argparse.ArgumentParser("Danelfin")

    parser.add_argument("api", type=str, help="API Key")

    parser.add_argument("tickers", type=str, nargs="+", help="Ticker of a stock")

    args = parser.parse_args()

    danelfin = DanelfinAPI(args.api)

    for ticker in args.tickers:
        results = danelfin.get_ticker_ai_score(ticker)
        print_results(ticker, results)


if __name__ == "__main__":
    main()
