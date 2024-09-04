from tipranks import TipRanks
from dataclasses import dataclass
from typing import Dict, Optional
from contextlib import suppress


@dataclass
class TipRanksScores:
    _raw_data: Dict

    @property
    def _data(self):
        return self._raw_data["data"][0]

    @property
    def _extra_data(self):
        return self._raw_data["extraData"][0]

    @property
    def company_name(self) -> str:
        return self._data["companyFullName"]

    @property
    def consensus(self) -> Optional[str]:
        with suppress(AttributeError):
            return self._data["analystConsensus"]["consensus"].upper()

    @property
    def best_price_target(self) -> Optional[float]:
        with suppress(TypeError):
            return float(self._data["bestPriceTarget"])

    @property
    def price_target(self) -> Optional[float]:
        with suppress(TypeError):
            return float(self._data["priceTarget"])

    @property
    def price(self) -> Optional[float]:
        with suppress(TypeError):
            return float(self._extra_data["research"]["price"])

    @property
    def pe_ratio(self) -> Optional[float]:
        with suppress(TypeError):
            return float(self._data["peRatio"])

    def _get_gain(self, months: int) -> str:
        translation = {
            1: "oneMonthGain",
            3: "threeMonthsGain",
            6: "sixMonthsGain",
            12: "ytdGain",
        }
        return f'{self._extra_data["research"][translation[months]] * 100:.2f}%'

    @property
    def one_month_gain(self):
        return self._get_gain(1)

    @property
    def three_months_gain(self):
        return self._get_gain(3)

    @property
    def six_months_gain(self):
        return self._get_gain(6)

    @property
    def ytd_gain(self):
        return self._get_gain(12)


class MyTipRanks(TipRanks):
    def __init__(self, email, password):
        super().__init__(email, password)

    def get_analyst_projection(self, ticker: str) -> TipRanksScores:
        return TipRanksScores(
            self._TipRanks__request(
                method="GET", endpoint="/api/assets", params={"tickers": ticker.lower()}
            )
        )
