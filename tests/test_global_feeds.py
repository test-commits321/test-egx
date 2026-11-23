"""Offline regressions for the requested index feeds and old history files."""
import io
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
import predictor
from predictor import ADDED_MARKET_KEYS, CAIRO, MARKETS, local_datetime


class GlobalFeedTests(unittest.TestCase):
    def setUp(self):
        self.markets = {market.key: market for market in MARKETS}
        self.dates = pd.to_datetime(['2026-11-19', '2026-11-20', '2026-11-23'])
        self.prices = pd.Series([100., 102., 101.], index=self.dates)

    def test_actual_index_tickers_and_local_close_buffer(self):
        config = {
            'NYSE': ('^GSPC', 'America/New_York', '16:30'),
            'DOW': ('^DJI', 'America/New_York', '16:30'),
            'FTSE': ('^FTSE', 'Europe/London', '17:05'),
            'IBEX': ('^IBEX', 'Europe/Madrid', '18:05'),
        }
        for key, (ticker, zone, ready_time) in config.items():
            with self.subTest(ticker=ticker), patch.object(app, 'yahoo_close', return_value=self.prices) as fetch:
                market = self.markets[key]
                ready = local_datetime(self.dates[-1], ready_time, zone).tz_convert(CAIRO)
                early = app.fetch_market(market, ready - timedelta(minutes=1), None)
                later = app.fetch_market(market, ready, None)
                self.assertEqual(early.index[-1], self.dates[-2])
                self.assertEqual(later.index[-1], self.dates[-1])
                self.assertEqual([call.args[0] for call in fetch.call_args_list], [ticker, ticker])
        self.assertEqual(sum(market.ticker == '^GSPC' for market in MARKETS), 1)

    def test_yahoo_rejects_wrong_symbol_and_uses_local_dates_on_fallback(self):
        timestamps = [int(pd.Timestamp(day).timestamp()) for day in
                      ['2026-11-19T01:00:00Z', '2026-11-20T01:00:00Z', '2026-11-21T01:00:00Z']]

        def response(symbol):
            result = {'meta': {'symbol': symbol, 'dataGranularity': '1d',
                               'exchangeTimezoneName': 'America/New_York'},
                      'timestamp': timestamps, 'indicators': {'quote': [{'close': [95., 100., 110.]}]}}
            return io.BytesIO(json.dumps({'chart': {'result': [result], 'error': None}}).encode())

        first, last = datetime(2026, 11, 19, tzinfo=CAIRO), datetime(2026, 11, 20, tzinfo=CAIRO)
        with patch.object(predictor, 'urlopen', side_effect=[response('OTHER'), response('^DJI')]) as fetch:
            prices = predictor.yahoo_close('^DJI', first, last)
        self.assertEqual(fetch.call_count, 2)
        self.assertIn('query2.finance.yahoo.com/v8/finance/chart/%5EDJI', fetch.call_args.args[0].full_url)
        self.assertEqual(list(prices.index), [pd.Timestamp('2026-11-19')])
        self.assertEqual(list(prices), [100.])

    def test_sp500_csv_alias_precedes_legacy_filename(self):
        asof = datetime(2026, 11, 23, 9, 45, tzinfo=CAIRO)
        market = self.markets['NYSE']
        self.assertEqual(market.exchange, 'S&P 500')
        with tempfile.TemporaryDirectory() as folder, patch.object(app, 'yahoo_close') as fetch:
            root = Path(folder)
            (root / 'NYSE.csv').write_text('Date,Close\n2026-11-19,100\n2026-11-20,102\n')
            self.assertEqual(app.fetch_market(market, asof, root).iloc[-1], 102.)
            (root / 'SP500.csv').write_text('Date,Close\n2026-11-19,100\n2026-11-20,103\n')
            self.assertEqual(app.fetch_market(market, asof, root).iloc[-1], 103.)
            fetch.assert_not_called()

    def test_fresher_cache_wins_over_stale_success(self):
        asof = datetime(2026, 11, 23, 9, 45, tzinfo=CAIRO)
        market = self.markets['FTSE']
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            previous = self.prices.iloc[:2]
            app.save_market_cache(market, previous, asof, cache, 'test')
            older = previous.set_axis(pd.to_datetime(['2026-11-18', '2026-11-19']))
            with patch.object(app, 'yahoo_close', return_value=older):
                actual = app.fetch_market(market, asof, None, cache_dir=cache)
            pd.testing.assert_series_equal(actual, previous, check_names=False)

    def test_legacy_history_has_an_explicit_upgrade_and_reuse_path(self):
        days = pd.bdate_range(end='2026-11-20', periods=600)
        rows = [{'date': str(day.date()), 'close': 1000 + i} for i, day in enumerate(days)]
        payload = {
            'schema_version': app.SCHEMA_VERSION,
            'egx30': {'data': rows},
            'markets': {m.key: {'ticker': m.ticker, 'data': rows}
                        for m in MARKETS if m.key not in ADDED_MARKET_KEYS},
            'excluded_markets': [],
        }
        with self.assertRaisesRegex(ValueError, 'Set --history to a new file'):
            app.validate_history(payload)
        _, reuse = app.validate_history(payload, allow_missing_markets=True)
        self.assertEqual(set(reuse), set(self.markets).difference(ADDED_MARKET_KEYS))
        self.assertEqual(reuse['NYSE'].iloc[-1], 1599.)

    def test_preclose_history_bar_rejected_for_training(self):
        days = pd.bdate_range(end='2026-11-20', periods=600)
        rows = [{'date': str(day.date()), 'close': 1000 + i} for i, day in enumerate(days)]
        payload = {
            'schema_version': app.SCHEMA_VERSION,
            'egx30': {'data': rows},
            'markets': {market.key: {'ticker': market.ticker, 'data': rows} for market in MARKETS},
            'excluded_markets': [],
        }
        payload['markets']['DOW']['data'] = rows[:-1] + [{'date': '2026-11-23', 'close': 1600.}]
        asof = datetime(2026, 11, 23, 10, 0, tzinfo=CAIRO)
        with self.assertRaisesRegex(ValueError, 'DOW: history includes a provisional'):
            app.validate_history(payload, asof=asof)


if __name__ == '__main__':
    unittest.main()
