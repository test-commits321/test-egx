"""Offline regression tests for the transient successful-check / failed-init case.

All prices are synthetic fixtures, never included in the distributed history.
"""
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from predictor import MARKETS, CAIRO


class CacheResilienceTests(unittest.TestCase):
    def setUp(self):
        self.asof = datetime(2026, 11, 23, 16, 0, tzinfo=CAIRO)
        self.saudi = next(m for m in MARKETS if m.key == 'SAUDI')
        # 725 business-day synthetic closes, ending before the tested date.
        d = pd.bdate_range(end='2026-11-20', periods=725)
        self.prices = pd.Series([10000 + i * .6 for i in range(len(d))], index=d, dtype=float)

    def test_check_then_init_uses_cache_without_second_tradingview_call(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {'SAUDI_SOURCE': 'tradingview'}):
            cache = Path(folder)
            with patch.object(app, 'tradingview_tasi_close', return_value=self.prices) as fetch:
                fetched = app.fetch_market(self.saudi, self.asof, None, min_sessions=600,
                                           cache_dir=cache, prefer_cache=False)
                self.assertEqual(len(fetched), 725)
                fetch.assert_called_once()
            self.assertTrue((cache / 'SAUDI.json').is_file())
            with patch.object(app, 'tradingview_tasi_close', side_effect=ConnectionError('offline')) as fail:
                reused = app.fetch_market(self.saudi, self.asof, None, min_sessions=600,
                                          cache_dir=cache, prefer_cache=True)
                fail.assert_not_called()
                self.assertEqual(len(reused), 725)
                self.assertEqual(reused.iloc[-1], fetched.iloc[-1])

    def test_connection_failure_retries_then_saves_real_prices(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {'SAUDI_SOURCE': 'tradingview'}), \
                patch.object(app.clock, 'sleep') as sleeper, \
                patch.object(app, 'tradingview_tasi_close', side_effect=[
                    ConnectionError('WebSocket lost'), ValueError('no data'), self.prices]) as source:
            data = app.fetch_market(self.saudi, self.asof, None, min_sessions=600,
                                    cache_dir=Path(folder))
            self.assertEqual(len(data), 725)
            self.assertEqual(source.call_count, 3)
            self.assertEqual(sleeper.call_count, 2)
            self.assertTrue((Path(folder) / 'SAUDI.json').is_file())

    def test_remote_failure_uses_cached_price_and_warns(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {'SAUDI_SOURCE': 'tradingview'}):
            cache = Path(folder)
            app.save_market_cache(self.saudi, self.prices, self.asof, cache, 'tradingview')
            with patch.object(app.clock, 'sleep'), \
                    patch.object(app, 'tradingview_tasi_close', side_effect=ConnectionError('disconnected')):
                out = app.fetch_market(self.saudi, self.asof, None, min_sessions=600,
                                       cache_dir=cache, prefer_cache=False)
            self.assertEqual(len(out), len(self.prices))

    def test_outdated_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {'SAUDI_SOURCE': 'tradingview'}):
            cache = Path(folder)
            old = self.prices.loc[:'2026-11-01']
            app.save_market_cache(self.saudi, old, self.asof, cache, 'tradingview')
            with patch.object(app.clock, 'sleep'), \
                    patch.object(app, 'tradingview_tasi_close', side_effect=ConnectionError('offline')):
                with self.assertRaisesRegex(RuntimeError, 'no fresh local cache'):
                    app.fetch_market(self.saudi, self.asof, None, min_sessions=600,
                                     cache_dir=cache)

    def test_partial_daily_update_merges_with_full_history(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            app.save_market_cache(self.saudi, self.prices, self.asof, cache, 'tradingview')
            recent = self.prices.tail(130).copy()
            recent.iloc[-1] += 1
            app.save_market_cache(self.saudi, recent, self.asof, cache, 'tradingview_live')
            merged = app.read_market_cache(self.saudi, self.asof, cache, min_sessions=600)
            self.assertEqual(len(merged), 725)
            self.assertEqual(merged.iloc[-1], self.prices.iloc[-1] + 1)

    def test_full_build_history_reuses_saudi_cache_after_connection_loss(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {'SAUDI_SOURCE': 'tradingview'}):
            path = Path(folder)
            cache = path / 'cache'
            app.save_market_cache(self.saudi, self.prices, self.asof, cache, 'tradingview')
            egypt_days = pd.date_range('2024-01-01', '2026-11-22')
            egypt_days = egypt_days[~egypt_days.dayofweek.isin([4, 5])][-620:]
            egy = pd.Series([30000 + i * .4 for i in range(len(egypt_days))],
                            index=egypt_days, dtype=float)
            with patch.object(app, 'DEFAULT_MARKET_CACHE', cache), \
                    patch.object(app, 'fetch_egx', return_value=egy), \
                    patch.object(app, 'yahoo_close', return_value=self.prices) as yahoo, \
                    patch.object(app, 'tradingview_tasi_close',
                                 side_effect=ConnectionError('WebSocket disconnected')) as tv:
                payload = app.build_history(path / 'history_600.json', self.asof,
                                            csv_dir=None, egx_csv=None)
            self.assertEqual(set(payload['markets']), {market.key for market in MARKETS})
            self.assertEqual(len(payload['markets']['SAUDI']['data']), 600)
            self.assertTrue((path / 'history_600.json').exists())
            self.assertEqual(yahoo.call_count, len(MARKETS) - 1)
            tv.assert_not_called()
            self.assertEqual(len(list(cache.glob('*.json'))), len(MARKETS))

    def test_wrong_market_or_future_bar_cache_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            app.save_market_cache(self.saudi, self.prices, self.asof, cache, 'tradingview')
            file = cache / 'SAUDI.json'
            doc = app.read_json(file)
            doc['ticker'] = '^OTHER'
            app.dump_atomic(file, doc)
            self.assertIsNone(app.read_market_cache(self.saudi, self.asof, cache, min_sessions=600))
            doc['ticker'] = self.saudi.ticker
            doc['data'].append({'date': '2026-11-24', 'close': 12000.0})
            app.dump_atomic(file, doc)
            self.assertIsNone(app.read_market_cache(self.saudi, self.asof, cache, min_sessions=600))


if __name__ == '__main__':
    unittest.main()
