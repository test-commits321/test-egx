"""Offline TradingView/CSV compatibility checks: all samples are synthetic."""
from __future__ import annotations

import os
import logging
import sys
import tempfile
import types
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from predictor import CAIRO, MARKETS, parse_close_csv


class TradingViewSaudiTests(unittest.TestCase):
    def setUp(self):
        self.saudi = next(m for m in MARKETS if m.key == 'SAUDI')
        self.asof = datetime(2026, 11, 23, 16, 0, tzinfo=CAIRO)

    def module_for(self, bars):
        class FakeReader:
            def get_hist(self, symbol, exchange, interval, n_bars):
                self_calls.append((symbol, exchange, interval, n_bars))
                return bars
        self_calls = []
        module = types.ModuleType('tvDatafeed')
        module.TvDatafeed = FakeReader
        module.Interval = types.SimpleNamespace(in_daily='D')
        return module, self_calls

    def test_host_local_to_riyadh_date_and_close(self):
        times = pd.to_datetime(['2026-11-22T21:00:00Z', '2026-11-23T21:00:00Z'])
        host_local = [t.to_pydatetime().astimezone().replace(tzinfo=None) for t in times]
        df = pd.DataFrame({'close': [11401., 11402.]},
                          index=pd.DatetimeIndex(host_local))
        module, calls = self.module_for(df)
        with patch.dict(sys.modules, {'tvDatafeed': module}):
            bars = app.tradingview_tasi_close(650)
        self.assertEqual(calls[0], ('TASI', 'TADAWUL', 'D', 650))
        self.assertEqual(list(bars.index), [pd.Timestamp('2026-11-23'), pd.Timestamp('2026-11-24')])
        self.assertEqual(bars.iloc[0], 11401.)

    def test_disconnect_keeps_the_useful_tradingview_diagnostic(self):
        module, _ = self.module_for(None)

        class DisconnectingReader:
            def get_hist(self, **kwargs):
                logger = logging.getLogger('tvDatafeed.main')
                logger.error('Remote host lost')
                logger.error('No data, check the exchange and symbol')
                return None

        module.TvDatafeed = DisconnectingReader
        with patch.dict(sys.modules, {'tvDatafeed': module}):
            with self.assertRaisesRegex(RuntimeError, r'^TradingView TADAWUL:TASI: Remote host lost$'):
                app.tradingview_tasi_close()

    def test_scheduled_saudi_refresh_reuses_cache_of_last_completed_session(self):
        preopen = datetime(2026, 11, 23, 9, 55, tzinfo=CAIRO)
        synthetic = pd.Series([98., 100.], index=pd.to_datetime(['2026-11-19', '2026-11-22']))
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            app.save_market_cache(self.saudi, synthetic, preopen, cache, 'test')
            with patch.object(app, 'tradingview_tasi_close') as feed:
                result = app.fetch_market(self.saudi, preopen, None, cache_dir=cache, reuse_current_saudi=True)
            feed.assert_not_called()
            self.assertEqual(result.iloc[-1], 100.)

    def test_auto_uses_tradingview_first_and_ignores_provisional_day(self):
        dates = pd.bdate_range('2026-11-15', '2026-11-24')
        # Controlled source includes all dates; Tue Nov 24 must not be counted on Mon Nov 23.
        dates = pd.to_datetime(['2026-11-18', '2026-11-19', '2026-11-22', '2026-11-23', '2026-11-24'])
        synthetic = pd.Series(range(101, 106), index=dates, dtype=float)
        with patch.dict(os.environ, {'SAUDI_SOURCE': 'auto'}), \
             patch.object(app, 'tradingview_tasi_close', return_value=synthetic) as tv, \
             patch.object(app, 'yahoo_close', side_effect=AssertionError('Yahoo should not be used')):
            result = app.fetch_market(self.saudi, self.asof, None, min_sessions=2)
        tv.assert_called_once()
        self.assertNotIn(pd.Timestamp('2026-11-24'), result.index)
        self.assertEqual(len(result), 4)

    def test_short_tradingview_history_falls_back_yahoo(self):
        tiny = pd.Series([100., 101.], index=pd.to_datetime(['2026-11-18', '2026-11-19']))
        full = pd.Series(range(1000, 1600), index=pd.bdate_range(end='2026-11-20', periods=600), dtype=float)
        with patch.dict(os.environ, {'SAUDI_SOURCE': 'auto', 'TASI_API_KEY': ''}), \
             patch.object(app, 'tradingview_tasi_close', return_value=tiny), \
             patch.object(app, 'yahoo_close', return_value=full) as yf:
            result = app.fetch_market(self.saudi, self.asof, None, min_sessions=600)
        self.assertEqual(len(result), 600)
        yf.assert_called_once()

    def test_tradingview_export_unix_seconds_to_correct_saudi_dates(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'SAUDI.csv'
            path.write_text('time,open,high,low,close,Volume\n'
                            '1795381200,1,2,1,11120.5,99\n'
                            '1795467600,1,2,1,11140.5,99\n', encoding='utf-8')
            parsed = parse_close_csv(path, timezone='Asia/Riyadh')
        self.assertEqual(len(parsed), 2)
        self.assertTrue(all(p.year == 2026 for p in parsed.index))
        self.assertEqual(list(parsed.values), [11120.5, 11140.5])

    def test_tradingview_export_iso_utc_to_riyadh_dates(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'SAUDI.csv'
            path.write_text('time,close\n2026-11-22T21:00:00Z,11401\n2026-11-23T21:00:00Z,11402\n')
            parsed = parse_close_csv(path, timezone='Asia/Riyadh')
        self.assertEqual(list(parsed.index), [pd.Timestamp('2026-11-23'), pd.Timestamp('2026-11-24')])

    def test_auto_reports_source_failures_without_made_up_data(self):
        with patch.dict(os.environ, {'SAUDI_SOURCE': 'auto', 'TASI_API_KEY': ''}), \
             patch.object(app, 'tradingview_tasi_close', side_effect=ValueError('blocked')), \
             patch.object(app.clock, 'sleep'), \
             patch.object(app, 'yahoo_close', side_effect=ValueError('empty')):
            with self.assertRaisesRegex(RuntimeError, 'tradingview attempt 1/3: ValueError: blocked'):
                app.fetch_market(self.saudi, self.asof, None, min_sessions=600)


if __name__ == '__main__':
    unittest.main()
