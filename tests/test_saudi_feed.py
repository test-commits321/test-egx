"""TASI provider and optional market exclusions; all mock bars are synthetic."""
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from predictor import MARKETS, CAIRO


class SaudiFeedTests(unittest.TestCase):
    def setUp(self):
        self.asof = datetime(2026, 11, 23, 16, 0, tzinfo=CAIRO)
        self.saudi = next(m for m in MARKETS if m.key == 'SAUDI')

    def test_keyed_tasi_api_paginates_and_only_keeps_final_bars(self):
        calls = []
        def fake_get(url, headers, params, timeout):
            self.assertIn('/history/TASI', url)
            self.assertEqual(headers['X-API-Key'], 'TEST-KEY')
            self.assertLessEqual((pd.Timestamp(params['to']) - pd.Timestamp(params['from'])).days, 350)
            calls.append(params)
            # End-of-day timestamps in Saudi zone, not UTC's local day.
            bars = [{'t': '2026-11-17T23:00:00Z', 'close': 11000.0, 'quality': 'final'},
                    {'t': '2026-11-18T12:00:00+03:00', 'close': 11090.0, 'quality': 'final'},
                    {'t': '2026-11-19T12:00:00+03:00', 'close': 11180.0, 'quality': 'provisional'}]
            r = Mock()
            r.json.return_value = {'data': bars}
            return r
        with patch.dict(os.environ, {'TASI_API_KEY': 'TEST-KEY'}), patch('requests.get', side_effect=fake_get):
            out = app.tasi_api_close(self.asof.replace(year=2024), self.asof)
        self.assertGreaterEqual(len(calls), 3)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.index[0], pd.Timestamp('2026-11-18'))
        # Equal-date duplicated final bars normalize without fabricated records.
        self.assertEqual(out.loc[pd.Timestamp('2026-11-18')], 11090.0)
        self.assertNotIn(pd.Timestamp('2026-11-19'), out.index)

    def test_yahoo_zero_falls_back_to_tasi_api(self):
        real = pd.Series([100., 101., 102.], index=pd.to_datetime(['2026-11-17', '2026-11-18', '2026-11-19']))
        with patch.dict(os.environ, {'TASI_API_KEY': 'TEST-KEY', 'SAUDI_SOURCE':'auto'}), \
             patch.object(app, 'tradingview_tasi_close', side_effect=ValueError('tv unavailable')), \
             patch.object(app.clock, 'sleep'), \
             patch.object(app, 'yahoo_close', side_effect=ValueError('no history')), \
             patch.object(app, 'tasi_api_close', return_value=real) as api:
            ret = app.fetch_market(self.saudi, self.asof, None, min_sessions=2)
        self.assertEqual(len(ret), 3)
        api.assert_called_once()

    def test_missing_saudi_not_silently_replaced(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(app, 'tradingview_tasi_close', side_effect=ValueError('tv unavailable')), \
             patch.object(app.clock, 'sleep'), \
             patch.object(app, 'yahoo_close', side_effect=ValueError('no history')):
            with self.assertRaisesRegex(RuntimeError, 'SAUDI.csv'):
                app.fetch_market(self.saudi, self.asof, None, min_sessions=600)

    def test_excluded_saudi_requires_explicit_opt_in(self):
        dates = pd.bdate_range('2024-01-01', periods=600)
        rows = [{'date': str(d.date()), 'close': float(1000 + i)} for i, d in enumerate(dates)]
        payload = {
            'schema_version': app.SCHEMA_VERSION,
            'egx30': {'data': rows},
            'markets': {m.key: {'ticker': m.ticker, 'data': rows} for m in MARKETS if m.key != 'SAUDI'},
            'excluded_markets': ['SAUDI'],
        }
        with self.assertRaisesRegex(ValueError, '--allow-missing-markets'):
            app.validate_history(payload)
        egx, markets = app.validate_history(payload, allow_missing_markets=True)
        self.assertEqual(len(egx), 600)
        self.assertEqual(len(markets), len(MARKETS) - 1)
        self.assertNotIn('SAUDI', markets)

if __name__ == '__main__':
    unittest.main()
