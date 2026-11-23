"""Point-in-time EGX data source behavior (all test prices are synthetic)."""
import io
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from predictor import CAIRO, parse_close_csv


class EGXFeedTests(unittest.TestCase):
    def test_egx_exchange_price_fields_and_request_timeout(self):
        days = pd.date_range('2023-01-01', periods=920, freq='D')
        day_labels = days.strftime('%Y-%m-%dT00:00:00+02:00')
        rows = pd.DataFrame({'indexDay': day_labels, 'indexClose': 14000 + np.arange(920)})
        data = json.dumps({'success': True, 'data': rows.to_dict(orient='records')})
        response = io.BytesIO(data.encode())
        asof = datetime(2025, 7, 10, 16, 30, tzinfo=CAIRO)
        with patch.object(app, 'urlopen', return_value=response) as request, \
             patch.object(app, 'yahoo_close', side_effect=AssertionError('Yahoo should not run')):
            result = app.fetch_egx(asof, None, None, 1300)
        self.assertIn('indexName=CASE30', request.call_args.args[0].full_url)
        self.assertEqual(request.call_args.kwargs['timeout'], 8)
        self.assertGreater(len(result), 600)
        self.assertEqual(result.index[0], days[0])
        self.assertEqual(result.index.max(), days[-1])

    def test_csv_override_does_not_call_network(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'EGX30.csv'
            path.write_text('Date,Close\n2026-11-18,51000.11\n2026-11-19,52000.22\n')
            asof = datetime(2026, 11, 23, 16, 30, tzinfo=CAIRO)
            with patch.object(app, 'egx_exchange_close', side_effect=AssertionError('Should not call provider')):
                close = app.fetch_egx(asof, None, path)
            self.assertEqual(len(close), 2)
            self.assertAlmostEqual(close.iloc[-1], 52000.22)

    def test_market_data_github_csv_format(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / 'EGX30.csv'
            file.write_text('time,datetime_utc,open,high,low,close,volume\n'
                            '1791356400,2026-10-07 07:00,53000,54000,52000,53265.20,172000000\n')
            parsed = parse_close_csv(file)
            self.assertEqual(parsed.index[-1], pd.Timestamp('2026-10-07'))
            self.assertAlmostEqual(parsed.iloc[-1], 53265.2)

    def test_unavailable_both_sources_gives_actionable_error(self):
        with patch.object(app, 'egx_exchange_close', side_effect=RuntimeError('site down')), \
             patch.object(app, 'yahoo_close', side_effect=ValueError('Yahoo no history')):
            with self.assertRaisesRegex(RuntimeError, '--egx-csv'):
                app.fetch_egx(datetime(2026, 11, 23, 16, tzinfo=CAIRO), None, None)

    def test_short_or_stale_primary_history_tries_yahoo(self):
        asof = datetime(2026, 11, 23, 16, tzinfo=CAIRO)
        current = pd.Series([100., 102.], index=pd.to_datetime(['2026-11-22', '2026-11-23']))
        stale = current.set_axis(pd.to_datetime(['2026-10-22', '2026-10-25']))
        for incomplete in (current.iloc[:1], stale):
            with self.subTest(primary_dates=list(incomplete.index)), \
                 patch.object(app, 'egx_exchange_close', return_value=incomplete), \
                 patch.object(app, 'yahoo_close', return_value=current) as yahoo:
                result = app.fetch_egx(asof, None, None, min_sessions=2, max_age_days=14)
                yahoo.assert_called_once()
                pd.testing.assert_series_equal(result, current)

    def test_missing_explicit_csv_does_not_fall_through(self):
        asof = datetime(2026, 11, 23, 16, tzinfo=CAIRO)
        with tempfile.TemporaryDirectory() as folder, patch.object(app, 'egx_exchange_close') as network:
            with self.assertRaisesRegex(ValueError, '--egx-csv file does not exist'):
                app.fetch_egx(asof, None, Path(folder) / 'missing.csv')
            network.assert_not_called()


if __name__ == '__main__':
    unittest.main()
