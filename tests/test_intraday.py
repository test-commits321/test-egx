"""Same-day price inputs must be dated and complete before the Cairo cutoff."""
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

import intraday
from predictor import CAIRO


class IntradayTests(unittest.TestCase):
    def test_bar_end_and_publication_delay_are_required_in_backtest_and_live(self):
        day = '2026-11-23'
        bars = pd.Series([110., 250., 400.], index=pd.DatetimeIndex([
            f'{day}T14:20+08:00', f'{day}T14:40+08:00', f'{day}T15:10+08:00']))
        asof = datetime(2026, 11, 23, 16, tzinfo=CAIRO)
        end_of_history = intraday.latest_finished_bars('SHANGHAI', bars, asof)
        self.assertEqual(end_of_history[day]['price'], 110.)
        self.assertEqual(end_of_history[day]['available_at_cairo'][11:16], '09:40')
        too_early = asof.replace(hour=9, minute=39)
        self.assertFalse(intraday.latest_finished_bars('SHANGHAI', bars, too_early))

    def test_saudi_is_eligible_at_exact_cutoff_in_winter_only(self):
        bars = pd.Series([10100.0, 10200.0], index=pd.DatetimeIndex([
            '2026-07-12T10:00+03:00', '2026-11-22T10:00+03:00']))
        results = intraday.latest_finished_bars('SAUDI', bars, datetime(2026, 11, 23, 16, tzinfo=CAIRO))
        self.assertEqual(list(results), ['2026-11-22'])
        self.assertEqual(results['2026-11-22']['available_at_cairo'][11:16], '09:50')

    def test_late_markets_use_the_last_complete_return_without_future_prices(self):
        day = pd.Timestamp('2026-11-23')
        asof = datetime(2026, 11, 23, 9, 55, tzinfo=CAIRO)
        close = pd.Series([101., 100., 125.], index=pd.to_datetime(['2026-11-19', '2026-11-20', day]))
        quote = {'price': 102., 'price_at_cairo': '2026-11-23T09:10:00+02:00',
                 'available_at_cairo': '2026-11-23T09:30:00+02:00', 'source': 'test'}
        book = {'markets': {key: {'rows': {'2026-11-23': quote}} for key in ('NSE', 'NYSE', 'DOW')}}
        features, rows = intraday.snapshot(book, {key: close for key in ('NSE', 'NYSE', 'DOW')},
                                           day, asof, ('NSE', 'NYSE', 'DOW'))
        self.assertAlmostEqual(features['NSE_LATEST_RETURN'], .02)
        self.assertEqual(rows[0]['previous_session'], '2026-11-20')
        self.assertEqual([row['status'] for row in rows], ['AVAILABLE'] * 3)
        self.assertAlmostEqual(features[intraday.RISING_SHARE], 1 / 3)
        self.assertAlmostEqual(features['NYSE_LATEST_RETURN'], 100 / 101 - 1)
        self.assertEqual(rows[1]['price_session'], '2026-11-20')
        self.assertEqual(rows[1]['same_day_status'], 'OPENS_LATER')
        self.assertEqual(rows[1]['movement_basis'], 'LAST_CLOSED_SESSION')

    def test_a_missing_move_leaves_the_full_market_sample_ineligible(self):
        asof = datetime(2026, 11, 23, 9, 55, tzinfo=CAIRO)
        day = pd.Timestamp(asof.date())
        prices = pd.Series([100., 105.], index=pd.to_datetime(['2026-11-08', '2026-11-09']))
        features, rows = intraday.snapshot({'markets': {}}, {'NSE': prices}, day, asof, ('NSE', 'DOW'))
        self.assertTrue(pd.isna(features['NSE_LATEST_RETURN']))
        self.assertTrue(pd.isna(features['DOW_LATEST_RETURN']))
        self.assertFalse(intraday.verified_days(pd.DataFrame([features], index=[day])).iloc[0])
        self.assertNotIn('return', rows[1])

    def test_final_same_day_price_requires_close_to_have_been_published(self):
        day = pd.Timestamp('2026-11-23')
        bars = pd.Series([100., 103.], index=pd.to_datetime(['2026-11-20', '2026-11-23']))
        asof = datetime(2026, 11, 23, 9, 55, tzinfo=CAIRO)
        early = intraday.market_move('TAIWAN', day, bars, {}, asof.replace(hour=7, minute=40))
        later = intraday.market_move('TAIWAN', day, bars, {}, asof)
        self.assertEqual(early['status'], 'NO_VERIFIED_PRICE')
        self.assertAlmostEqual(later['return'], .03)
        self.assertEqual(later['source'], 'Final same-session close')

    def test_manual_bar_csv_requires_unambiguous_timezones(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'SAUDI_INTRADAY.csv'
            path.write_text('Timestamp,Close\n2026-11-23T10:00:00,100\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'timezone'):
                intraday.read_csv_prices(path)
            path.write_text('Timestamp,Close\n2026-11-23T10:00:00+03:00,100\n', encoding='utf-8')
            parsed = intraday.read_csv_prices(path)
            self.assertEqual(parsed.index[0].isoformat(), '2026-11-23T07:00:00+00:00')
