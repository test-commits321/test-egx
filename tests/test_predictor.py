import sys
import tempfile
import unittest
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predictor import (CAIRO, MARKETS, SIGNAL_THRESHOLD, assemble_market_features, country_weights,
                       egx_controls, evaluate, fit_direction_model, make_model, parse_close_csv, walk_forward,
                       wilson_interval)

class PredictorTests(unittest.TestCase):
    def test_market_count(self):
        self.assertEqual(len(MARKETS), 18)
        tickers = {market.ticker for market in MARKETS}
        self.assertTrue({'^GSPC', '^DJI', '^FTSE', '^IBEX'} <= tickers)
        self.assertEqual(len(tickers), len(MARKETS))

    def test_no_future_return_leakage(self):
        # January Shanghai bars are available at 09:30 Cairo and HK bars at 10:30.
        prices = pd.Series([100, 99, 103, 102, 104, 105, 115, 98],
                           index=pd.bdate_range('2026-01-05', periods=8))
        closes = {'HONG_KONG': prices, 'SHANGHAI': prices}
        days = prices.index[-2:][::-1]  # Caller order must be preserved.
        day = prices.index[-2]
        frame = assemble_market_features(days, closes, datetime(2026, 1, 15, 12, tzinfo=CAIRO), 30)
        self.assertTrue(frame.index.equals(days))
        self.assertAlmostEqual(frame.at[day, 'HONG_KONG'], 105 / 104 - 1)
        self.assertAlmostEqual(frame.at[day, 'HONG_KONG_MOM_5'], 105 / 100 - 1)
        self.assertAlmostEqual(frame.at[day, 'SHANGHAI'], 115 / 105 - 1)
        self.assertAlmostEqual(frame.at[day, 'SHANGHAI_MOM_5'], 115 / 99 - 1)
        early = assemble_market_features(days[:1], closes, datetime(2026, 1, 14, 9, 29, tzinfo=CAIRO), 30)
        self.assertAlmostEqual(early.iloc[0]['SHANGHAI'], 115 / 105 - 1)

    def test_short_market_history_and_staleness_do_not_use_future_prices(self):
        closes = {'DOW': pd.Series([100, 110], index=pd.to_datetime(['2026-01-01', '2026-01-02']))}
        days = pd.to_datetime(['2026-01-04', '2026-01-14'])
        frame = assemble_market_features(days, closes, datetime(2026, 1, 15, 12, tzinfo=CAIRO), 30)
        self.assertAlmostEqual(frame.iloc[0]['DOW'], .10)
        self.assertTrue(pd.isna(frame.iloc[0]['DOW_MOM_5']))
        self.assertTrue(pd.isna(frame.iloc[1]['DOW']))

    def test_egx_lags(self):
        d = pd.date_range('2026-01-01', periods=10, freq='D')
        closes = pd.Series(range(100, 110), index=d, dtype=float)
        c = egx_controls(d, closes)
        self.assertTrue(np.isnan(c.loc[d[0], 'EGX_PREV_1']))
        self.assertAlmostEqual(c.loc[d[2], 'EGX_PREV_1'], 1/100)
        trailing = closes.pct_change(fill_method=None).iloc[1:6]
        self.assertAlmostEqual(c.loc[d[6], 'EGX_VOL_5'], trailing.std(ddof=0))
        self.assertAlmostEqual(c.loc[d[6], 'EGX_PREV_5'], 105/100 - 1)

    def test_malformed_csv_row_does_not_invalidate_epoch_closes(self):
        seconds = int(pd.Timestamp('2026-01-07T02:00:00Z').timestamp())
        millis = int(pd.Timestamp('2026-01-08T02:00:00Z').timestamp() * 1000)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'example.csv'
            path.write_text(f'Date,Close\n2026-01-05,100\n{seconds},101\ninvalid,99\n'
                            f'{millis},102\n2026-01-08T02:00:00Z,103\n', encoding='utf-8')
            result = parse_close_csv(path, 'America/New_York')
        self.assertEqual(result.index.tolist(), list(pd.date_range('2026-01-05', periods=3)))
        self.assertEqual(result.tolist(), [100, 101, 103])

    def test_no_random_split_backtest(self):
        rng = np.random.default_rng(9)
        d = pd.date_range('2024-01-01', periods=230, freq='D')
        X = pd.DataFrame({'signal': rng.normal(size=len(d)), 'other': rng.normal(size=len(d))}, index=d)
        y = pd.Series((X.signal > 0).astype(int), index=d)
        pred = walk_forward(X, y, min_train=170)
        self.assertEqual(len(pred), 60)
        scores = evaluate(pred)
        self.assertGreater(scores['accuracy'], 0.75)
        self.assertGreater(scores['brier_skill_vs_majority'], 0)
        self.assertEqual(scores['signal_threshold'], SIGNAL_THRESHOLD)
        self.assertAlmostEqual(pred.iloc[0]['baseline_probability_up'], (y.iloc[:170].sum() + 1) / 172)
        self.assertAlmostEqual(pred.iloc[0]['recent_baseline_probability_up'], (y.iloc[110:170].sum() + 1) / 62)
        changed_y = y.copy()
        changed_y.iloc[170:] = 1 - changed_y.iloc[170:]
        changed = walk_forward(X, changed_y, min_train=170)
        self.assertEqual(pred.iloc[0]['probability_up'], changed.iloc[0]['probability_up'])

    def test_legacy_comparison_excludes_new_features(self):
        rng = np.random.default_rng(12)
        d = pd.date_range('2024-01-01', periods=120, freq='D')
        X = pd.DataFrame(rng.normal(size=(120, 3)), index=d, columns=['NYSE', 'DOW', 'NYSE_MOM_5'])
        y = (X['NYSE_MOM_5'] > 0).astype(int)
        preds = walk_forward(X, y, min_train=70)
        expected_legacy = make_model().fit(X[['NYSE']].iloc[:70], y.iloc[:70])
        np.testing.assert_allclose(preds.iloc[0]['legacy_model_probability_up'],
                                   expected_legacy.predict_proba(X[['NYSE']].iloc[70:71])[0, 1])
        self.assertGreater(evaluate(preds)['brier_skill_vs_legacy'], 0)

    def test_block_reuses_one_fit_without_using_labels_in_block(self):
        rng = np.random.default_rng(13)
        dates = pd.date_range('2024-01-01', periods=120, freq='D')
        X = pd.DataFrame({'NYSE': rng.normal(size=len(dates))}, index=dates)
        y = (X['NYSE'] > 0).astype(int)
        changed_y = y.copy()
        changed_y.iloc[70:90] = 1 - changed_y.iloc[70:90]
        predictions = walk_forward(X, y, min_train=70, refit_every=20)
        changed = walk_forward(X, changed_y, min_train=70, refit_every=20)
        expected = [dates[69]] * 20 + [dates[89]] * 20 + [dates[109]] * 10
        self.assertEqual(predictions['trained_through'].tolist(), expected)
        columns = ['probability_up', 'legacy_model_probability_up', 'baseline_probability_up',
                   'recent_baseline_probability_up']
        pd.testing.assert_frame_equal(predictions.iloc[:20][columns], changed.iloc[:20][columns])

    def test_weights_sum_and_confidence(self):
        rng = np.random.default_rng(22)
        X = pd.DataFrame(rng.normal(size=(210, len(MARKETS))), columns=[m.key for m in MARKETS])
        X['NYSE_MOM_5'] = rng.normal(size=len(X))
        y = (X['NYSE_MOM_5'] + rng.normal(scale=.2, size=210) > 0).astype(int)
        m = fit_direction_model(X, y)
        exch, countries = country_weights(m, X.columns.tolist())
        self.assertAlmostEqual(exch.model_weight.sum(), 1.0)
        self.assertAlmostEqual(countries.model_weight.sum(), 1.0)
        self.assertGreater(exch.set_index('key').loc['NYSE', 'signed_momentum_standardized_coefficient'], 0)
        lo, hi = wilson_interval(7, 10)
        self.assertTrue(lo < .7 < hi)

if __name__ == '__main__':
    unittest.main()
