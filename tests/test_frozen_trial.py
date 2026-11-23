"""Synthetic data are used ONLY inside tests, never bundled as market observations."""
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from predictor import CAIRO, MARKETS


class ModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        dates = pd.date_range('2024-05-01', periods=960, freq='D')
        egy = dates[~dates.dayofweek.isin([4, 5])][-600:]
        self.egx = pd.Series(10000 * np.exp(np.cumsum(rng.normal(.0003, .008, len(egy)))), index=egy)
        self.payload = {
            'schema_version': app.SCHEMA_VERSION,
            'egx30': {'ticker': '^CASE30', 'data': app.date_rows(self.egx)},
            'markets': {},
        }
        self.foreign_series = {}
        for m in MARKETS:
            if m.key == 'SAUDI':
                sessions = dates[~dates.dayofweek.isin([4, 5])][-600:]
            else:
                sessions = dates[~dates.dayofweek.isin([5, 6])][-600:]
            series = pd.Series(1000 * np.exp(np.cumsum(rng.normal(.0001, .01, len(sessions)))), index=sessions)
            self.foreign_series[m.key] = series
            self.payload['markets'][m.key] = {'ticker': m.ticker, 'data': app.date_rows(series)}
        self.now = datetime(2026, 12, 17, 16, 0, tzinfo=CAIRO)

    def test_train_predict_freeze_and_report(self):
        mdl, egypt = app.train_from_history(self.payload, self.now)
        self.assertEqual(mdl['egx_prices_in_history'], 600)
        self.assertEqual(mdl['egx_target_days'], 599)
        self.assertEqual(len(mdl['market_weights']), len(MARKETS))
        self.assertEqual(set(mdl['active_market_keys']), {m.key for m in MARKETS})
        self.assertIn('majority_baseline_brier', mdl['historical_walk_forward'])
        self.assertAlmostEqual(sum(x['model_weight'] for x in mdl['market_weights']), 1.0, places=6)
        feat = {c: 0.005 for c in mdl['features']}
        p1 = app.predict_proba_frozen(mdl, feat)
        self.assertTrue(0 < p1 < 1)
        state = {'schema_version': app.SCHEMA_VERSION, 'trial_target': 30,
                 'started_at_cairo': self.now.isoformat(), 'predictions': [],
                 'recent_egx': app.date_rows(egypt.tail(6)), 'completed_at_cairo': None}
        self.assertEqual(app.summarize(state)['status'], 'IN_PROGRESS')
        self.assertEqual(app.summarize(state)['accuracy'], None)
        with tempfile.TemporaryDirectory() as d:
            rt = Path(d)
            app.dump_atomic(rt / 'model.json', mdl)
            app.dump_atomic(rt / 'state.json', state)
            app.write_reports(rt, state, mdl)
            self.assertTrue((rt / 'predictions_30.csv').exists())
            self.assertEqual(app.read_json(rt / 'model.json')['trained_egx_last_date'], str(egypt.index[-1].date()))

    def test_only_preopen_and_never_double_forecast(self):
        mdl, egy = app.train_from_history(self.payload, self.now)
        d = datetime(2026, 12, 20, 9, 55, tzinfo=CAIRO)
        self.assertIsNotNone(app.schedule_candidate(d))
        self.assertIsNone(app.schedule_candidate(d.replace(hour=10)))
        self.assertIsNone(app.schedule_candidate(d.replace(hour=9, minute=49)))
        self.assertIsNone(app.schedule_candidate(datetime(2026, 12, 19, 9, 55, tzinfo=CAIRO)))
        state = {'schema_version': app.SCHEMA_VERSION, 'trial_target': 30,
                 'started_at_cairo': self.now.isoformat(), 'predictions': [],
                 'recent_egx': app.date_rows(egy.tail(6)), 'completed_at_cairo': None}
        # Need to mock fresh post-training foreign closes, not data known from the future.
        new_foreign = {}
        for m in MARKETS:
            s = self.foreign_series[m.key]
            extended = pd.concat([s, pd.Series([float(s.iloc[-1])*1.002],
                                index=[s.index[-1] + timedelta(days=3)])])
            new_foreign[m.key] = extended
        with patch.object(app, 'fetch_market', side_effect=lambda m, *rest, **kw: new_foreign[m.key]):
            self.assertFalse(app.live_forecast(state, mdl, d, None, publication_clock=lambda: d.replace(hour=10)))
            self.assertTrue(app.live_forecast(state, mdl, d, None, publication_clock=lambda: d))
            self.assertFalse(app.live_forecast(state, mdl, d, None, publication_clock=lambda: d))
        self.assertEqual(len(state['predictions']), 1)
        self.assertEqual(state['predictions'][0]['status'], 'PENDING')
        self.assertEqual(state['predictions'][0]['date'], '2026-12-20')
        self.assertIsNone(state['predictions'][0]['actual_egx_close'])
        self.assertEqual(app.settle_pending(state, pd.Series([], dtype=float), d), 0)
        before = app.summarize(state)
        self.assertEqual(before['evaluated_predictions'], 0)
        d_late = datetime(2026, 12, 20, 15, 50, tzinfo=CAIRO)
        previous = state['predictions'][0]['previous_egx_close']
        after_close = pd.Series([previous, previous * 1.01], index=pd.DatetimeIndex([egy.index[-1], '2026-12-20']))
        self.assertEqual(app.settle_pending(state, after_close, d_late), 1)
        self.assertEqual(app.settle_pending(state, after_close, d_late), 0)
        self.assertEqual(app.summarize(state)['evaluated_predictions'], 1)
        self.assertEqual(app.summarize(state)['correct'], int(state['predictions'][0]['direction']=='UP'))

class TrialClockTests(unittest.TestCase):
    def test_binary_non_up_scores_flat_and_voids_wrong_reference(self):
        asof = datetime(2026, 12, 20, 16, tzinfo=CAIRO)
        egx = pd.Series([98.0, 100.0, 100.0], index=pd.to_datetime(['2026-12-16', '2026-12-17', '2026-12-20']))
        flat = {'date': '2026-12-20', 'status': 'PENDING', 'model_version': 2, 'direction': 'NON_UP',
                'source_last_egx_day': '2026-12-17', 'previous_egx_close': 100.0}
        stale = {**flat, 'source_last_egx_day': '2026-12-16', 'previous_egx_close': 98.0}
        self.assertEqual(app.settle_pending({'predictions': [flat, stale]}, egx, asof), 2)
        self.assertEqual(flat['actual_direction'], 'FLAT')
        self.assertTrue(flat['correct'])
        self.assertEqual(stale['status'], 'VOID')
        self.assertEqual(stale['void_kind'], 'EGX_REFERENCE')

    def test_runtime_lock_rejects_a_second_writer(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = Path(folder)
            with app.trial_lock(runtime):
                with self.assertRaisesRegex(RuntimeError, 'Another process'):
                    with app.trial_lock(runtime):
                        self.fail('second writer entered')
            with app.trial_lock(runtime):
                pass
    def test_calendar_expiration_and_pending_settlement(self):
        s = {'trial_mode': 'calendar', 'trial_target': 30, 'first_forecast_date': '2026-12-20',
             'end_exclusive_date': '2027-01-19',
             'predictions': [
                {'date': '2027-01-18', 'status': 'PENDING', 'direction': 'UP', 'prob_up': .6,
                 'previous_egx_close': 100.0, 'correct': None}],
             'recent_egx': [{'date': '2027-01-15', 'close':100.0}], 'completed_at_cairo':None}
        before = app.summarize(s, datetime(2027, 1, 18, 18, tzinfo=CAIRO))
        self.assertEqual(before['status'], 'IN_PROGRESS')
        self.assertEqual(before['trial_mode'], 'calendar')
        self.assertIsNone(before['remaining_to_issue'])
        after_end = app.summarize(s, datetime(2027, 1, 19, 9, tzinfo=CAIRO))
        self.assertEqual(after_end['status'], 'IN_PROGRESS')  # Pending never auto-scored.
        s['predictions'][0].update(status='SETTLED', correct=True, actual_direction='UP')
        finished = app.summarize(s, datetime(2027, 1, 19, 16, tzinfo=CAIRO))
        self.assertEqual(finished['status'], 'COMPLETE')
        self.assertEqual(finished['accuracy'], 1.0)
        self.assertEqual(finished['correct'], 1)
        self.assertEqual(finished['evaluated_predictions'], 1)

    def test_session_mode_30_scores_required(self):
        s = {'trial_mode': 'sessions', 'trial_target': 30, 'predictions': [],
             'first_forecast_date': None, 'end_exclusive_date': None}
        for i in range(29):
            s['predictions'].append({'status':'SETTLED','correct': i%2==0,
                 'direction': 'UP', 'actual_direction': 'UP' if i%2==0 else 'DOWN', 'prob_up': .7})
        self.assertEqual(app.summarize(s)['status'], 'IN_PROGRESS')
        s['predictions'].append({'status':'SETTLED', 'correct':True,
                                'direction':'UP','actual_direction':'UP', 'prob_up':.6})
        self.assertEqual(app.summarize(s)['status'], 'COMPLETE')


if __name__ == '__main__':
    unittest.main()
