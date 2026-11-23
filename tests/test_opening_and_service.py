"""Regression checks for market clocks, opening gaps, and unattended operation."""
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import pandas as pd
import numpy as np

import app
import intraday
import opening
from predictor import CAIRO


class OpeningTests(TestCase):
    def test_saudi_only_precedes_egypt_in_winter(self):
        november_sunday = opening.scheduled_opening_markets('2026-11-22')
        july_sunday = opening.scheduled_opening_markets('2026-07-12')
        july_monday = opening.scheduled_opening_markets('2026-07-13')
        self.assertEqual(november_sunday, ('SAUDI',))
        self.assertEqual(july_sunday, ())
        self.assertIn('NSE', july_monday)
        self.assertNotIn('SAUDI', july_monday)
        self.assertNotIn('FTSE', july_monday)

    def test_open_gap_uses_yesterdays_close_and_only_ready_bars(self):
        days = pd.to_datetime(['2026-11-19', '2026-11-22', '2026-11-23'])
        data = pd.DataFrame({'Open': [100, 105, 106], 'Close': [110, 100, 990]}, index=days)
        before = datetime(2026, 11, 23, 9, 55, tzinfo=CAIRO)
        after = before.replace(hour=10, minute=25)
        self.assertNotIn(days[-1], opening.known_prices('EGX30', data, before).index)
        published = opening.known_prices('EGX30', data, after)
        self.assertTrue(pd.isna(published.loc[days[-1], 'Close']))
        self.assertAlmostEqual(opening.opening_gaps(published).loc[days[-1]], 0.06)

    def test_counts_pair_same_date_and_keep_flat_in_the_complement(self):
        days = pd.to_datetime(['2026-11-19', '2026-11-22', '2026-11-23', '2026-11-24'])
        egx = pd.Series([.01, -.02, 0, .025], index=days)
        japan = pd.Series([.01, .01, -.03], index=days[[0, 2, 3]])
        counts = opening.individual_counts(opening.align_history({'EGX30': egx, 'JAPAN': japan}))[0]
        self.assertEqual(counts['paired_sessions'], 3)
        self.assertEqual(counts['foreign_opened_up'], 2)
        self.assertEqual(counts['both_opened_up'], 1)
        self.assertEqual(counts['egypt_up_given_foreign_up'], .5)
        self.assertEqual(counts['egypt_up_given_foreign_non_up'], 1.)

    def test_issue_once_and_exclude_todays_egypt_open_from_model(self):
        asof = datetime(2026, 11, 23, 9, 55, tzinfo=CAIRO)
        days = pd.date_range('2026-08-01', str(asof.date()))
        days = days[days.dayofweek.isin(range(4))]
        gaps = {'EGX30': pd.Series([.01] * (len(days) - 1) + [-.01], index=days)}
        gaps.update({key: pd.Series(.01, index=days) for key in ('JAPAN', 'SHANGHAI', 'SHENZHEN')})
        state = {'predictions': []}
        self.assertFalse(opening.publish(state, gaps, asof, lambda: asof.replace(hour=10)))
        self.assertTrue(opening.publish(state, gaps, asof, lambda: asof))
        self.assertFalse(opening.publish(state, gaps, asof, lambda: asof))
        issued = state['predictions'][0]
        japan = issued['market_conditions']['JAPAN']
        self.assertEqual(japan['egypt_up'], len(days) - 1)
        self.assertEqual(japan['egypt_up'], japan['sessions'])
        self.assertEqual(issued['groups'], 2)
        self.assertFalse(opening.settle(state, gaps, asof))
        self.assertTrue(opening.settle(state, gaps, asof.replace(hour=10, minute=25)))
        self.assertFalse(issued['correct'])
        self.assertFalse(opening.settle(state, gaps, asof.replace(hour=10, minute=30)))

    def test_missing_egypt_ohlc_does_not_stop_the_service_or_score_a_forecast(self):
        asof = datetime(2026, 11, 23, 9, 55, tzinfo=CAIRO)
        dates = pd.to_datetime(['2026-11-23'])
        gaps = {'JAPAN': pd.Series(.01, index=dates)}
        state = {'predictions': [{'date': '2026-11-22', 'status': 'PENDING'}]}
        self.assertFalse(opening.publish(state, gaps, asof, lambda: asof))
        self.assertFalse(opening.settle(state, gaps, asof.replace(hour=11)))
        self.assertEqual(state['predictions'][0]['status'], 'PENDING')

    def test_600_calendar_days_and_prior_only_walk_forward(self):
        end = pd.Timestamp('2026-11-23')
        dates = pd.date_range(end - timedelta(days=620), end, freq='D')
        gaps = {'EGX30': pd.Series(.01, index=dates), 'JAPAN': pd.Series(.01, index=dates)}
        gaps['EGX30'].loc[end] = -.01
        gaps['EGX30'].loc[dates < (end - timedelta(days=600))] = -.01
        history = opening.align_history(gaps)
        prior = opening.calendar_window(history, end)
        self.assertEqual(len(prior), 600)
        estimate = opening.predict_from_opens(prior, {'JAPAN': .01})
        self.assertEqual(estimate['market_conditions']['JAPAN']['egypt_up'], 600)
        self.assertEqual(estimate['market_conditions']['JAPAN']['sessions'], 600)
        score = opening.walk_forward(history, end, end + timedelta(days=1))
        self.assertEqual(score['evaluated_sessions'], 1)
        self.assertAlmostEqual(score['brier'], estimate['prob_up'] ** 2)


class ServiceTests(TestCase):
    def test_morning_window_stops_at_10_cairo(self):
        with TemporaryDirectory() as directory:
            args = app.make_parser().parse_args(['start', '--until', '10:00', '--runtime', directory])
            current = datetime(2026, 11, 23, 9, 59, 50, tzinfo=CAIRO)

            def clock():
                return current

            def sleep(seconds):
                nonlocal current
                current += timedelta(seconds=seconds)

            with patch.object(app, 'scheduled_openings') as opens, patch.object(app, 'scheduled_closes') as closes, \
                    patch.object(app, 'scheduled_global_moves') as markets, patch.object(app, 'now_cairo', side_effect=clock), \
                    patch.object(app.clock, 'sleep', side_effect=sleep) as pause:
                app.start_service(args)
                app.start_service(args)
            pause.assert_called_once_with(10.0)
            opens.assert_called_once()
            closes.assert_called_once()
            markets.assert_called_once()

    def test_closed_regular_day_rolls_back_to_thursday_on_friday(self):
        friday = datetime(2026, 11, 27, 16, tzinfo=CAIRO)
        sunday = datetime(2026, 11, 29, 9, 55, tzinfo=CAIRO)
        time_of_close = app.time(15, 15)
        self.assertEqual(app.expected_egx_day(friday, time_of_close), '2026-11-26')
        self.assertEqual(app.expected_egx_day(sunday, time_of_close), '2026-11-26')
        self.assertEqual(app.expected_egx_day(sunday.replace(hour=16), time_of_close), '2026-11-29')

    def test_existing_work_does_not_fetch_on_an_idle_evening(self):
        asof = datetime(2026, 11, 27, 22, tzinfo=CAIRO)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = app.make_parser().parse_args(['start', '--runtime', str(root), '--once'])
            for location, mode in [(root, 'calendar'), (root / 'adaptive', 'continuous')]:
                model = {'schema_version': app.SCHEMA_VERSION, 'trained_egx_last_date': '2026-11-26'}
                state = {'schema_version': app.SCHEMA_VERSION, 'trial_mode': mode, 'predictions': [], 'completed_at_cairo': None}
                app.dump_atomic(location / 'model.json', model)
                app.dump_atomic(location / 'state.json', state)
                app.dump_atomic(location / 'report.json', {})
            app.dump_atomic(root / 'opening' / 'report.json', {'egx_open_sessions': 600, 'last_egx_open_session': '2026-11-26'})
            with patch.object(app, 'fetch_egx') as egx, patch.object(opening, 'refresh_prices') as early:
                app.scheduled_openings(args, asof, {}, lambda: asof)
                app.scheduled_closes(args, asof, {}, lambda: asof)
            egx.assert_not_called()
            early.assert_not_called()

    def test_missing_close_is_voided_only_after_two_known_later_sessions(self):
        asof = datetime(2026, 11, 26, 17, tzinfo=CAIRO)
        row = {'date': '2026-11-23', 'status': 'PENDING'}
        egx = pd.Series([100.0, 101.0, 102.0], index=pd.to_datetime(['2026-11-22', '2026-11-24', '2026-11-25']))
        self.assertEqual(app.settle_pending({'predictions': [row]}, egx.iloc[:2], asof, allow_unverified_void=True), 0)
        self.assertEqual(app.settle_pending({'predictions': [row]}, egx, asof, allow_unverified_void=True), 1)
        self.assertEqual(row['status'], 'VOID')
        self.assertEqual(row['void_kind'], 'NO_VERIFIED_SESSION')

    def test_first_start_then_five_live_predictions_and_one_daily_update(self):
        morning = datetime(2026, 11, 23, 9, 31, tzinfo=CAIRO)
        all_dates = pd.date_range('2023-08-01', morning.date())
        today = pd.Timestamp(morning.date())
        rng = np.random.default_rng(208)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            csvs = root / 'csv'
            csvs.mkdir()
            runtime = root / 'runtime'
            history = root / 'history.json'
            params = ['start', '--once', '--runtime', str(runtime), '--history', str(history), '--csv-dir', str(csvs)]

            def bars(dates, initial):
                closes = initial * np.exp(np.cumsum(rng.normal(.0004, .017, len(dates))))
                opens = np.r_[initial, closes[:-1]] * np.exp(rng.normal(.001, .008, len(dates)))
                return pd.DataFrame({'Open': opens, 'Close': closes}, index=dates).rename_axis('Date')

            for market in app.MARKETS:
                weekend = (4, 5) if market.key == 'SAUDI' else (5, 6)
                foreign = bars(all_dates[~all_dates.dayofweek.isin(weekend)], 1000)
                foreign[['Close']].to_csv(csvs / market.csv_name)
                if market.key in opening.EARLY_MARKETS:
                    foreign.loc[today, 'Open'] = foreign['Close'].iloc[-2] * 1.01
                    foreign.to_csv(csvs / f'{market.key}_OHLC.csv')
                    hour, minute = map(int, intraday.OPEN_TIMES[market.key].split(':'))
                    times = foreign.index.tz_localize(market.timezone) + pd.to_timedelta(hour * 60 + minute, unit='min')
                    first_bars = pd.Series(foreign['Open'].to_numpy(), index=times.tz_convert('UTC'), name='Close')
                    first_bars.rename_axis('Timestamp').to_csv(csvs / f'{market.key}_INTRADAY.csv')
            egypt = bars(all_dates[~all_dates.dayofweek.isin((4, 5))], 30000)
            egypt.iloc[:-1][['Close']].to_csv(csvs / 'EGX30.csv')
            egypt.iloc[:-1].to_csv(csvs / 'EGX30_OHLC.csv')

            def run(at):
                with patch.object(app, 'now_cairo', return_value=at), patch.object(app, 'DEFAULT_MARKET_CACHE', root / 'cache'):
                    app.main(params)

            model_file = runtime / 'model.json'
            adaptive_file = runtime / 'adaptive' / 'model.json'
            today_fixed = runtime / 'global_moves' / 'frozen' / 'model.json'
            today_adaptive = runtime / 'global_moves' / 'adaptive' / 'model.json'
            state_files = [runtime / 'state.json', runtime / 'opening' / 'state.json', runtime / 'adaptive' / 'state.json',
                           today_fixed.with_name('state.json'), today_adaptive.with_name('state.json')]
            run(morning)
            frozen = model_file.read_bytes()
            same_day_frozen = today_fixed.read_bytes()
            start_weights = app.read_json(today_adaptive)['coefficients']
            self.assertTrue(adaptive_file.is_file())
            self.assertEqual(app.read_json(adaptive_file)['learning_updates'], 0)
            self.assertEqual(app.read_json(today_fixed)['active_market_keys'], app.read_json(model_file)['active_market_keys'])
            self.assertEqual(app.read_json(today_adaptive)['active_market_keys'], app.read_json(model_file)['active_market_keys'])

            preopen = morning.replace(minute=52)
            run(preopen)
            run(preopen.replace(minute=55))
            forecasts = [app.read_json(path)['predictions'] for path in state_files]
            self.assertEqual([len(records) for records in forecasts], [1, 1, 1, 1, 1])
            self.assertTrue(all(records[0]['status'] == 'PENDING' for records in forecasts))
            self.assertEqual(forecasts[2][0]['model_trained_through'], '2026-11-22')
            self.assertEqual(forecasts[4][0]['model_trained_through'], '2026-11-22')
            self.assertEqual(forecasts[3][0]['latest_markets_count'], len(app.MARKETS))
            self.assertEqual(forecasts[3][0]['market_moves'], forecasts[4][0]['market_moves'])
            us500 = next(row for row in forecasts[3][0]['market_moves'] if row['market'] == 'NYSE')
            self.assertEqual(us500['movement_basis'], 'LAST_CLOSED_SESSION')
            self.assertLess(us500['price_session'], '2026-11-23')

            egypt.to_csv(csvs / 'EGX30_OHLC.csv')
            run(morning.replace(hour=10, minute=26))
            self.assertEqual(app.read_json(state_files[1])['predictions'][0]['status'], 'SETTLED')
            self.assertEqual(app.read_json(state_files[2])['predictions'][0]['status'], 'PENDING')
            self.assertEqual(app.read_json(state_files[4])['predictions'][0]['status'], 'PENDING')

            egypt[['Close']].to_csv(csvs / 'EGX30.csv')
            run(morning.replace(hour=15, minute=30))
            run(morning.replace(hour=16, minute=30))
            adaptive = app.read_json(adaptive_file)
            self.assertEqual(adaptive['trained_egx_last_date'], '2026-11-23')
            self.assertEqual(adaptive['learning_updates'], 1)
            self.assertEqual(app.read_json(state_files[2])['predictions'][0]['status'], 'SETTLED')
            self.assertEqual(app.read_json(state_files[0])['predictions'][0]['status'], 'SETTLED')
            self.assertEqual(app.read_json(state_files[3])['predictions'][0]['status'], 'SETTLED')
            self.assertEqual(app.read_json(state_files[4])['predictions'][0]['status'], 'SETTLED')
            updated = app.read_json(today_adaptive)
            self.assertEqual(updated['trained_egx_last_date'], '2026-11-23')
            self.assertEqual(updated['learning_updates'], 1)
            self.assertNotEqual(updated['coefficients'], start_weights)
            self.assertEqual(today_fixed.read_bytes(), same_day_frozen)
            self.assertEqual(model_file.read_bytes(), frozen)
