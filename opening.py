"""Early-market opening gaps and a separate EGX30 opening forecast."""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import time, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from predictor import (
    CAIRO, EGX_CLOSE, EGX_OPEN, FORECAST_CUTOFF, MARKETS, PUBLICATION_BUFFER_MINUTES,
    local_datetime, parse_daily_csv, wilson_interval, yahoo_daily_prices,
)
from runtime_io import atomic_text, dump_atomic, read_json
from tradingview_feed import tradingview_daily_prices

HISTORY_DAYS = 600
REPORT_VERSION = 2
MIN_CONDITIONAL_SESSIONS = 30
OPENING_PREP = time(9, 30)
OPENING_RESULT = time(10, 25)
FOREIGN_OPEN_DELAY = timedelta(minutes=20)
FOREIGN_OPEN_TIMES = {
    'JAPAN': '09:00', 'KOREA': '09:00', 'TAIWAN': '09:00',
    'SHANGHAI': '09:30', 'SHENZHEN': '09:30', 'HONG_KONG': '09:30',
    'BSE': '09:15', 'NSE': '09:15', 'SAUDI': '10:00',
}
EARLY_MARKETS = {m.key: m for m in MARKETS if m.key in FOREIGN_OPEN_TIMES}
# Each group gets one vote in the combined probability.
OPENING_GROUPS = (('JAPAN',), ('KOREA',), ('TAIWAN',), ('SHANGHAI', 'SHENZHEN'),
                  ('HONG_KONG',), ('BSE', 'NSE'), ('SAUDI',))
MARKET_NAMES = dict(zip(FOREIGN_OPEN_TIMES, (
    'Nikkei 225', 'KOSPI', 'TAIEX', 'Shanghai Composite', 'Shenzhen Component',
    'Hang Seng', 'Sensex', 'Nifty 50', 'TASI',
)))


@lru_cache(maxsize=1600)
def markets_for_date(day):
    cutoff = local_datetime(day, FORECAST_CUTOFF.strftime('%H:%M'), 'Africa/Cairo')
    weekday = day.weekday()
    return tuple(key for key, market in EARLY_MARKETS.items()
                 if weekday in ((6, 0, 1, 2, 3) if key == 'SAUDI' else (0, 1, 2, 3))
                 and local_datetime(day, FOREIGN_OPEN_TIMES[key], market.timezone) + FOREIGN_OPEN_DELAY <= cutoff)


def scheduled_opening_markets(day):
    return markets_for_date(pd.Timestamp(day).date())


def fetch_daily_prices(key, asof, csv_dir, lookback):
    market = EARLY_MARKETS.get(key)
    zone = market.timezone if market else 'Africa/Cairo'
    path = csv_dir / f'{key}_OHLC.csv' if csv_dir else None
    if path and path.is_file():
        return parse_daily_csv(path, zone, require_open=True), f'CSV {path.name}'
    if key in ('EGX30', 'SAUDI'):
        symbol, exchange = ('EGX30', 'EGX') if key == 'EGX30' else ('TASI', 'TADAWUL')
        try:
            prices = tradingview_daily_prices(symbol, exchange, zone, 900 if lookback > 100 else 32)
        except (RuntimeError, OSError, ValueError) as first:
            print(f'{key} opening data: {first}; retrying TradingView.')
            prices = tradingview_daily_prices(symbol, exchange, zone, 900 if lookback > 100 else 32)
        return prices, f'TradingView {exchange}:{symbol}'
    return yahoo_daily_prices(market.ticker, asof - timedelta(days=lookback), asof + timedelta(days=1)), f'Yahoo {market.ticker}'


def known_prices(key, prices, asof):
    prices = prices.copy()
    if key == 'EGX30':
        zone, open_time, close_time = 'Africa/Cairo', OPENING_RESULT.strftime('%H:%M'), EGX_CLOSE.strftime('%H:%M')
        open_delay, close_delay = timedelta(), timedelta(minutes=45)
    else:
        market = EARLY_MARKETS[key]
        zone, open_time, close_time = market.timezone, FOREIGN_OPEN_TIMES[key], market.close
        open_delay, close_delay = FOREIGN_OPEN_DELAY, timedelta(minutes=PUBLICATION_BUFFER_MINUTES)
    local_day = pd.Timestamp(asof.astimezone(ZoneInfo(zone)).date())
    later = prices.index > local_day
    today = prices.index == local_day
    for column, time_of_day, delay in [('Open', open_time, open_delay), ('Close', close_time, close_delay)]:
        same_day_is_preliminary = local_datetime(local_day, time_of_day, zone) + delay > asof
        prices.loc[later | (today & same_day_is_preliminary), column] = float('nan')
    return prices.dropna(how='all')


def prices_from_entry(entry):
    rows = entry.get('prices', [])
    dates = pd.DatetimeIndex([row['date'] for row in rows])
    return pd.DataFrame([(row['open'], row['close']) for row in rows], index=dates, columns=['Open', 'Close'], dtype=float)


def load_prices(root):
    path = root / 'prices.json'
    return read_json(path) if path.is_file() else {'markets': {}}


def load_state(root):
    path = root / 'state.json'
    return read_json(path) if path.is_file() else {'predictions': []}


def refresh_prices(root, cache, keys, asof, csv_dir: Path | None, *, full=False):
    keys = tuple(keys)
    if not keys:
        return
    asof = asof.astimezone(CAIRO)

    def update(key):
        old = cache['markets'].get(key, {})
        stored = prices_from_entry(old)
        long_history = (full or len(stored) < 300 or (asof.date() - stored.index[-1].date()).days > 30)
        prices, source = fetch_daily_prices(key, asof, csv_dir, 1350 if long_history else 45)
        prices = known_prices(key, prices, asof).groupby(level=0).last()
        if prices['Open'].notna().sum() == 0:
            raise ValueError(f'{source} returned no opening prices')
        if old.get('source') == source:
            prices = prices.combine_first(stored)
        prices = prices.sort_index().tail(920)
        rows = [{'date': day.date().isoformat(),
                 'open': float(row.Open) if pd.notna(row.Open) else None,
                 'close': float(row.Close) if pd.notna(row.Close) else None}
                for day, row in prices.iterrows()]
        return {'source': source, 'observed_at_cairo': asof.isoformat(), 'prices': rows}

    changed = False
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {key: pool.submit(update, key) for key in keys}
        for key, future in pending.items():
            try:
                cache['markets'][key] = future.result()
                changed = True
            except (ValueError, RuntimeError, OSError) as exc:
                print(f'{key} opening data unavailable: {exc}. Existing observations were kept.')
    if changed:
        dump_atomic(root / 'prices.json', cache)


def opening_gaps(prices):
    prior = prices['Close'].shift()
    intervals = prices.index.to_series().diff().dt.days
    return (prices['Open'] / prior - 1).where(intervals <= 14).dropna()


def gaps_for_cache(cache, asof):
    gaps = {}
    for key, entry in cache['markets'].items():
        values = opening_gaps(known_prices(key, prices_from_entry(entry), asof))
        if key != 'EGX30':
            values = values.loc[[key in scheduled_opening_markets(day) for day in values.index]]
        gaps[key] = values
    return gaps


def calendar_window(history, before):
    start = before - timedelta(days=HISTORY_DAYS)
    return history.loc[(history.index >= start) & (history.index < before)]


def align_history(gaps):
    egx = gaps.get('EGX30', pd.Series(index=pd.DatetimeIndex([]), dtype=float))
    frame = egx.rename('EGX30').to_frame()
    for key in EARLY_MARKETS:
        frame[key] = gaps.get(key, pd.Series(dtype=float)).reindex(frame.index)
    return frame


def group_probabilities(probabilities):
    return pd.DataFrame({group[0]: probabilities.reindex(columns=group).mean(axis=1) for group in OPENING_GROUPS},
                        index=probabilities.index)


def predict_from_opens(history, observed):
    conditions = {}
    for row in individual_counts(history):
        key = row['key']
        if key not in observed:
            continue
        up = observed[key] > 0
        support = row['foreign_opened_up'] if up else row['foreign_opened_non_up']
        hits = row['both_opened_up'] if up else row['egypt_up_when_foreign_non_up']
        if support >= MIN_CONDITIONAL_SESSIONS:
            conditions[key] = {'condition': 'UP' if up else 'NON_UP', 'sessions': support, 'egypt_up': hits,
                               'raw_rate': hits / support, 'probability': (hits + 1) / (support + 2)}
    if not conditions:
        return None
    grouped = group_probabilities(pd.DataFrame([{key: row['probability'] for key, row in conditions.items()}])).iloc[0].dropna()
    return {'prob_up': float(grouped.mean()), 'groups': len(grouped), 'market_conditions': conditions}


def walk_forward(history, start, end):
    limits = f'{HISTORY_DAYS}D'
    actual = history['EGX30'] > 0

    def previous_count(frame):
        return frame.astype(int).rolling(limits, closed='left').sum()

    observed = history[list(EARLY_MARKETS)]
    up = observed.gt(0)
    non_up = observed.le(0)
    matches = previous_count(up).where(up, previous_count(non_up)).where(observed.notna())
    hits = previous_count(up.mul(actual, axis=0)).where(up, previous_count(non_up.mul(actual, axis=0)))
    per_market = ((hits + 1) / (matches + 2)).where(matches >= MIN_CONDITIONAL_SESSIONS)
    probabilities = group_probabilities(per_market).mean(axis=1)
    eligible = probabilities.notna() & (history.index >= start) & (history.index < end)
    p = probabilities.loc[eligible]
    result = actual.loc[eligible]
    baseline = ((previous_count(actual) + 1) / (previous_count(actual.notna()) + 2)).loc[eligible]
    return {'evaluated_sessions': len(p), 'accuracy': float(((p >= .5) == result).mean()) if len(p) else None,
            'baseline_accuracy': float(((baseline >= .5) == result).mean()) if len(p) else None,
            'brier': float(((p - result) ** 2).mean()) if len(p) else None,
            'baseline_brier': float(((baseline - result) ** 2).mean()) if len(p) else None}


def individual_counts(history):
    rows = []
    for key, label in MARKET_NAMES.items():
        paired = history[['EGX30', key]].dropna()
        foreign_up = paired[key] > 0
        egypt_up = paired['EGX30'] > 0
        total, up, joint = len(paired), int(foreign_up.sum()), int((foreign_up & egypt_up).sum())
        non_up_hits = int((~foreign_up & egypt_up).sum())
        lo, hi = wilson_interval(joint, up) if up else (None, None)
        rows.append({'key': key, 'index': label, 'paired_sessions': total,
                     'foreign_opened_up': up, 'both_opened_up': joint,
                     'foreign_opened_non_up': total - up, 'egypt_up_when_foreign_non_up': non_up_hits,
                     'egypt_up_given_foreign_up': joint / up if up else None,
                     'egypt_up_given_foreign_non_up': non_up_hits / (total - up) if total > up else None,
                     'egypt_up_on_paired_days': int(egypt_up.sum()) / total if total else None,
                     'conditional_ci_low': lo, 'conditional_ci_high': hi})
    return rows


def todays_foreign_gaps(gaps, day):
    return {key: float(gaps[key].loc[day]) for key in scheduled_opening_markets(day)
            if key in gaps and day in gaps[key].index}


def direction_of(gap):
    return 'UP' if gap > 0 else 'DOWN' if gap < 0 else 'FLAT'


def publish(state, gaps, asof, publication_clock):
    day = pd.Timestamp(asof.date())
    identifier = day.date().isoformat()
    if any(row['date'] == identifier for row in state['predictions']):
        return False
    observed = todays_foreign_gaps(gaps, day)
    if observed:
        print('Early opens ' + identifier + ': ' + ', '.join(f'{MARKET_NAMES[key]} {gap:+.2%}' for key, gap in observed.items()))
    estimate = predict_from_opens(calendar_window(align_history(gaps), day), observed)
    if estimate is None:
        print(f'{identifier} OPENING: no signal with adequate early markets and prior comparable sessions.')
        return False
    published_at = publication_clock().astimezone(CAIRO)
    if published_at < asof or published_at.date() != day.date() or published_at.weekday() in (4, 5) \
            or not FORECAST_CUTOFF <= published_at.time() < EGX_OPEN:
        return False
    probability = estimate['prob_up']
    direction = 'UP' if probability >= .5 else 'NON_UP'
    state['predictions'].append({'date': identifier, 'status': 'PENDING', 'published_at_cairo': published_at.isoformat(),
                                 'direction': direction, **estimate, 'rule': '600_CALENDAR_DAYS_CONDITIONAL',
                                 'history_start': (day - timedelta(days=HISTORY_DAYS)).date().isoformat(),
                                 'market_open_gaps': observed})
    print(f'{identifier} OPENING: {direction}, P(EGX opens up) {probability:.1%}. '
          f"Conditional history from {estimate['groups']} early market groups.")
    return True


def settle(state, gaps, asof):
    egypt = gaps.get('EGX30')
    if egypt is None or egypt.empty:
        return False
    changed = False
    for row in state['predictions']:
        if row['status'] != 'PENDING':
            continue
        day = pd.Timestamp(row['date'])
        if local_datetime(day, OPENING_RESULT.strftime('%H:%M'), 'Africa/Cairo') > asof:
            continue
        if day in egypt.index:
            gap = float(egypt.loc[day])
            direction = direction_of(gap)
            row.update(status='SETTLED', actual_open_gap=gap, actual_direction=direction,
                       correct=(row['direction'] == 'UP') == (gap > 0), settled_at_cairo=asof.isoformat())
            print(f"{row['date']} OPENING result: {direction} ({gap:+.2%}).")
            changed = True
        elif (egypt.index > day).sum() >= 2 and (egypt.index < day).any():
            row.update(status='VOID', void_reason='No verified opening price for this date',
                       settled_at_cairo=asof.isoformat())
            changed = True
    return changed


def write_reports(root, cache, state, gaps, asof):
    full = align_history(gaps)
    day = pd.Timestamp(asof.date())
    end = day + timedelta(days=1)
    start = end - timedelta(days=HISTORY_DAYS)
    history = calendar_window(full, end)
    counts = individual_counts(history)
    backtest = walk_forward(full, start, end)
    settled = [p for p in state['predictions'] if p['status'] == 'SETTLED']
    latest = history.index[-1].date().isoformat() if len(history) else None
    last_observed = todays_foreign_gaps(gaps, day)
    estimate = predict_from_opens(calendar_window(full, day), last_observed)
    report = {'report_version': REPORT_VERSION, 'updated_at_cairo': asof.isoformat(), 'lookback_calendar_days': HISTORY_DAYS,
              'window_start': start.date().isoformat(), 'window_end': day.date().isoformat(),
              'first_egx_open_session': history.index[0].date().isoformat() if len(history) else None,
              'last_egx_open_session': latest, 'egx_open_sessions': len(history),
              'last_egx_open_gap': float(history.iloc[-1]['EGX30']) if len(history) else None,
              'market_counts': counts, 'historical_walk_forward': backtest, 'estimate_from_prior_days': estimate,
              'today_early_market_gaps': last_observed,
              'sources': {key: entry['source'] for key, entry in cache['markets'].items()},
              'live': {'evaluated': len(settled), 'correct': sum(row['correct'] for row in settled),
                       'pending': sum(row['status'] == 'PENDING' for row in state['predictions']),
                       'void': sum(row['status'] == 'VOID' for row in state['predictions'])}}
    dump_atomic(root / 'report.json', report)
    try:
        with atomic_text(root / 'conditional_counts.csv', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(counts[0]))
            writer.writeheader()
            writer.writerows(counts)
        with atomic_text(root / 'predictions.csv', encoding='utf-8-sig') as handle:
            fields = ['date', 'status', 'direction', 'prob_up', 'rule', 'historical_egx_up', 'historical_sessions',
                      'history_start', 'groups', 'published_at_cairo', 'actual_open_gap', 'actual_direction', 'correct',
                      'settled_at_cairo', 'void_reason']
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(state['predictions'])
    except PermissionError:
        print('Opening CSV report is in use. Close it before the next update.')


def print_report(root):
    path = root / 'report.json'
    if not path.is_file():
        print('OPENING: waiting for first data collection via python app.py start.')
        return
    report = read_json(path)
    print(f"OPENING: {report['egx_open_sessions']} EGX openings from {report.get('window_start', report['first_egx_open_session'])} "
          f"through {report.get('window_end', report['last_egx_open_session'])}.")
    observed = report.get('today_early_market_gaps', {})
    if observed:
        print(f"Observed early opens on {report['updated_at_cairo'][:10]}:")
        for key, gap in observed.items():
            print(f'  {MARKET_NAMES[key]:<23} {direction_of(gap):<4} {gap:+.2%}')
    estimate = report.get('estimate_from_prior_days')
    if estimate:
        print(f"Estimated EGX opening-up probability from earlier days: {estimate['prob_up']:.1%} "
              f"({estimate['groups']} groups). See the trial file for published forecasts.")
    print('Early index                EGX opened up / index opened up    Probability    Matched sessions')
    for row in report['market_counts']:
        rate = row['egypt_up_given_foreign_up']
        printable = f'{rate:.1%}' if rate is not None else 'n/a'
        print(f"{row['index']:<25} {row['both_opened_up']:>8}/{row['foreign_opened_up']:<21} {printable:>8}    {row['paired_sessions']:>12}")
    live = report['live']
    score = report['historical_walk_forward']
    if score['evaluated_sessions']:
        print(f"Opening walk-forward: {score['accuracy']:.1%} vs {score['baseline_accuracy']:.1%} baseline "
              f"across {score['evaluated_sessions']} sessions.")
    print(f"OPENING live: {live['correct']}/{live['evaluated']} correct, {live['pending']} pending.")
