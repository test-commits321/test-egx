"""Latest verified moves across the full market roster at the Cairo cutoff."""
from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from opening import FOREIGN_OPEN_TIMES, MARKET_NAMES
from predictor import CAIRO, FORECAST_CUTOFF, MARKETS, PUBLICATION_BUFFER_MINUTES, local_datetime
from runtime_io import atomic_text, dump_atomic, read_json
from tradingview_feed import tradingview_bars

MARKET_BY_KEY = {market.key: market for market in MARKETS}
OPEN_TIMES = {**FOREIGN_OPEN_TIMES,
              'NASDAQ': '09:30', 'NYSE': '09:30', 'DOW': '09:30', 'CANADA': '09:30',
              'FTSE': '08:00', 'GERMANY': '09:00', 'IBEX': '09:00', 'EURONEXT': '09:00', 'SWITZERLAND': '09:00'}
# The four North American cash indices open after the Cairo cutoff throughout the year.
TV_FEEDS = {
    'JAPAN': ('TVC', 'NI225', 60), 'KOREA': ('KRX', 'KOSPI', 60), 'TAIWAN': ('TWSE', 'IX0001', 60),
    'SHANGHAI': ('SSE', '000001', 60), 'SHENZHEN': ('SZSE', '399001', 60), 'HONG_KONG': ('HSI', 'HSI', 60),
    'BSE': ('BSE', 'SENSEX', 60), 'NSE': ('NSE', 'NIFTY', 60), 'SAUDI': ('TADAWUL', 'TASI', 30),
    'FTSE': ('FTSE', 'UKX', 30), 'IBEX': ('BME', 'IBC', 30), 'GERMANY': ('XETR', 'DAX', 30),
    'EURONEXT': ('EURONEXT', 'PX1', 30), 'SWITZERLAND': ('SIX', 'SMI', 30),
}
BAR_DELAY_MINUTES = 20
MOVE_SUFFIX = '_LATEST_RETURN'
RISING_SHARE = 'GLOBAL_RISING_SHARE'
MEAN_MOVE = 'GLOBAL_MEAN_RETURN'
MAX_CLOSED_AGE = timedelta(days=8)


def cutoff_on(day):
    return local_datetime(day, FORECAST_CUTOFF.strftime('%H:%M'), 'Africa/Cairo')


def expected_same_day(key, day):
    market = MARKET_BY_KEY[key]
    weekday = pd.Timestamp(day).weekday()
    scheduled = (6, 0, 1, 2, 3) if key == 'SAUDI' else range(5)
    opened = local_datetime(day, OPEN_TIMES[key], market.timezone)
    return weekday in scheduled and opened < cutoff_on(day)


def read_csv_prices(path):
    frame = pd.read_csv(path, encoding='utf-8-sig')
    columns = {str(name).strip().lower(): name for name in frame}
    if not {'timestamp', 'close'} <= set(columns):
        raise ValueError(f'{path.name}: use Timestamp,Close for bar-start timestamps')
    stamps = frame[columns['timestamp']].astype('string').str.strip()
    if not stamps.str.contains(r'(?:Z|[+-]\d{2}:?\d{2})$', case=False, na=False).all():
        raise ValueError(f'{path.name}: Timestamp must have a timezone, e.g. 2026-11-23T06:00:00Z')
    dates = pd.DatetimeIndex(pd.to_datetime(stamps, errors='raise', utc=True, format='mixed'))
    return pd.Series(frame[columns['close']].to_numpy(), index=dates)


def yahoo_hourly(market, asof, full):
    # Public Yahoo chart history is limited to recent dates for hourly bars.
    days = 727 if full else 7
    params = urlencode({'period1': int((asof - timedelta(days=days)).timestamp()),
                        'period2': int(asof.timestamp()), 'interval': '1h', 'includePrePost': 'false'})
    failures = []
    for host in ('query1.finance.yahoo.com', 'query2.finance.yahoo.com'):
        request = Request(f'https://{host}/v8/finance/chart/{quote(market.ticker, safe="")}?{params}',
                          headers={'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'})
        try:
            with urlopen(request, timeout=8) as response:
                result = json.load(response)['chart']['result'][0]
            meta = result['meta']
            if meta.get('symbol', '').casefold() != market.ticker.casefold() or meta.get('dataGranularity') not in ('1h', '60m'):
                raise ValueError('mismatched index or interval')
            times = pd.to_datetime(result['timestamp'], unit='s', utc=True)
            prices = pd.Series(result['indicators']['quote'][0]['close'], index=times)
            if prices.dropna().empty:
                raise ValueError('no usable hourly prices')
            return prices
        except (OSError, ValueError, TypeError, AttributeError, KeyError, IndexError) as exc:
            failures.append(f'{host}: {exc}')
    raise RuntimeError('; '.join(failures))


def fetch_bars(key, asof, csv_dir, full):
    market = MARKET_BY_KEY[key]
    path = csv_dir / f'{key}_INTRADAY.csv' if csv_dir else None
    if path and path.is_file():
        return read_csv_prices(path), path.name
    exchange, symbol, minutes = TV_FEEDS[key]
    interval = 'in_1_hour' if minutes == 60 else 'in_30_minute'
    first_error = None
    for _ in range(2 if full else 1):
        try:
            bars = tradingview_bars(symbol, exchange, market.timezone, 5000 if full else 230, interval)
            return bars['Close'], f'TradingView {exchange}:{symbol}'
        except (RuntimeError, ValueError, OSError) as exc:
            first_error = str(exc)
    if minutes == 60:
        try:
            return yahoo_hourly(market, asof, full), f'Yahoo {market.ticker}'
        except (RuntimeError, OSError, ValueError) as exc:
            raise RuntimeError(f'{first_error}; hourly fallback: {exc}') from exc
    raise RuntimeError(first_error)


def latest_finished_bars(key, bars, asof):
    market = MARKET_BY_KEY[key]
    interval = TV_FEEDS[key][2]
    index = pd.DatetimeIndex(bars.index)
    if index.tz is None:
        raise ValueError(f'{key}: bar timestamps must include their timezone')
    local = index.tz_convert(market.timezone)
    cairo_days = index.tz_convert(CAIRO).tz_localize(None).normalize()
    local_days = local.tz_localize(None).normalize()
    ready = index + timedelta(minutes=interval + BAR_DELAY_MINUTES)
    frame = pd.DataFrame({'date': cairo_days, 'local_day': local_days, 'local_hour': local.strftime('%H:%M'),
                          'bar_end': index + timedelta(minutes=interval), 'ready': ready,
                          'price': pd.to_numeric(bars, errors='coerce').to_numpy()})
    frame = frame.loc[(frame['date'] == frame['local_day']) & (frame['local_hour'] >= OPEN_TIMES[key])
                      & (frame['local_hour'] < market.close) & (frame['ready'] <= asof)
                      & np.isfinite(frame['price']) & (frame['price'] > 0)]
    if frame.empty:
        return {}
    dates = frame['date'].drop_duplicates()
    cutoffs = {day: cutoff_on(day) for day in dates}
    sessions = {day: expected_same_day(key, day) for day in dates}
    # A bar qualifies only after its complete interval and publication buffer. A
    # shortened final bar can instead enter as an independently finalized close.
    frame = frame.loc[(frame['ready'] <= frame['date'].map(cutoffs))
                      & frame['date'].map(sessions)]
    latest = frame.sort_values('ready', kind='stable').drop_duplicates('date', keep='last')
    return {row.date.date().isoformat(): {'price': float(row.price),
                                         'price_at_cairo': row.bar_end.tz_convert(CAIRO).isoformat(),
                                         'available_at_cairo': row.ready.tz_convert(CAIRO).isoformat()}
            for row in latest.itertuples(index=False)}


def load_book(root):
    path = root / 'observations.json'
    return read_json(path) if path.is_file() else {'markets': {}}


def refresh_book(root, book, asof, csv_dir, keys, *, full=False, catch_up=False):
    today = pd.Timestamp(asof.date())
    day_key = today.date().isoformat()

    def needed(key):
        # Manual closes should only be paired with explicit same-scale manual bars.
        manual_bars = csv_dir and (csv_dir / f'{key}_INTRADAY.csv').is_file()
        if csv_dir and MARKET_BY_KEY[key].csv_override(csv_dir) and not manual_bars:
            return False
        return full or catch_up or expected_same_day(key, today)

    def get(key):
        old = book['markets'].get(key, {})
        rows = old.get('rows', {})
        if not full and not catch_up and day_key in rows and asof.time() >= FORECAST_CUTOFF:
            # Reuse a snapshot already obtained at or after the fixed cutoff.
            recorded = pd.Timestamp(old['checked_at_cairo'])
            if recorded >= cutoff_on(today):
                return None
        prices, source = fetch_bars(key, asof, csv_dir, full)
        fresh = latest_finished_bars(key, prices, asof)
        if not fresh:
            return None
        for row in fresh.values():
            row['source'] = source
        merged = {**rows}
        for day, row in fresh.items():
            if day not in merged or row['available_at_cairo'] >= merged[day]['available_at_cairo']:
                merged[day] = row
        return {'checked_at_cairo': asof.isoformat(), 'rows': dict(sorted(merged.items())[-950:])}

    changed = False
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {key: pool.submit(get, key) for key in keys if key in TV_FEEDS and needed(key)}
        for key, future in futures.items():
            try:
                entry = future.result()
                if entry is not None:
                    book['markets'][key] = entry
                    changed = True
            except (ValueError, RuntimeError, OSError, KeyError) as exc:
                print(f'{key} same-day feed unavailable: {exc}.')
    if changed:
        dump_atomic(root / 'observations.json', book)


def market_move(key, day, closes, observations, asof, cutoff=None):
    if cutoff is None:
        cutoff = min(asof, cutoff_on(day))
    market = MARKET_BY_KEY[key]
    identifier = day.date().isoformat()
    result = {'market': key, 'index': MARKET_NAMES.get(key, market.exchange), 'date_cairo': identifier,
              'status': 'OPENS_LATER' if day.weekday() in range(5) else 'NO_REGULAR_SESSION'}
    if not expected_same_day(key, day):
        if key == 'SAUDI' and day.weekday() == 6:
            result['status'] = 'OPENS_LATER'
        return result
    result['status'] = 'NO_VERIFIED_PRICE'
    if closes is None or closes.empty:
        return result
    pos = closes.index.searchsorted(day)
    if not pos or (day - closes.index[pos - 1]).days > 14:
        result['status'] = 'NO_PREVIOUS_CLOSE'
        return result
    previous_close = float(closes.iloc[pos - 1])
    final_at = local_datetime(day, market.close, market.timezone) + timedelta(minutes=PUBLICATION_BUFFER_MINUTES)
    if pos < len(closes) and closes.index[pos] == day and final_at <= cutoff:
        price_at = final_at - timedelta(minutes=PUBLICATION_BUFFER_MINUTES)
        observation = {'price': float(closes.iloc[pos]), 'price_at_cairo': price_at.tz_convert(CAIRO).isoformat(),
                       'available_at_cairo': final_at.tz_convert(CAIRO).isoformat(), 'source': 'Final same-session close'}
    else:
        observation = observations.get(identifier)
        if not observation or pd.Timestamp(observation['available_at_cairo']) > cutoff:
            return result
    price = observation['price']
    if not .5 <= price / previous_close <= 1.5:
        result['status'] = 'PRICE_SCALE_MISMATCH'
        return result
    return {**result, **observation, 'status': 'AVAILABLE', 'movement_basis': 'SAME_SESSION',
            'price_session': identifier, 'previous_close': previous_close,
            'previous_session': closes.index[pos - 1].date().isoformat(), 'return': price / previous_close - 1}


def latest_closed_move(key, closes, cutoff):
    if closes is None or len(closes) < 2:
        return None
    market = MARKET_BY_KEY[key]
    local_day = pd.Timestamp(cutoff).tz_convert(market.timezone).date()
    index = closes.index
    pos = index.searchsorted(pd.Timestamp(local_day), side='right') - 1
    if pos < 1:
        return None
    delay = timedelta(minutes=PUBLICATION_BUFFER_MINUTES)
    closing = local_datetime(index[pos], market.close, market.timezone)
    if closing + delay > cutoff:
        pos -= 1
        if pos < 1:
            return None
        closing = local_datetime(index[pos], market.close, market.timezone)
    available = closing + delay
    if cutoff - available > MAX_CLOSED_AGE or (index[pos] - index[pos - 1]).days > 14:
        return None
    price, previous = float(closes.iloc[pos]), float(closes.iloc[pos - 1])
    if not np.isfinite(price) or not np.isfinite(previous) or price <= 0 or previous <= 0 or not .5 <= price / previous <= 1.5:
        return None
    return {'price': price, 'previous_close': previous, 'return': price / previous - 1,
            'price_at_cairo': closing.tz_convert(CAIRO).isoformat(), 'available_at_cairo': available.tz_convert(CAIRO).isoformat(),
            'price_session': index[pos].date().isoformat(), 'previous_session': index[pos - 1].date().isoformat(),
            'source': 'Verified daily closes', 'status': 'AVAILABLE', 'movement_basis': 'LAST_CLOSED_SESSION'}


def snapshot(book, closes, day, asof, keys):
    cutoff = min(asof, cutoff_on(day))
    rows = []
    for key in keys:
        daily = closes.get(key)
        observed = market_move(key, day, daily, book['markets'].get(key, {}).get('rows', {}), asof, cutoff)
        if observed['status'] != 'AVAILABLE':
            last_session = latest_closed_move(key, daily, cutoff)
            observed['same_day_status'] = observed['status']
            if last_session is not None:
                observed.update(last_session)
        rows.append(observed)
    changes = {row['market']: row['return'] for row in rows if row['status'] == 'AVAILABLE'}
    features = {key + MOVE_SUFFIX: changes.get(key, np.nan) for key in keys}
    complete = len(changes) == len(features) and bool(changes)
    features[RISING_SHARE] = sum(value > 0 for value in changes.values()) / len(changes) if complete else np.nan
    features[MEAN_MOVE] = float(np.mean(tuple(changes.values()))) if complete else np.nan
    return features, rows


def history_features(book, closes, days, asof):
    return pd.DataFrame((snapshot(book, closes, day, asof, closes)[0] for day in days), index=days)


def verified_days(features):
    return features[RISING_SHARE].notna()


def save_snapshot(root, features, rows, asof):
    moves = [row for row in rows if row['status'] == 'AVAILABLE']
    counts = {name: sum(compare(row['return']) for row in moves)
              for name, compare in [('up', lambda value: value > 0), ('down', lambda value: value < 0), ('flat', lambda value: value == 0)]}
    same_session = sum(row['movement_basis'] == 'SAME_SESSION' for row in moves)
    dump_atomic(root / 'market_moves.json', {'recorded_at_cairo': asof.isoformat(), **counts,
                'same_session': same_session, 'last_closed_session': len(moves) - same_session,
                'rising_share': features[RISING_SHARE] if len(moves) == len(rows) and moves else None, 'markets': rows})
    fields = ['date_cairo', 'market', 'index', 'status', 'return_percent', 'price', 'previous_close',
              'movement_basis', 'price_session', 'same_day_status', 'previous_session', 'price_at_cairo', 'available_at_cairo', 'source']
    try:
        with atomic_text(root / 'market_moves.csv', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, 'return_percent': round(row['return'] * 100, 6) if 'return' in row else None})
    except PermissionError:
        print('Close market_moves.csv before the next report update.')


def print_snapshot(root):
    path = root / 'market_moves.json'
    if not path.is_file():
        print('GLOBAL MOVES: waiting for the first snapshot via python app.py start.')
        return
    report = read_json(path)
    rising = report['rising_share']
    print(f"Full market snapshot saved at {report['recorded_at_cairo']}, using the 09:50 Cairo cutoff.")
    if rising is not None:
        print(f"All {len(report['markets'])} indices: {report['up']} up, {report['down']} down, {report['flat']} flat; "
              f"{report['same_session']} same-session readings and {report['last_closed_session']} last completed sessions.")
    for row in report['markets']:
        rendered = (f"{row['return']:+.2%} ({row['price_session']} {row['movement_basis']})"
                    if 'return' in row else row['status'])
        print(f"  {row['index']:<26} {rendered}")
