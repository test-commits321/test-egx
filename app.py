#!/usr/bin/env python3
"""Train an EGX30 direction model and track its pre-open forecasts."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time as clock
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time, timedelta
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np
import pandas as pd

import intraday
import opening
from predictor import (
    ADDED_MARKET_KEYS, CAIRO, EGX_CLOSE, EGX_FEATURE_COLUMNS, EGX_OPEN, FORECAST_CUTOFF, MARKETS,
    MODEL_VERSION, PUBLICATION_BUFFER_MINUTES, SIGNAL_THRESHOLD, TRAINING_HALF_LIFE,
    assemble_market_features, clean_close, country_weights, egx_controls,
    evaluate, fit_direction_model, local_datetime, market_feature_columns, now_cairo, parse_close_csv,
    walk_forward, yahoo_close,
)
from runtime_io import atomic_text, dump_atomic, read_json, trial_lock
from tradingview_feed import tradingview_daily_prices

ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / 'data' / 'history_600.json'
DEFAULT_RUNTIME = ROOT / 'runtime'
DEFAULT_MARKET_CACHE = ROOT / 'data' / 'market_cache'
SCHEMA_VERSION = 2
EGX_SYMBOL = '^CASE30'
DURATION = 30  # Default trial: 30 CALENDAR days from the first actually issued forecast.
SETTLE_DELAY = 45  # Minutes after scheduled Cairo close.
LIVE_MAX_AGE = 8  # Calendar-day max age of an available foreign-market close.
EGX_MAX_AGE = 7
HISTORY_MAX_AGE = 14
HISTORICAL_REFIT_SESSIONS = 20
GLOBAL_MOVE_MODELS = {'frozen': 'FROZEN ALL-MARKET CLOSE', 'adaptive': 'ADAPTIVE ALL-MARKET CLOSE'}


def market_time_passed(day, timezone, close, asof, delay=SETTLE_DELAY):
    return local_datetime(day, close, timezone) + timedelta(minutes=delay) <= pd.Timestamp(asof)


def date_rows(series: pd.Series) -> list[dict]:
    return [{'date': x.date().isoformat(), 'close': float(y)} for x, y in series.items()]


def series_from_rows(rows: list, label: str) -> pd.Series:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{label}: no daily prices in JSON')
    for row in rows:
        if not isinstance(row, dict) or not {'date', 'close'} <= row.keys():
            raise ValueError(f'{label}: each row must contain date and close')
    dates = pd.to_datetime([x['date'] for x in rows], format='%Y-%m-%d', errors='raise')
    values = [x['close'] for x in rows]
    s = clean_close(pd.Series(values, index=pd.DatetimeIndex(dates), name='Close'))
    if len(s) != len(rows):
        raise ValueError(f'{label}: duplicate dates, null, or invalid price')
    if len(s) >= 2 and any((s.index.to_series().diff().dt.days.dropna() > 20)):
        raise ValueError(f'{label}: gap longer than 20 days (check data)')
    return s


def get_prices(ticker: str, key: str, from_date: datetime, asof: datetime, csv_dir: Path | None,
               direct_csv: Path | None = None) -> pd.Series:
    local = direct_csv if direct_csv and direct_csv.exists() else csv_dir / f'{key}.csv' if csv_dir and (csv_dir / f'{key}.csv').exists() else None
    return parse_close_csv(local) if local else yahoo_close(ticker, from_date, asof + timedelta(days=2))


def egx_exchange_close(lookback_days: int, asof: datetime) -> pd.Series:
    """Read daily EGX30 closes from the exchange's public history page endpoint."""
    end = pd.Timestamp(asof).tz_convert('Africa/Cairo').date()
    start = end - timedelta(days=lookback_days)
    page_size = 1000
    rows = []
    for page in range(1, lookback_days // page_size + 2):
        request = Request(
            'https://beta.egx.com.eg/api/bff/egx/index-since-inception?indexName=CASE30',
            data=json.dumps({'fromDate': start.isoformat(), 'toDate': end.isoformat(),
                             'pageNumber': page, 'pageSize': page_size}).encode('utf-8'),
            headers={'Accept': 'application/json', 'Content-Type': 'application/json',
                     'x-egx-bff-request': '1'},
        )
        with urlopen(request, timeout=8) as response:
            payload = json.load(response)
        items = payload.get('data') if isinstance(payload, dict) and payload.get('success') is True else None
        if not isinstance(items, list):
            raise ValueError('EGX website returned an invalid history response')
        rows.extend(items)
        total = payload.get('totalCount')
        if len(items) < page_size or (isinstance(total, int) and len(rows) >= total):
            break
    if not rows:
        raise ValueError('EGX website returned no EGX30 observations')
    df = pd.DataFrame(rows)
    if not {'indexDay', 'indexClose'}.issubset(df.columns):
        raise ValueError(f'EGX website returned unrecognized fields: {list(df.columns)}')
    # Keep the calendar day of the exchange's session-date label.
    dates = df['indexDay'].astype(str).str.extract(r'^(\d{4}-\d{2}-\d{2})', expand=False)
    idx = pd.DatetimeIndex(pd.to_datetime(dates, errors='coerce', format='%Y-%m-%d'))
    values = pd.to_numeric(df['indexClose'], errors='coerce')
    return clean_close(pd.Series(values.to_numpy(), index=idx, name='Close'))


def usable_prices(prices, label, timezone, close, asof, delay, min_sessions,
                  max_age_days=None):
    prices = clean_close(prices)
    ready = prices.loc[[market_time_passed(day, timezone, close, asof, delay)
                        for day in prices.index]]
    if len(ready) < min_sessions:
        raise ValueError(f'{label}: {len(ready)} completed sessions (require {min_sessions})')
    latest = ready.index[-1].date()
    local_day = pd.Timestamp(asof).tz_convert(timezone).date()
    if max_age_days is not None and (local_day - latest).days > max_age_days:
        raise ValueError(f'{label}: last close {latest} is older than {max_age_days} days')
    return ready


def fetch_egx(asof: datetime, csv_dir: Path | None, egx_csv: Path | None, lookback_days=1300,
              min_sessions=1, max_age_days=None) -> pd.Series:
    """Prefer an explicitly supplied CSV, then the EGX chart endpoint, then Yahoo.

    Return only published, fully closed market sessions. A single Yahoo daily
    quote cannot be mistaken for 600 historical sessions.
    """
    if egx_csv is not None and not egx_csv.is_file():
        raise ValueError(f'EGX30: --egx-csv file does not exist: {egx_csv}')
    local = egx_csv or (csv_dir / 'EGX30.csv' if csv_dir and (csv_dir / 'EGX30.csv').is_file() else None)

    def verified(prices, source):
        ready = usable_prices(prices, f'EGX30 {source}', 'Africa/Cairo',
                              EGX_CLOSE.strftime('%H:%M'), asof, SETTLE_DELAY,
                              min_sessions, max_age_days)
        print(f'EGX30: {source} supplied {len(ready)} completed sessions '
              f'(latest {ready.index[-1].date()})')
        return ready

    if local is not None:
        return verified(parse_close_csv(local, timezone='Africa/Cairo'), f'CSV {local}')

    try:
        return verified(egx_exchange_close(lookback_days, asof), 'exchange website')
    except Exception as exc:
        exchange_error = str(exc)
    try:
        return verified(yahoo_close(EGX_SYMBOL, asof - timedelta(days=lookback_days),
                                    asof + timedelta(days=2)), 'Yahoo')
    except Exception as exc:
        raise RuntimeError('EGX30 history unavailable from either provider. '
                           f'EGX website: {exchange_error}; Yahoo: {exc}. '
                           'Supply --egx-csv data/EGX30.csv with genuine Date,Close data.') from exc


def tasi_api_close(start: datetime, end: datetime) -> pd.Series:
    """Fetch actual final Saudi TASI daily closes using an optional TASI_API_KEY.

    API permits <=366 calendar days per request; split the historical window.
    The exchange session date is determined in Riyadh time, never UTC date.
    """
    key = os.environ.get('TASI_API_KEY', '').strip()
    if not key:
        raise RuntimeError('TASI_API_KEY is not configured')
    import requests
    all_bars = []
    begin = pd.Timestamp(start.date())
    final = pd.Timestamp(end.date())
    while begin <= final:
        last = min(begin + timedelta(days=350), final)
        response = requests.get(
            'https://api.tasi-api.com/api/v1/history/TASI',
            headers={'X-API-Key': key},
            params={'interval': '1d', 'from': begin.date().isoformat(),
                    'to': last.date().isoformat(), 'limit': 5000},
            timeout=25,
        )
        response.raise_for_status()
        body = response.json()
        bars = body.get('data')
        if not isinstance(bars, list):
            raise ValueError('TASI API returned an invalid data array')
        # Provisional bars are not final session closing prices.
        all_bars.extend(b for b in bars if isinstance(b, dict)
                        and b.get('quality') == 'final')
        begin = last + timedelta(days=1)
    if not all_bars:
        raise ValueError('TASI API returned zero final closes')
    ts = pd.to_datetime([b.get('t') for b in all_bars], errors='coerce', utc=True)
    idx = pd.DatetimeIndex(ts).tz_convert('Asia/Riyadh').tz_localize(None).normalize()
    vals = [b.get('close') for b in all_bars]
    return clean_close(pd.Series(vals, index=idx, name='Close'))


def tradingview_tasi_close(n_bars: int = 750) -> pd.Series:
    result = clean_close(tradingview_daily_prices('TASI', 'TADAWUL', 'Asia/Riyadh', n_bars)['Close'])
    if result.empty:
        raise ValueError('TradingView provided no valid TASI closing prices')
    return result


def finalized_market_prices(prices: pd.Series, market, asof: datetime) -> pd.Series:
    """Never count today's provisional candle as a completed daily close."""
    return prices.loc[[market_time_passed(day, market.timezone, market.close,
                                           asof, PUBLICATION_BUFFER_MINUTES)
                       for day in prices.index]]


def cache_file_for(market, cache_dir: Path) -> Path:
    return cache_dir / f'{market.key}.json'


def read_market_cache(market, asof: datetime, cache_dir: Path, min_sessions: int = 2,
                      max_age_days: int | None = LIVE_MAX_AGE) -> pd.Series | None:
    """Use only genuine, finalized prices previously saved by this program.

    Stale, corrupted, foreign-symbol or provisional candles are rejected.
    A persistent price cache is separate from the frozen model and is not used
    to update its coefficients.
    """
    file = cache_file_for(market, cache_dir)
    if not file.is_file():
        return None
    try:
        payload = read_json(file)
        if not isinstance(payload, dict) or payload.get('cache_schema') != 1 or payload.get('market_key') != market.key \
                or payload.get('ticker') != market.ticker:
            raise ValueError('cache metadata does not match configured market')
        prices = series_from_rows(payload['data'], market.key)
        if len(prices) != payload.get('rows', len(payload['data'])):
            raise ValueError('incomplete cached observations')
        # Reject any candle whose closing timestamp is still in the future.
        if len(finalized_market_prices(prices, market, asof)) != len(prices):
            raise ValueError('cached data include uncompleted or future closes')
        if len(prices) < min_sessions:
            return None
        local_day = pd.Timestamp(asof).tz_convert(market.timezone).date()
        age = (local_day - prices.index[-1].date()).days
        if age < 0 or (max_age_days is not None and age > max_age_days):
            return None
        return prices
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(f'{market.key}: ignoring invalid cache {file}: {exc}')
        return None


def save_market_cache(market, prices: pd.Series, asof: datetime, cache_dir: Path,
                      source: str):
    """Store successfully fetched finalized closes; merge rolling updates without
    throwing away the full 600-session historical cache."""
    prices = finalized_market_prices(clean_close(prices), market, asof)
    if len(prices) < 2:
        return
    previous = read_market_cache(market, asof, cache_dir, min_sessions=1, max_age_days=None)
    if previous is not None:
        gap = max(prices.index[0] - previous.index[-1], previous.index[0] - prices.index[-1])
        if gap <= timedelta(days=20):
            prices = clean_close(pd.concat([previous, prices])).tail(1100)
    dump_atomic(cache_file_for(market, cache_dir), {
        'cache_schema': 1, 'market_key': market.key, 'ticker': market.ticker,
        'exchange': market.exchange, 'source_last_update': source,
        'saved_at_cairo': asof.isoformat(), 'rows': len(prices),
        'data': date_rows(prices),
    })


def fetch_market(market, asof, csv_dir: Path | None, lookback_days=1300, min_sessions=2,
                 cache_dir: Path | None = None, prefer_cache: bool = False,
                 max_age_days: int = LIVE_MAX_AGE, reuse_current_saudi=False):
    """Fetch real closed daily candles, with persistent, validated outage fallback.

    check-saudi performs a remote fetch and saves it; init reuses a fresh cache.
    Manual CSV is always preferred. For live sessions remote is tried first and
    cache is a fallback only. Nothing is inferred or forward-filled.
    """
    local = market.csv_override(csv_dir)
    start = asof - timedelta(days=lookback_days)

    def verified(prices, source):
        return usable_prices(prices, f'{market.exchange} {source}', market.timezone,
                             market.close, asof, PUBLICATION_BUFFER_MINUTES,
                             min_sessions, max_age_days)

    def save(prices, source):
        if cache_dir:
            try:
                save_market_cache(market, prices, asof, cache_dir, source)
            except OSError as exc:
                print(f'  {market.key}: could not update the price cache: {exc}')

    if local:
        prices = verified(parse_close_csv(local, timezone=market.timezone), f'CSV {local}')
        save(prices, 'manual_csv')
        return prices

    cached = read_market_cache(market, asof, cache_dir, min_sessions, max_age_days) if cache_dir else None
    saudi_cache_is_current = False
    if reuse_current_saudi and market.key == 'SAUDI' and cached is not None:
        local_day = pd.Timestamp(asof).tz_convert(market.timezone).date()
        if not market_time_passed(local_day, market.timezone, market.close, asof, PUBLICATION_BUFFER_MINUTES):
            local_day -= timedelta(days=1)
        while local_day.weekday() in (4, 5):
            local_day -= timedelta(days=1)
        saudi_cache_is_current = cached.index[-1].date() == local_day
    if cached is not None and (prefer_cache or saudi_cache_is_current):
        print(f'  {market.key}: using {len(cached)} locally cached real sessions '
              f'(latest {cached.index[-1].date()}, no network request)')
        return cached

    if market.key != 'SAUDI':
        try:
            prices = verified(yahoo_close(market.ticker, start, asof + timedelta(days=2)),
                              f'Yahoo {market.ticker}')
            if cached is not None and prices.index[-1] < cached.index[-1]:
                raise ValueError('Yahoo returned an older close than the local cache')
            save(prices, 'yahoo')
            return prices
        except Exception as exc:
            if cached is None:
                raise RuntimeError(f'{market.exchange}: Yahoo {market.ticker} failed ({exc}). '
                                   f'Supply {market.csv_name} through --csv-dir.') from exc
            print(f'  {market.key}: WARNING online feed unavailable ({exc}); using {len(cached)} '
                  f'fresh cached completed sessions (last {cached.index[-1].date()}).')
            return cached

    selection = os.environ.get('SAUDI_SOURCE', 'auto').strip().lower()
    choices = ('auto', 'tradingview', 'yahoo', 'tasi_api')
    if selection not in choices:
        raise ValueError(f'SAUDI_SOURCE must be one of {choices}; got {selection!r}')
    candidates = (['tradingview', 'yahoo', 'tasi_api'] if selection == 'auto' else [selection])
    errors = []
    for source in candidates:
        if source == 'tasi_api' and not os.environ.get('TASI_API_KEY', '').strip():
            errors.append('TASI API: no key configured')
            continue
        attempts = 3 if source == 'tradingview' else 1
        for attempt in range(1, attempts + 1):
            try:
                if source == 'tradingview':
                    raw = tradingview_tasi_close(max(min_sessions + 125, 130))
                elif source == 'yahoo':
                    raw = yahoo_close(market.ticker, start, asof + timedelta(days=2))
                else:
                    raw = tasi_api_close(start, asof)
                try:
                    ready = verified(raw, source)
                except ValueError as exc:
                    errors.append(str(exc))
                    break  # Reconnecting will not refresh or lengthen a valid response.
                if cached is not None and ready.index[-1] < cached.index[-1]:
                    errors.append(f'{source}: returned an older close than the local cache')
                    break
                save(ready, source)
                print(f'SAUDI: {source} supplied {len(ready)} closed daily sessions '
                      f'(latest {ready.index[-1].date()})' +
                      (f'; saved {cache_file_for(market, cache_dir)}' if cache_dir else ''))
                return ready
            except Exception as exc:
                errors.append(f'{source} attempt {attempt}/{attempts}: {type(exc).__name__}: {exc}')
                if attempt < attempts:
                    print(f'SAUDI: {source} connection attempt {attempt}/{attempts} failed ({exc}); retrying.')
                    clock.sleep(min(2.0 * attempt, 4.0))
    if cached is not None:
        print(f'SAUDI: WARNING all online attempts failed; using {len(cached)} '
              f'locally cached verified completed sessions (last {cached.index[-1].date()}).')
        return cached
    raise RuntimeError('SAUDI: every configured source failed and no fresh local cache is available. '
                       + '; '.join(errors) +
                       '. Run check-saudi again until it saves data/market_cache/SAUDI.json, '
                       'or use --csv-dir data\\csv with a genuine SAUDI.csv export.')


def build_history(history_path: Path, asof: datetime, csv_dir: Path | None,
                  egx_csv: Path | None, allow_missing_markets=False):
    """Collect genuine 600 daily closes per instrument; fail closed if a feed is incomplete."""
    print('Collecting 600 genuine closed-session data points for EGX30 and each foreign index...')
    egx = fetch_egx(asof, csv_dir, egx_csv, min_sessions=600, max_age_days=HISTORY_MAX_AGE).tail(600)
    markets = {}
    errors = []
    for m in MARKETS:
        try:
            p = fetch_market(m, asof, csv_dir, min_sessions=600,
                             cache_dir=DEFAULT_MARKET_CACHE, prefer_cache=True,
                             max_age_days=HISTORY_MAX_AGE)
            p = p.tail(600)
            markets[m.key] = {'ticker': m.ticker, 'exchange': m.exchange, 'country': m.country,
                              'timezone': m.timezone, 'close_local': m.close,
                              'data': date_rows(p)}
            print(f'  {m.key:<12} 600 records, last {p.index[-1].date()}')
        except Exception as e:
            errors.append(f'{m.key}: {e}')
            print(f'  {m.key:<12} ERROR: {e}')
    if errors:
        if not allow_missing_markets:
            raise RuntimeError(f'All {len(MARKETS)} foreign histories are required by default: '
                               + '; '.join(errors) +
                               '. Supply verified daily closes with --csv-dir, or explicitly use '
                               'init --allow-missing-markets (minimum 10 valid indices).')
        if len(markets) < 10:
            raise RuntimeError(f'Only {len(markets)} valid international markets. '
                               'Need at least 10 even in --allow-missing-markets mode. ' + '; '.join(errors))
        print('WARNING: Training will EXCLUDE unavailable markets, not fabricate them: ' + '; '.join(errors))
    payload = {'schema_version': SCHEMA_VERSION, 'generated_at_cairo': asof.isoformat(),
               'historical_prices_require_source_verification': True, 'reference_index': 'EGX30',
               'source_policy': 'Egyptian Exchange history (Yahoo fallback); Yahoo daily charts for foreign indices; '
                                'Saudi TradingView, Yahoo, or optional TASI API; verified local CSV overrides',
               'cutoff_cairo': FORECAST_CUTOFF.strftime('%H:%M'), 'egx30': {'ticker': EGX_SYMBOL, 'data': date_rows(egx)},
               'markets': markets, 'excluded_markets': [m.key for m in MARKETS if m.key not in markets]}
    dump_atomic(history_path, payload)
    print(f'Historical JSON saved: {history_path.resolve()}')
    return payload


def validate_history(payload: dict, allow_missing_markets=False, asof: datetime | None = None):
    if payload.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f'Unsupported history schema. Expected schema_version={SCHEMA_VERSION}')
    egx = series_from_rows(payload['egx30']['data'], 'EGX30')
    if len(egx) != 600:
        raise ValueError(f'EGX30: expected 600 genuine session closes, received {len(egx)}')
    if asof is not None and not market_time_passed(egx.index[-1], 'Africa/Cairo',
                                                   EGX_CLOSE.strftime('%H:%M'), asof):
        raise ValueError('EGX30: history includes a provisional or future daily close')
    closes = {}
    excluded = set(payload.get('excluded_markets', []))
    known = {m.key for m in MARKETS}
    included = set(payload['markets'])
    if not (excluded | included) <= known or excluded & included:
        raise ValueError('Historical JSON has contradictory or unrecognized excluded_markets')
    if known - (excluded | included) == ADDED_MARKET_KEYS:
        if not allow_missing_markets:
            raise ValueError('History predates Dow Jones, FTSE 100 and IBEX 35. '
                             'Set --history to a new file to collect them or use '
                             'init --allow-missing-markets to reuse the existing prices.')
        excluded |= ADDED_MARKET_KEYS
    if excluded and not allow_missing_markets:
        raise ValueError('JSON excludes markets: ' + ', '.join(sorted(excluded)) +
                         '; rerun init --allow-missing-markets only if this is intentional')
    for m in MARKETS:
        entry = payload['markets'].get(m.key)
        if entry is None:
            if m.key in excluded and allow_missing_markets:
                continue
            raise ValueError(f'{m.key}: missing data; explicit exclusion is required')
        if entry.get('ticker') != m.ticker:
            raise ValueError(f'{m.key}: mismatched ticker')
        p = series_from_rows(entry['data'], m.key)
        if len(p) != 600:
            raise ValueError(f'{m.key}: expected 600 session closes, received {len(p)}')
        if asof is not None and not market_time_passed(p.index[-1], m.timezone, m.close, asof,
                                                      PUBLICATION_BUFFER_MINUTES):
            raise ValueError(f'{m.key}: history includes a provisional or future daily close')
        closes[m.key] = p
    if len(closes) < 10:
        raise ValueError(f'Only {len(closes)} usable indices; minimum is 10')
    return egx, closes


def serialize_model(pipe, columns, weights_exchange, country, target, egx, backtest, created_at, min_markets):
    imp, scal, clf = [pipe.named_steps[x] for x in ('imputer', 'scaler', 'classifier')]
    model = {
        'schema_version': SCHEMA_VERSION, 'model_version': MODEL_VERSION,
        'trained_at_cairo': created_at.isoformat(),
        'trained_egx_first_date': str(egx.index[0].date()),
        'trained_egx_last_date': str(target.index[-1].date()),
        'egx_prices_in_history': int(len(egx)), 'egx_target_days': int(len(target)),
        'frozen': True, 'features': columns,
        'active_market_keys': [m.key for m in MARKETS if m.key in columns],
        'excluded_market_keys': [m.key for m in MARKETS if m.key not in columns],
        'imputer_medians': [float(x) for x in imp.statistics_],
        'scaler_mean': [float(x) for x in scal.mean_],
        'scaler_scale': [float(x) for x in scal.scale_],
        'coefficients': [float(x) for x in clf.coef_[0]],
        'intercept': float(clf.intercept_[0]),
        'market_weights': weights_exchange.to_dict(orient='records'),
        'country_weights': country.to_dict(orient='records'),
        'historical_walk_forward': backtest,
        'min_available_live_markets': min_markets,
        'foreign_delay_minutes': PUBLICATION_BUFFER_MINUTES,
        'signal_threshold': SIGNAL_THRESHOLD,
        'training_half_life_sessions': TRAINING_HALF_LIFE,
        'training_up_rate': float(target.mean()),
        'notes': 'Probabilities refer to an increase versus the previous EGX30 close. '
                 'Coefficient weights describe model associations. Live and historical results are measured separately.',
    }
    return model


def train_from_history(payload: dict, asof: datetime, allow_missing_markets=False, *,
                       refit_every=HISTORICAL_REFIT_SESSIONS, historical_scores=None, global_frame=None):
    egx, markets = validate_history(payload, allow_missing_markets=allow_missing_markets, asof=asof)
    stale = [key for key, prices in {'EGX30': egx, **markets}.items()
             if (asof.date() - prices.index[-1].date()).days > HISTORY_MAX_AGE]
    if stale:
        raise ValueError(f"Historical JSON is stale for {', '.join(stale)}. Run init --refresh-history.")
    days = egx.index[1:]
    X = assemble_market_features(days, markets, asof, PUBLICATION_BUFFER_MINUTES, max_age_days=LIVE_MAX_AGE)
    X = X.join(egx_controls(days, egx))
    coverage = {key: float(X[key].notna().mean()) for key in markets}
    if any(x < 0.60 for x in coverage.values()):
        insufficient = [f'{m}: {v:.0%}' for m, v in coverage.items() if v < .60]
        raise ValueError('Historical timestamps do not align: feature coverage under 60% for ' + ', '.join(insufficient))
    close_features = market_feature_columns([m.key for m in MARKETS if m.key in markets]) + list(EGX_FEATURE_COLUMNS)
    X = X[close_features]
    target = (egx.pct_change(fill_method=None).loc[days] > 0).astype(int)
    if len(X) != 599 or target.nunique() < 2:
        raise ValueError('Expected 599 labeled next-session returns from 600 EGX closes and both UP/NON-UP classes')
    min_markets = max(10, math.ceil(0.70 * len(markets)))
    comparator = None
    if global_frame is not None:
        global_frame = global_frame.reindex(days)
        usable = intraday.verified_days(global_frame)
        if not usable.iloc[-1]:
            raise ValueError('Waiting for verified moves from every model-1 index for the latest EGX session.')
        comparator = X.loc[usable].copy()
        X = comparator.copy()
        returns = global_frame.loc[usable, [key + intraday.MOVE_SUFFIX for key in markets]]
        returns.columns = list(markets)
        X[list(markets)] = returns
        X = X.join(global_frame.loc[usable, [intraday.RISING_SHARE, intraday.MEAN_MOVE]])
        target = target.loc[usable]
    features = X.columns.tolist()
    historical = historical_scores
    if historical is None:
        wf = walk_forward(X, target, min_train=350, refit_every=refit_every, close_only=comparator)
        eligible = X.loc[wf.index, list(markets)].notna().sum(axis=1) >= min_markets
        if not eligible.any():
            raise ValueError(f'No walk-forward days meet the live minimum of {min_markets} available foreign indices.')
        historical = evaluate(wf.loc[eligible])
        historical.update(available_test_days=int(len(wf)), live_data_coverage=float(eligible.mean()),
                          refit_every_egx_sessions=refit_every, last_test_session=egx.index[-1].date().isoformat())
    pipe = fit_direction_model(X, target)
    exchange, countries = country_weights(pipe, features)
    model = serialize_model(pipe, features, exchange, countries, target, egx, historical, asof, min_markets)
    if global_frame is not None:
        model.update(uses_global_moves=True, complete_market_training_days=len(target), eligible_egx_days=len(days),
                     market_observations={key: int(global_frame[key + intraday.MOVE_SUFFIX].notna().sum())
                                          for key in markets})
    return model, egx


def init(args, asof: datetime):
    with trial_lock(Path(args.runtime)):
        init_trial(args, asof)


def init_trial(args, asof: datetime):
    rt = Path(args.runtime)
    model_file = rt / 'model.json'
    state_file = rt / 'state.json'
    if (model_file.exists() or state_file.exists()) and not args.force_reset:
        raise RuntimeError('Frozen model/state already exist. Use status/once/watch to continue, '
                           'or select a new --runtime folder to train an upgraded model.')
    h_path = Path(args.history)
    if h_path.exists() and not getattr(args, 'refresh_history', False):
        print(f'Reading existing historical JSON: {h_path}')
        payload = read_json(h_path)
    else:
        payload = build_history(h_path, asof, Path(args.csv_dir) if args.csv_dir else None,
                                Path(args.egx_csv) if args.egx_csv else None,
                                allow_missing_markets=args.allow_missing_markets)
    model, egx = train_from_history(payload, asof, allow_missing_markets=args.allow_missing_markets)
    trial_id = uuid4().hex
    model['trial_id'] = trial_id
    # No state mutation until validation/training finishes.
    dump_atomic(model_file, model)
    state = {
        'schema_version': SCHEMA_VERSION, 'trial_id': trial_id, 'trial_target': DURATION,
        'trial_mode': args.trial_mode, 'first_forecast_date': None, 'end_exclusive_date': None,
        'started_at_cairo': asof.isoformat(), 'predictions': [],
        'recent_egx': date_rows(egx.tail(6)),
        'completed_at_cairo': None,
    }
    dump_atomic(state_file, state)
    write_reports(rt, state, model, asof)
    if model['excluded_market_keys']:
        print('EXCLUDED from frozen model:', ', '.join(model['excluded_market_keys']))
    print(f'Training finished. Coefficients FROZEN in {model_file}; state in {state_file}')
    print(f'Training history: {h_path}')
    print_historical_scores(model)
    print('LIVE accuracy: not measured yet (0 predictions settled).')


def predict_proba_frozen(model: dict, feature_values: dict) -> float:
    cols = model['features']
    xs = np.array([float(feature_values.get(k)) if feature_values.get(k) is not None and
                   np.isfinite(float(feature_values[k])) else np.nan for k in cols], dtype=float)
    med = np.array(model['imputer_medians'])
    means = np.array(model['scaler_mean'])
    scale = np.array(model['scaler_scale'])
    coef = np.array(model['coefficients'])
    xs = np.where(np.isfinite(xs), xs, med)
    z = float(np.dot((xs - means) / scale, coef) + model['intercept'])
    return 1. / (1. + math.exp(-max(-500.0, min(500.0, z))))


def recent_price_series(state):
    return series_from_rows(state['recent_egx'], 'recent EGX30')


def update_recent_egx(state: dict, fetched: pd.Series):
    old = recent_price_series(state)
    merged = clean_close(pd.concat([old, fetched]))
    # Restrict to last six egx closes; this is NOT the 600-day training file.
    state['recent_egx'] = date_rows(merged.tail(6))
    return merged


def settle_pending(state: dict, egx: pd.Series, asof: datetime, *, allow_unverified_void=False, label='CLOSE'):
    changed = 0
    for pred in state['predictions']:
        if pred['status'] != 'PENDING':
            continue
        day = pd.Timestamp(pred['date'])
        if not market_time_passed(day, 'Africa/Cairo', EGX_CLOSE.strftime('%H:%M'), asof):
            continue
        if day not in egx.index:
            if allow_unverified_void and (egx.index > day).sum() >= 2 and (egx.index < day).any():
                pred.update(status='VOID', void_kind='NO_VERIFIED_SESSION',
                            void_reason='No verified closing price after two later EGX sessions',
                            void_at_cairo=asof.isoformat())
                changed += 1
                print(f'{label} voided {pred["date"]}: no verified closing price for that date.')
            continue
        current_close = float(egx.loc[day])
        previous_close = float(pred['previous_egx_close'])
        if pred.get('model_version', 1) >= 2:
            preceding = egx.loc[egx.index < day]
            if preceding.empty:
                continue  # Need both verified closes to score a next-session return.
            actual_previous_day = preceding.index[-1].date().isoformat()
            if actual_previous_day != pred['source_last_egx_day']:
                if actual_previous_day < pred['source_last_egx_day']:
                    continue  # This result feed is missing a close already used in the forecast.
                pred.update(status='VOID', void_reason='A newer EGX close preceded the target session',
                            void_kind='EGX_REFERENCE', void_at_cairo=asof.isoformat())
                print(f"Voided {pred['date']}: previous close should be from {actual_previous_day}.")
                changed += 1
                continue
            previous_close = float(preceding.iloc[-1])
            pred['actual_previous_egx_close'] = previous_close
        direction = 'UP' if current_close > previous_close else 'DOWN' if current_close < previous_close else 'FLAT'
        expected = 'NON_UP' if pred['direction'] == 'NON_UP' and direction != 'UP' else direction
        hit = pred['direction'] == expected
        pred.update(status='SETTLED', actual_egx_close=current_close,
                    actual_direction=direction, correct=bool(hit),
                    settled_at_cairo=asof.isoformat())
        changed += 1
        print(f"{label} settled {pred['date']}: predicted {pred['direction']}, actual {direction} "
              f"(EGX30 {previous_close:.2f} -> {current_close:.2f}): {'CORRECT' if hit else 'WRONG'}")
    return changed


def schedule_candidate(asof: datetime):
    # Current trading day only. The model never backfills an unissued prediction.
    if asof.weekday() in (4, 5):
        return None
    d = pd.Timestamp(asof.date())
    cutoff = local_datetime(d, FORECAST_CUTOFF.strftime('%H:%M'), 'Africa/Cairo')
    opening = local_datetime(d, EGX_OPEN.strftime('%H:%M'), 'Africa/Cairo')
    return d if cutoff <= pd.Timestamp(asof) < opening else None


def can_issue_close(state, day):
    if day is None or state.get('completed_at_cairo'):
        return False
    if any(p['date'] == day.date().isoformat() for p in state['predictions']):
        return False
    mode = state.get('trial_mode', 'sessions')  # backward-compatible with older state snapshots
    if mode == 'sessions' and sum(p['status'] != 'VOID' for p in state['predictions']) >= DURATION:
        return False
    if mode == 'calendar' and state.get('end_exclusive_date') and day.date() >= pd.Timestamp(state['end_exclusive_date']).date():
        return False
    return True


def collect_market_closes(active_keys, asof, csv_dir, *, lookback=45, min_sessions=2, max_age_days=LIVE_MAX_AGE):
    markets, failed = {}, []
    with ThreadPoolExecutor(max_workers=4) as pool:
        lookups = {m.key: pool.submit(fetch_market, m, asof, csv_dir, lookback_days=lookback, min_sessions=min_sessions,
                                     cache_dir=DEFAULT_MARKET_CACHE, prefer_cache=False, max_age_days=max_age_days,
                                     reuse_current_saudi=True)
                   for m in MARKETS if m.key in active_keys}
        for key, lookup in lookups.items():
            try:
                markets[key] = lookup.result()
            except Exception as exc:
                failed.append(f'{key}: {exc}')
    return markets, failed


def live_forecast(state: dict, model: dict, asof: datetime, csv_dir: Path | None, *, publication_clock=None,
                  market_data=None, label='FROZEN CLOSE', global_data=None):
    day = schedule_candidate(asof)
    if not can_issue_close(state, day):
        return False
    egypt_history = recent_price_series(state)
    prev_session = egypt_history.index[-1]
    # A short 6-price rolling cache must not silently bridge months of missing prices.
    gaps = egypt_history.index.to_series().diff().dt.days.dropna()
    if prev_session >= day or (day - prev_session).days > EGX_MAX_AGE or (gaps > EGX_MAX_AGE).any():
        print('Recent EGX prices have gaps or are stale; refusing to publish prediction.')
        return False
    active_keys = set(model.get('active_market_keys', [m.key for m in MARKETS]))
    markets, failed = market_data if market_data is not None else collect_market_closes(active_keys, asof, csv_dir)
    X = assemble_market_features(pd.DatetimeIndex([day]), markets, asof,
                                 model['foreign_delay_minutes'], max_age_days=LIVE_MAX_AGE)
    controls = egx_controls(pd.DatetimeIndex([day]), egypt_history)
    X = X.join(controls)
    row = X.loc[day]
    available = sum(pd.notna(row.get(k, np.nan)) for k in active_keys)
    if available < model['min_available_live_markets']:
        print(f'{label}: only {available}/{len(active_keys)} fresh foreign returns; no prediction. ' + '; '.join(failed[:5]))
        return False
    if model.get('uses_same_day'):
        print(f'{label}: use start to run the current all-market models.')
        return False
    if model.get('uses_global_moves'):
        if global_data is None:
            return False
        latest, market_rows = global_data
        missing = [key for key in model['active_market_keys'] if pd.isna(latest.get(key + intraday.MOVE_SUFFIX))]
        if missing:
            print(f"{label}: waiting for verified moves from {', '.join(missing)}.")
            return False
        row = row.copy()
        for key in active_keys:
            row[key] = latest[key + intraday.MOVE_SUFFIX]
        row[intraday.RISING_SHARE] = latest[intraday.RISING_SHARE]
        row[intraday.MEAN_MOVE] = latest[intraday.MEAN_MOVE]
    feats = {key: float(row[key]) if pd.notna(row[key]) else None for key in model['features']}
    prob = predict_proba_frozen(model, feats)
    model_version = model.get('model_version', 1)
    p_direction = 'UP' if prob >= .5 else ('NON_UP' if model_version >= 2 else 'DOWN')
    threshold = model.get('signal_threshold', SIGNAL_THRESHOLD)
    signal = 'ELEVATED' if max(prob, 1 - prob) >= threshold else 'TENTATIVE'
    # When production runs slowly, never back-date the publication timestamp.
    publication_time = (publication_clock or now_cairo)()
    if schedule_candidate(publication_time) != day or publication_time < asof:
        print('Forecast feed finished outside the pre-open window; no retrospective publication.')
        return False
    trained_through = model.get('trained_egx_last_date')
    if trained_through and trained_through >= day.date().isoformat():
        print(f'{label}: model training includes the target session; no prediction.')
        return False
    record = {
        'date': day.date().isoformat(), 'status': 'PENDING', 'model_version': model_version,
        'published_at_cairo': publication_time.isoformat(),
        'cutoff_cairo': local_datetime(day, FORECAST_CUTOFF.strftime('%H:%M'), 'Africa/Cairo').isoformat(),
        'source_last_egx_day': str(prev_session.date()),
        'previous_egx_close': float(egypt_history.iloc[-1]),
        'markets_available': int(available),
        'model_trained_through': trained_through,
        'baseline_prob_up': model.get('training_up_rate'),
        'features_at_publication': feats, 'signal': signal,
        'prob_up': round(prob, 8), 'prob_down': round(1-prob, 8),
        'direction': p_direction,
        'actual_direction': None, 'actual_egx_close': None, 'correct': None,
    }
    if model_version >= 2:
        record['prob_non_up'] = record.pop('prob_down')
    if model.get('uses_global_moves'):
        record['market_moves'] = market_rows
        record['latest_markets_count'] = len(market_rows)
        record['same_session_count'] = sum(entry.get('movement_basis') == 'SAME_SESSION' for entry in market_rows)
        record['global_rising_share'] = feats[intraday.RISING_SHARE]
    state['predictions'].append(record)
    if state.get('trial_mode') == 'calendar' and not state.get('first_forecast_date'):
        state['first_forecast_date'] = record['date']
        state['end_exclusive_date'] = (day + timedelta(days=DURATION)).date().isoformat()
        print(f"Calendar trial: {state['first_forecast_date']} through "
              f"{(day + timedelta(days=DURATION-1)).date()}, inclusive.")
    other = 'NON-UP' if model_version >= 2 else 'DOWN'
    detail = f" Verified moves from all {record['latest_markets_count']} markets." if 'latest_markets_count' in record else ''
    print(f"{record['date']} {label}: {p_direction} ({signal}). P(UP) {prob:.1%}, "
          f"P({other}) {1-prob:.1%}. Markets available: {available}/{len(active_keys)}.{detail}")
    return True


def summarize(state, asof: datetime | None = None):
    asof = asof or now_cairo()
    preds = state['predictions']
    settled = [p for p in preds if p['status'] == 'SETTLED']
    n = len(settled)
    correct = sum(int(p['correct']) for p in settled)
    pending = sum(p['status'] == 'PENDING' for p in preds)
    void = sum(p['status'] == 'VOID' for p in preds)
    nonvoid = len(preds) - void
    mode = state.get('trial_mode', 'sessions')
    end = state.get('end_exclusive_date')
    elapsed = bool(end and pd.Timestamp(asof.date()) >= pd.Timestamp(end))
    complete = (n >= DURATION) if mode == 'sessions' else (mode == 'calendar' and elapsed and pending == 0)
    res = {'trial_mode': mode, 'target': None if mode == 'continuous' else DURATION,
           'first_forecast_date': state.get('first_forecast_date'),
           'end_exclusive_date': end,
           'issued_predictions': nonvoid,
           'evaluated_predictions': n, 'correct': correct, 'incorrect': n - correct,
           'pending': pending,
           'void_holidays': sum(p['status'] == 'VOID' and p.get('void_kind', 'HOLIDAY') == 'HOLIDAY' for p in preds),
           'void_reference_errors': sum(p.get('void_kind') == 'EGX_REFERENCE' for p in preds),
           'void_unverified_sessions': sum(p.get('void_kind') == 'NO_VERIFIED_SESSION' for p in preds),
           'remaining_to_issue': max(0, DURATION - nonvoid) if mode == 'sessions' else None,
           'accuracy': correct / n if n else None,
           'status': 'COMPLETE' if complete else 'IN_PROGRESS'}
    if n:
        res['up_predictions'] = sum(p['direction'] == 'UP' for p in settled)
        res['down_predictions'] = sum(p['direction'] == 'DOWN' for p in settled)
        res['non_up_predictions'] = sum(p['direction'] == 'NON_UP' for p in settled)
        res['brier'] = sum((p['prob_up'] - int(p['actual_direction'] == 'UP'))**2 for p in settled) / n
        elevated = [p for p in settled if p.get('signal') == 'ELEVATED']
        res['elevated_signals'] = len(elevated)
        res['elevated_accuracy'] = sum(p['correct'] for p in elevated) / len(elevated) if elevated else None
    return res


def write_reports(runtime: Path, state: dict, model: dict, asof: datetime | None = None):
    stats = summarize(state, asof)
    settled = [p for p in state['predictions'] if p['status'] == 'SETTLED']
    if settled and 'training_up_rate' in model:
        rates = [p.get('baseline_prob_up', model['training_up_rate']) for p in settled]
        actual = [int(p['actual_direction'] == 'UP') for p in settled]
        stats['baseline_accuracy'] = sum(x == int(rate >= 0.5) for x, rate in zip(actual, rates)) / len(actual)
        stats['baseline_brier'] = sum((rate - x) ** 2 for x, rate in zip(actual, rates)) / len(actual)
    dump_atomic(runtime / 'report.json', {
        'live_trial': stats,
        'frozen_model_trained_at': model['trained_at_cairo'],
        'historical_walk_forward': model['historical_walk_forward'],
        'model_version': model.get('model_version', 1),
        'trained_through': model['trained_egx_last_date'],
        'active_markets': model.get('active_market_keys', []),
        'excluded_markets': model.get('excluded_market_keys', []),
        **({'complete_market_training_days': model['complete_market_training_days'],
            'eligible_egx_days': model['eligible_egx_days'], 'market_observations': model['market_observations']}
           if model.get('uses_global_moves') else {}),
        'country_weights': model['country_weights'],
        'market_weights': model['market_weights'],
        'method': ('Model refits after each newly confirmed EGX close. ' if not model.get('frozen', True)
                   else 'Model remains frozen throughout the trial. ') +
                  'The target compares the final EGX30 close to the previous session close.',
    })
    columns = ['date', 'status', 'direction', 'signal', 'prob_up', 'prob_non_up', 'prob_down',
               'markets_available', 'model_trained_through', 'source_last_egx_day', 'previous_egx_close', 'actual_previous_egx_close',
               'actual_egx_close', 'actual_direction', 'correct', 'published_at_cairo', 'settled_at_cairo', 'void_reason']
    if model.get('uses_global_moves'):
        columns += ['latest_markets_count', 'same_session_count', 'global_rising_share']
    filename = 'predictions.csv' if state.get('trial_mode') == 'continuous' else 'predictions_30.csv'
    try:
        with atomic_text(runtime / filename, encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(state['predictions'])
    except PermissionError:
        print(f'Could not replace {filename}. Close the file before the next update.', file=sys.stderr)


def daily_once(args, asof: datetime):
    with trial_lock(Path(args.runtime)):
        return update_trial(args, asof)


def load_trial(runtime: Path):
    model_file, state_file = runtime / 'model.json', runtime / 'state.json'
    if not (model_file.exists() and state_file.exists()):
        raise RuntimeError('Not initialized. Run: python app.py init [--egx-csv path/to/EGX30.csv]')
    model, state = read_json(model_file), read_json(state_file)
    if model.get('schema_version') != SCHEMA_VERSION or state.get('schema_version') != SCHEMA_VERSION:
        raise RuntimeError('Unsupported saved model/state format')
    if model.get('trial_id') != state.get('trial_id'):
        raise RuntimeError('Model and state are from different trials. Restore the matching pair or use a new --runtime.')
    return model, state


def update_trial(args, asof: datetime):
    runtime = Path(args.runtime)
    model, state = load_trial(runtime)
    if summarize(state, asof)['status'] == 'COMPLETE' or state['completed_at_cairo']:
        if not state['completed_at_cairo']:
            state['completed_at_cairo'] = asof.isoformat()
            dump_atomic(runtime / 'state.json', state)
        write_reports(runtime, state, model, asof)
        return summarize(state, asof)
    csvdir = Path(args.csv_dir) if args.csv_dir else None
    # A watcher restarted after a long outage also needs enough history to score old pending rows.
    oldest = min((pd.Timestamp(p['date']).date() for p in state['predictions'] if p['status'] == 'PENDING'),
                 default=asof.date())
    lookback = max(45, (asof.date() - oldest).days + 14)
    refreshed = False
    preopen = schedule_candidate(asof) is not None
    try:
        egx = fetch_egx(asof, csvdir, Path(args.egx_csv) if args.egx_csv else None, lookback_days=lookback,
                        min_sessions=6 if preopen else 2, max_age_days=EGX_MAX_AGE if preopen else None)
    except Exception as exc:
        print('EGX30 refresh unavailable. Forecast and result updates skipped:', str(exc))
    else:
        egx = update_recent_egx(state, egx)
        settle_pending(state, egx, asof)
        refreshed = True
    stats = summarize(state, asof)
    if stats['status'] == 'COMPLETE':
        state['completed_at_cairo'] = asof.isoformat()
    elif preopen and refreshed:
        live_forecast(state, model, asof, csvdir)
    dump_atomic(runtime / 'state.json', state)
    write_reports(runtime, state, model, asof)
    summary = summarize(state, asof)
    print(f"TRIAL: {summary['correct']}/{summary['evaluated_predictions']} correct "
          f"({summary['accuracy']:.1%} accuracy)" if summary['accuracy'] is not None
          else 'TRIAL: no scored sessions yet')
    if summary['trial_mode'] == 'calendar':
        print(f"calendar window {summary['first_forecast_date']} to {summary['end_exclusive_date']} (end exclusive); "
              f"predicted={summary['issued_predictions']}, pending={summary['pending']}")
    else:
        print(f"issued={summary['issued_predictions']}/30; pending={summary['pending']}; "
              f"remaining={summary['remaining_to_issue']}")
    return summary


def print_historical_scores(model):
    scores = model['historical_walk_forward']
    print(f"Historical walk-forward accuracy: {scores['accuracy']:.1%}")
    if scores.get('majority_baseline_accuracy') is not None:
        print(f"Historical majority baseline: {scores['majority_baseline_accuracy']:.1%}")
    if scores.get('legacy_model_accuracy') is not None:
        print(f"Historical previous model: {scores['legacy_model_accuracy']:.1%}")
    if scores.get('close_only_accuracy') is not None:
        print(f"Historical close-only model on the same dates: {scores['close_only_accuracy']:.1%}")
    if scores.get('brier_score') is not None:
        print(f"Historical Brier score: {scores['brier_score']:.4f} (lower is better)")


def adaptive_seed(args, asof, runtime, frozen):
    history = runtime / 'history.json'
    csvdir = Path(args.csv_dir) if args.csv_dir else None
    egx_csv = Path(args.egx_csv) if args.egx_csv else None
    active = set(frozen['active_market_keys'])

    def train_seed(payload):
        missing = active - set(payload['markets'])
        if missing:
            raise ValueError(f"Adaptive history is missing {', '.join(sorted(missing))}")
        payload['markets'] = {key: row for key, row in payload['markets'].items() if key in active}
        payload['excluded_markets'] = [m.key for m in MARKETS if m.key not in active]
        model, egx = train_from_history(payload, asof, allow_missing_markets=True, refit_every=1)
        return model, egx, payload

    source = history if history.is_file() else Path(args.history)
    try:
        model, egx, payload = train_seed(read_json(source))
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f'ADAPTIVE CLOSE: collecting a current seed history ({exc}).')
        payload = build_history(history, asof, csvdir, egx_csv, allow_missing_markets=len(active) < len(MARKETS))
        model, egx, payload = train_seed(payload)
    if (runtime / 'state.json').is_file():
        state = read_json(runtime / 'state.json')
        if state['predictions']:
            raise RuntimeError(f'{runtime / "model.json"} is missing for a recorded adaptive trial; restore its backup.')
        trial_id = state['trial_id']
    else:
        trial_id = uuid4().hex
        state = {'trial_id': trial_id, 'schema_version': SCHEMA_VERSION, 'trial_mode': 'continuous',
                 'started_at_cairo': asof.isoformat(), 'predictions': [], 'recent_egx': date_rows(egx.tail(6)),
                 'completed_at_cairo': None}
    model.update(trial_id=trial_id, frozen=False, learning_updates=0)
    dump_atomic(history, payload)
    dump_atomic(runtime / 'state.json', state)
    dump_atomic(runtime / 'model.json', model)
    write_reports(runtime, state, model, asof)
    print(f"ADAPTIVE CLOSE initialized through {model['trained_egx_last_date']}.")
    print_historical_scores(model)
    return model, state


def advance_adaptive(runtime, model, egx, market_data, asof, *, moves_book=None, label='ADAPTIVE CLOSE'):
    latest = egx.index[-1].date().isoformat()
    if latest <= model['trained_egx_last_date']:
        return model
    history = runtime / 'history.json'
    payload = read_json(history)
    egypt, markets = validate_history(payload, allow_missing_markets=True)
    fresh, failures = market_data
    for key, prices in markets.items():
        if key in fresh:
            prices = clean_close(pd.concat([prices, fresh[key]])).tail(600)
            markets[key] = prices
        payload['markets'][key]['data'] = date_rows(prices)
    egypt = clean_close(pd.concat([egypt, egx])).tail(600)
    payload['egx30']['data'] = date_rows(egypt)
    payload['generated_at_cairo'] = asof.isoformat()
    if failures:
        print(f'{label}: some sources unavailable for refresh; checking cached history. ' + '; '.join(failures[:3]))
    global_frame = intraday.history_features(moves_book, markets, egypt.index[1:], asof) if moves_book is not None else None
    updated, _ = train_from_history(payload, asof, allow_missing_markets=True,
                                    historical_scores=model['historical_walk_forward'], global_frame=global_frame)
    updated.update(trial_id=model['trial_id'], frozen=False, learning_updates=model.get('learning_updates', 0) + 1)
    dump_atomic(history, payload)
    dump_atomic(runtime / 'model.json', updated)
    print(f'{label} learned through the confirmed {latest} EGX session.')
    return updated


def next_due(retries, key, asof, minutes):
    if key in retries and asof < retries[key]:
        return False
    retries[key] = asof + timedelta(minutes=minutes)
    return True


def expected_egx_day(asof, time_of_day):
    day = asof.date()
    if asof.time() < time_of_day:
        day -= timedelta(days=1)
    while day.weekday() in (4, 5):
        day -= timedelta(days=1)
    return day.isoformat()


def scheduled_openings(args, asof, retries, publication_clock):
    root = Path(args.runtime) / 'opening'
    csvdir = Path(args.csv_dir) if args.csv_dir else None
    path = root / 'report.json'
    report = read_json(path) if path.is_file() else {}
    rebuild_report = report.get('report_version') != opening.REPORT_VERSION
    day = pd.Timestamp(asof.date())
    date_key = day.date().isoformat()
    state = opening.load_state(root)
    valid_day = asof.weekday() not in (4, 5)
    setup = (report.get('egx_open_sessions', 0) < 2 * opening.MIN_CONDITIONAL_SESSIONS
             and next_due(retries, 'opening setup', asof, 30))
    prepare = (valid_day and opening.OPENING_PREP <= asof.time() < FORECAST_CUTOFF
               and next_due(retries, 'early open capture', asof, 10))
    publish = (schedule_candidate(asof) is not None and not any(row['date'] == date_key for row in state['predictions'])
               and next_due(retries, 'opening forecast', asof, 2))
    settle = ((report.get('last_egx_open_session') or '') < expected_egx_day(asof, opening.OPENING_RESULT)
              and next_due(retries, 'EGX opening result', asof, 20))
    if not (rebuild_report or setup or prepare or publish or settle):
        return
    cache = opening.load_prices(root)
    eligible = opening.scheduled_opening_markets(day)
    keys = set(opening.EARLY_MARKETS) | {'EGX30'} if setup else set()
    if prepare or (settle and valid_day and asof.time() >= opening.OPENING_RESULT):
        keys.update(eligible)
    if publish:
        for key in eligible:
            entry = cache['markets'].get(key, {})
            last_capture = entry.get('observed_at_cairo', '')
            if not last_capture.startswith(date_key) or not any(row['date'] == date_key and row['open'] is not None
                                                                for row in entry.get('prices', [])[-2:]):
                keys.add(key)
    if settle:
        keys.add('EGX30')
    if keys:
        opening.refresh_prices(root, cache, sorted(keys), asof, csvdir, full=setup)
    gaps = opening.gaps_for_cache(cache, asof)
    changed = opening.settle(state, gaps, asof)
    if publish:
        changed = opening.publish(state, gaps, asof, publication_clock) or changed
    if changed or not (root / 'state.json').is_file():
        dump_atomic(root / 'state.json', state)
    opening.write_reports(root, cache, state, gaps, asof)


def scheduled_closes(args, asof, retries, publication_clock, *, shared=None):
    root = Path(args.runtime)
    adaptive_root = root / 'adaptive'
    csvdir = Path(args.csv_dir) if args.csv_dir else None
    egx_csv = Path(args.egx_csv) if args.egx_csv else None
    with trial_lock(root), trial_lock(adaptive_root):
        if not any((root / name).exists() for name in ('model.json', 'state.json')):
            if not next_due(retries, 'frozen setup', asof, 5):
                return
            args.force_reset = False
            args.refresh_history = False
            try:
                init_trial(args, asof)
            except ValueError:
                if not Path(args.history).is_file():
                    raise
                print('FROZEN CLOSE: rebuilding unusable seed history.')
                args.refresh_history = True
                init_trial(args, asof)
        frozen, frozen_state = load_trial(root)
        adaptive = adaptive_state = None
        if (adaptive_root / 'model.json').is_file():
            try:
                adaptive, adaptive_state = load_trial(adaptive_root)
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                if next_due(retries, 'adaptive status', asof, 20):
                    print(f'ADAPTIVE CLOSE could not read saved state: {exc}', file=sys.stderr)
        elif next_due(retries, 'adaptive setup', asof, 10):
            try:
                adaptive, adaptive_state = adaptive_seed(args, asof, adaptive_root, frozen)
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                print(f'ADAPTIVE CLOSE initialization will retry: {exc}', file=sys.stderr)

        jobs = [(root, frozen, frozen_state, 'FROZEN CLOSE')]
        if adaptive is not None:
            jobs.append((adaptive_root, adaptive, adaptive_state, 'ADAPTIVE CLOSE'))
        original = {job_root: copy.deepcopy(state) for job_root, _, state, _ in jobs}
        for _, _, state, _ in jobs:
            if not state.get('completed_at_cairo') and summarize(state, asof)['status'] == 'COMPLETE':
                state['completed_at_cairo'] = asof.isoformat()
        day = schedule_candidate(asof)
        issuing = [job for job in jobs if can_issue_close(job[2], day)]
        active_issues = issuing if issuing and next_due(retries, 'close forecast', asof, 1) else []
        due_session = expected_egx_day(asof, time(15, 15))
        behind = adaptive is not None and adaptive['trained_egx_last_date'] < due_session
        pending = [pd.Timestamp(row['date']).date() for _, _, state, _ in jobs
                   for row in state['predictions'] if row['status'] == 'PENDING' and row['date'] <= due_session]
        refresh = active_issues or ((pending or behind) and next_due(retries, 'confirmed close', asof, 20))
        if refresh:
            earliest = min([asof.date(), *pending] + ([pd.Timestamp(adaptive['trained_egx_last_date']).date()] if behind else []))
            lookback = max(45, (asof.date() - earliest).days + 14)
            egx = fetch_egx(asof, csvdir, egx_csv, lookback_days=lookback, min_sessions=6,
                            max_age_days=EGX_MAX_AGE if active_issues else HISTORY_MAX_AGE)
            for _, _, state, label in jobs:
                update_recent_egx(state, egx)
                settle_pending(state, egx, asof, allow_unverified_void=True, label=label)
                if summarize(state, asof)['status'] == 'COMPLETE' and not state.get('completed_at_cairo'):
                    state['completed_at_cairo'] = asof.isoformat()

            learning = adaptive is not None and adaptive['trained_egx_last_date'] < egx.index[-1].date().isoformat()
            keys = {key for _, model, _, _ in active_issues for key in model['active_market_keys']}
            if learning:
                keys.update(adaptive['active_market_keys'])
            market_data = collect_market_closes(keys, asof, csvdir, lookback=lookback,
                                                max_age_days=LIVE_MAX_AGE if active_issues else HISTORY_MAX_AGE) if keys else ({}, [])
            if shared is not None:
                shared.update(egx=egx, lookback=lookback, market_data=market_data, market_keys=keys)
            if active_issues and can_issue_close(frozen_state, day):
                live_forecast(frozen_state, frozen, asof, csvdir, publication_clock=publication_clock,
                              market_data=market_data)

            if learning:
                try:
                    adaptive = advance_adaptive(adaptive_root, adaptive, egx, market_data, asof)
                except (OSError, RuntimeError, ValueError, KeyError) as exc:
                    print(f'ADAPTIVE CLOSE learning will retry: {exc}', file=sys.stderr)
            if adaptive is not None and active_issues and can_issue_close(adaptive_state, day):
                if adaptive['trained_egx_last_date'] == recent_price_series(adaptive_state).index[-1].date().isoformat():
                    live_forecast(adaptive_state, adaptive, asof, csvdir, publication_clock=publication_clock,
                                  market_data=market_data, label='ADAPTIVE CLOSE')
                else:
                    print('ADAPTIVE CLOSE: waiting for training through the latest verified EGX close.')

        for folder, model, state, _ in jobs:
            current = adaptive if folder == adaptive_root else model
            if state != original[folder]:
                dump_atomic(folder / 'state.json', state)
            if state != original[folder] or current != model or not (folder / 'report.json').is_file():
                write_reports(folder, state, current, asof)


def seed_global_moves(args, runtime, root, book, asof, csvdir):
    original, _ = load_trial(runtime)
    active = original['active_market_keys']
    seed_paths = (runtime / 'adaptive' / 'history.json', Path(args.history))
    payload = None
    for path in seed_paths:
        if path.is_file():
            candidate = read_json(path)
            if set(active) <= set(candidate['markets']):
                payload = candidate
                break
    if payload is None:
        raise ValueError('Waiting for the 600-session history for every index in method 1.')
    payload['markets'] = {key: payload['markets'][key] for key in active}
    payload['excluded_markets'] = [m.key for m in MARKETS if m.key not in active]
    egx, markets = validate_history(payload, allow_missing_markets=True, asof=asof)
    if not book['markets']:
        book['markets'] = intraday.load_book(runtime / 'intraday')['markets']
        if book['markets']:
            dump_atomic(root / 'observations.json', book)
        else:
            intraday.refresh_book(root, book, asof, csvdir, markets, full=True)
    global_frame = intraday.history_features(book, markets, egx.index[1:], asof)
    if asof.time() >= FORECAST_CUTOFF:
        features, rows = intraday.snapshot(book, markets, pd.Timestamp(asof.date()), asof, active)
        intraday.save_snapshot(root, features, rows, asof)
    for kind, label in GLOBAL_MOVE_MODELS.items():
        branch = root / kind
        if (branch / 'model.json').exists():
            continue
        is_frozen = kind == 'frozen'
        model, _ = train_from_history(payload, asof, allow_missing_markets=True, global_frame=global_frame,
                                      refit_every=HISTORICAL_REFIT_SESSIONS if is_frozen else 1)
        state_path = branch / 'state.json'
        if state_path.is_file():
            state = read_json(state_path)
            if state['predictions']:
                raise RuntimeError(f'{label}: restore the missing model for the existing forecast history.')
        else:
            state = {'schema_version': SCHEMA_VERSION, 'trial_id': uuid4().hex,
                     'trial_mode': args.trial_mode if is_frozen else 'continuous', 'predictions': [],
                     'started_at_cairo': asof.isoformat(), 'completed_at_cairo': None, 'recent_egx': date_rows(egx.tail(6))}
        model.update(trial_id=state['trial_id'], frozen=is_frozen, learning_updates=0)
        if not is_frozen:
            dump_atomic(branch / 'history.json', payload)
        dump_atomic(state_path, state)
        dump_atomic(branch / 'model.json', model)
        write_reports(branch, state, model, asof)
        print(f"{label} initialized: {model['complete_market_training_days']}/{model['eligible_egx_days']} "
              f"sessions with all {len(active)} verified market moves through {model['trained_egx_last_date']}.")
        print_historical_scores(model)


def scheduled_global_moves(args, asof, retries, publication_clock, *, shared=None):
    runtime = Path(args.runtime)
    if not (runtime / 'model.json').is_file():
        return
    root = runtime / 'global_moves'
    csvdir = Path(args.csv_dir) if args.csv_dir else None
    egx_csv = Path(args.egx_csv) if args.egx_csv else None
    reused = shared or {}
    with trial_lock(root):
        book = None
        if not all((root / kind / 'model.json').is_file() for kind in GLOBAL_MOVE_MODELS):
            if next_due(retries, 'global moves seed', asof, 30):
                book = intraday.load_book(root)
                try:
                    seed_global_moves(args, runtime, root, book, asof, csvdir)
                except (ValueError, RuntimeError, OSError, KeyError) as exc:
                    print(f'ALL-MARKET setup will retry: {exc}', file=sys.stderr)

        jobs = {}
        for kind, label in GLOBAL_MOVE_MODELS.items():
            folder = root / kind
            if (folder / 'model.json').is_file():
                try:
                    jobs[kind] = load_trial(folder)
                except (ValueError, RuntimeError, OSError, KeyError) as exc:
                    if next_due(retries, f'global moves state {kind}', asof, 20):
                        print(f'{label} could not read its saved trial: {exc}', file=sys.stderr)
        if not jobs:
            return
        active = next(iter(jobs.values()))[0]['active_market_keys']
        previous_states = {kind: copy.deepcopy(state) for kind, (_, state) in jobs.items()}
        for _, state in jobs.values():
            if not state.get('completed_at_cairo') and summarize(state, asof)['status'] == 'COMPLETE':
                state['completed_at_cairo'] = asof.isoformat()

        day = schedule_candidate(asof)
        issuing = [kind for kind, (_, state) in jobs.items() if can_issue_close(state, day)]
        due_session = expected_egx_day(asof, time(15, 15))
        adaptive = jobs['adaptive'][0] if 'adaptive' in jobs else None
        behind = adaptive is not None and adaptive['trained_egx_last_date'] < due_session
        pending = [pd.Timestamp(row['date']).date() for _, state in jobs.values()
                   for row in state['predictions'] if row['status'] == 'PENDING' and row['date'] <= due_session]
        prewarm = (asof.weekday() not in (4, 5) and time(9, 35) <= asof.time() < FORECAST_CUTOFF
                   and next_due(retries, 'global moves prewarm', asof, 15))
        if prewarm and book is None:
            book = intraday.load_book(root)
            intraday.refresh_book(root, book, asof, csvdir, active)

        issue = bool(issuing and next_due(retries, 'global moves forecast', asof, 2))
        refresh = issue or ((behind or pending) and next_due(retries, 'global moves closed outcomes', asof, 20))
        if refresh:
            earliest = min([asof.date(), *pending] + ([pd.Timestamp(adaptive['trained_egx_last_date']).date()] if behind else []))
            lookback = max(45, (asof.date() - earliest).days + 14)
            egx = reused.get('egx') if reused.get('lookback', 0) >= lookback else None
            if egx is None:
                egx = fetch_egx(asof, csvdir, egx_csv, lookback_days=lookback, min_sessions=6,
                                max_age_days=EGX_MAX_AGE if issue else HISTORY_MAX_AGE)
            for kind, (_, state) in jobs.items():
                update_recent_egx(state, egx)
                settle_pending(state, egx, asof, allow_unverified_void=True, label=GLOBAL_MOVE_MODELS[kind])
                if summarize(state, asof)['status'] == 'COMPLETE' and not state.get('completed_at_cairo'):
                    state['completed_at_cairo'] = asof.isoformat()

            latest = egx.index[-1].date().isoformat()
            learning = adaptive is not None and adaptive['trained_egx_last_date'] < latest
            if issue or learning:
                keys = {key for model, _ in jobs.values() for key in model['active_market_keys']}
                if keys <= reused.get('market_keys', set()) and reused.get('lookback', 0) >= lookback:
                    market_data = reused['market_data']
                else:
                    market_data = collect_market_closes(keys, asof, csvdir, lookback=lookback,
                                                        max_age_days=LIVE_MAX_AGE if issue else HISTORY_MAX_AGE)
                if book is None:
                    book = intraday.load_book(root)
                catch_up = learning and (egx.index[-1] - pd.Timestamp(adaptive['trained_egx_last_date'])).days > 3
                intraday.refresh_book(root, book, asof, csvdir, keys, full=lookback > 45, catch_up=catch_up)
                markets = market_data[0]
                snapshot_day = pd.Timestamp(asof.date())
                features, rows = intraday.snapshot(book, markets, snapshot_day, asof, active)
                if issue or asof.time() >= FORECAST_CUTOFF:
                    intraday.save_snapshot(root, features, rows, asof)
                if learning:
                    try:
                        adaptive = advance_adaptive(root / 'adaptive', adaptive, egx, market_data, asof,
                                                    moves_book=book, label=GLOBAL_MOVE_MODELS['adaptive'])
                    except (OSError, RuntimeError, ValueError, KeyError) as exc:
                        print(f"{GLOBAL_MOVE_MODELS['adaptive']} learning will retry: {exc}", file=sys.stderr)
                if issue:
                    for kind in issuing:
                        model, state = jobs[kind]
                        if kind == 'adaptive':
                            model = adaptive
                            if model['trained_egx_last_date'] != latest:
                                print(f'{GLOBAL_MOVE_MODELS[kind]}: awaiting training through the last verified session.')
                                continue
                        live_forecast(state, model, asof, csvdir, publication_clock=publication_clock,
                                      market_data=market_data, label=GLOBAL_MOVE_MODELS[kind], global_data=(features, rows))

        for kind, (model, state) in jobs.items():
            current = adaptive if kind == 'adaptive' else model
            folder = root / kind
            changed = state != previous_states[kind]
            if changed:
                dump_atomic(folder / 'state.json', state)
            if changed or current != model or not (folder / 'report.json').is_file():
                write_reports(folder, state, current, asof)


def start_service(args):
    print('EGX30 five-method service: early data from 09:30, forecasts 09:50-09:59, opening results after 10:25, '
          'close results and daily training after 15:15 Cairo time.')
    root = Path(args.runtime)
    retries = {}
    start_time = now_cairo()
    deadline = datetime.combine(start_time.date(), args.until, tzinfo=CAIRO) if getattr(args, 'until', None) else None
    with trial_lock(root, '.service.lock'):
        try:
            while True:
                if deadline and now_cairo() >= deadline:
                    print(f'Scheduled service window ended at {deadline:%H:%M} Cairo.')
                    return
                shared = {}
                tasks = ((scheduled_openings, 'OPENING', {}), (scheduled_closes, 'CLOSE', {'shared': shared}),
                         (scheduled_global_moves, 'ALL-MARKET CLOSE', {'shared': shared}))
                for task, name, options in tasks:
                    asof = now_cairo()
                    if deadline and asof >= deadline:
                        break
                    try:
                        task(args, asof, retries, now_cairo, **options)
                    except (ValueError, OSError, RuntimeError, KeyError) as exc:
                        print(f'{name} service: {exc}. Automatic checks will retry.', file=sys.stderr)
                if args.once:
                    return
                pause = args.interval if deadline is None else min(args.interval, (deadline - now_cairo()).total_seconds())
                if pause > 0:
                    clock.sleep(pause)
        except KeyboardInterrupt:
            print('Service stopped. Forecasts and training history are saved.')


def global_move_status(runtime):
    root = runtime / 'global_moves'
    for kind, label in GLOBAL_MOVE_MODELS.items():
        folder = root / kind
        if (folder / 'model.json').is_file() and (folder / 'state.json').is_file():
            model, state = load_trial(folder)
            stats = summarize(state)
            print(f"{label}: trained through {model['trained_egx_last_date']}, "
                  f"{stats['correct']}/{stats['evaluated_predictions']} live results correct.")
            print_historical_scores(model)
    intraday.print_snapshot(root)


def status(args):
    runtime = Path(args.runtime)
    if not any((runtime / name).exists() for name in ('state.json', 'model.json')):
        print('NOT INITIALIZED: python app.py init')
        opening.print_report(runtime / 'opening')
        return
    model, state = load_trial(runtime)
    s = summarize(state)
    print('FROZEN MODEL trained:' if model.get('frozen', True) else 'ADAPTIVE MODEL trained:', model['trained_at_cairo'])
    print(f"EGX sessions trained: {model['egx_prices_in_history']} prices / {model['egx_target_days']} labels")
    print('Last training EGX session:', model['trained_egx_last_date'])
    active = set(model.get('active_market_keys', (m.key for m in MARKETS)))
    print(f'Trained markets: {len(active)}/{len(MARKETS)}')
    print('Indices:', ', '.join(m.exchange for m in MARKETS if m.key in active))
    print('Last EGX close tracked:', state['recent_egx'][-1]['date'])
    print(f"LIVE TRIAL: correct={s['correct']} incorrect={s['incorrect']} evaluated={s['evaluated_predictions']} "
          f"pending={s['pending']} issued={s['issued_predictions']}")
    print('TRIAL MODE:', s['trial_mode'], 'first forecast:', s['first_forecast_date'],
          'end-exclusive date:', s['end_exclusive_date'])
    print('LIVE ACCURACY:', f"{s['accuracy']:.2%}" if s['accuracy'] is not None else 'Not measured')
    print('Status:', s['status'])
    print_historical_scores(model)
    print('Country weight ranking (association, not causality):')
    for row in model['country_weights']:
        print(f"  {row['country']:<15} {row['model_weight']:.1%}")
    adaptive_root = runtime / 'adaptive'
    if (adaptive_root / 'model.json').is_file():
        adaptive, adaptive_state = load_trial(adaptive_root)
        adaptive_stats = summarize(adaptive_state)
        print(f"ADAPTIVE CLOSE trained through {adaptive['trained_egx_last_date']}, "
              f"{adaptive.get('learning_updates', 0)} daily updates, live "
              f"{adaptive_stats['correct']}/{adaptive_stats['evaluated_predictions']} correct.")
        print_historical_scores(adaptive)
    opening.print_report(runtime / 'opening')
    global_move_status(runtime)


def mark_void(args, asof):
    rt = Path(args.runtime)
    with trial_lock(rt):
        model, state = load_trial(rt)
        matches = [p for p in state['predictions'] if p['date'] == args.date and p['status'] == 'PENDING']
        if len(matches) != 1:
            raise RuntimeError('No pending prediction for requested date')
        if pd.Timestamp(args.date).date() >= asof.date():
            raise RuntimeError('Cannot mark current/future date as a confirmed holiday')
        matches[0].update(status='VOID', void_kind='HOLIDAY',
                          void_reason='Confirmed EGX trading holiday/no session, manually verified',
                          void_at_cairo=asof.isoformat())
        dump_atomic(rt / 'state.json', state)
        write_reports(rt, state, model, asof)
    print(args.date, 'VOID. It does not count as correct/wrong or as a predicted trading session.')


def check_markets(args, asof):
    markets = {m.key: m for m in MARKETS}
    aliases = {m.csv_alias: m.key for m in MARKETS if m.csv_alias}
    requested = [aliases.get(key.upper(), key.upper()) for key in args.keys] or list(markets)
    unknown = sorted(set(requested) - markets.keys())
    if unknown:
        raise ValueError(f"Unknown indices: {', '.join(unknown)}. Valid keys: {', '.join([*markets, *aliases])}")
    errors = []
    for key in dict.fromkeys(requested):
        market = markets[key]
        try:
            prices = fetch_market(market, asof, Path(args.csv_dir) if args.csv_dir else None,
                                  min_sessions=600, cache_dir=DEFAULT_MARKET_CACHE, prefer_cache=True,
                                  max_age_days=HISTORY_MAX_AGE)
        except Exception as exc:
            errors.append(key)
            print(f'{key} {market.exchange}: {exc}')
        else:
            print(f'{key} {market.exchange}: {len(prices)} closes, last {prices.index[-1].date()}')
    if errors:
        raise RuntimeError(f"600-session history unavailable for: {', '.join(errors)}")


def add_shared_options(parser, on_subcommand=False):
    default = argparse.SUPPRESS if on_subcommand else None
    parser.add_argument('--runtime', default=argparse.SUPPRESS if on_subcommand else str(DEFAULT_RUNTIME),
                        help='Persistent frozen model and state folder')
    parser.add_argument('--csv-dir', default=default, help='Optional folder with EGX30.csv and market CSVs')
    parser.add_argument('--egx-csv', default=default, help='Path to an EGX30 Date,Close CSV')


def make_parser():
    parser = argparse.ArgumentParser(description='EGX30 direction forecasts and a frozen prospective trial')
    add_shared_options(parser)
    sub = parser.add_subparsers(dest='command', required=True)
    initp = sub.add_parser('init', help='First-time collect JSON (or reuse existing) and train exactly once')
    initp.add_argument('--history', default=str(DEFAULT_HISTORY), help='600-session historical JSON path')
    initp.add_argument('--refresh-history', action='store_true', help='Collect current history even if the JSON exists')
    initp.add_argument('--force-reset', action='store_true', help='Danger: overwrite model and ERASE 30-day trial')
    initp.add_argument('--allow-missing-markets', action='store_true',
                       help='Explicitly allow training with >=10 real markets; freeze excluded weights at zero')
    initp.add_argument('--trial-mode', choices=['calendar', 'sessions'], default='calendar',
                       help='calendar: default 30 calendar days from first forecast; sessions: 30 settled forecasts')
    startp = sub.add_parser('start', help='Initialize once and run all five methods on the Cairo schedule')
    startp.add_argument('--history', default=str(DEFAULT_HISTORY), help='Seed history JSON path')
    startp.add_argument('--allow-missing-markets', action='store_true',
                        help='Allow first training with at least 10 real indices if some feeds are unavailable')
    startp.add_argument('--trial-mode', choices=['calendar', 'sessions'], default='calendar',
                        help='Frozen trial duration for a new runtime')
    startp.add_argument('--interval', type=int, default=60, help='Check the Cairo schedule every 60 seconds')
    startp.add_argument('--until', type=lambda value: datetime.strptime(value, '%H:%M').time(),
                        metavar='HH:MM', help='Stop starting new work at this Cairo time today')
    startp.add_argument('--once', action='store_true', help='Check the schedule once and exit')
    sub.add_parser('once', help='One check: settle post-close forecasts and predict if 09:50–09:59 Cairo')
    watchp = sub.add_parser('watch', help='Keep running (PC/VPS must remain powered)')
    watchp.add_argument('--interval', type=int, default=60, help='Poll interval in seconds, default 60')
    sub.add_parser('status', help='Read stored progress without internet or historical JSON')
    sub.add_parser('opening-report', help='Read early-market opening counts and opening-trial results')
    sub.add_parser('intraday-report', help='Read the full latest-move snapshot and both all-market model scores')
    sub.add_parser('check-egx', help='Diagnose current EGX30 data source without training or changing the frozen model')
    indices = sub.add_parser('check-markets', help='Check availability of 600 recent closes for selected indices')
    indices.add_argument('keys', nargs='*', metavar='KEY', help='Example: SP500 DOW FTSE IBEX. Default: all indices')
    saudip = sub.add_parser('check-saudi', help='Check 600 Saudi TASI closes (TradingView first), no training')
    saudip.add_argument('--source', choices=('auto','tradingview','yahoo','tasi_api'),
                        default='auto', help='Saudi source override for this diagnostic only')
    voidp = sub.add_parser('void', help='Mark verified Egyptian exchange holiday (does not count)')
    voidp.add_argument('--date', required=True, help='YYYY-MM-DD; only past pending dates')
    for command_parser in sub.choices.values():
        add_shared_options(command_parser, on_subcommand=True)
    return parser


def watch(args):
    runtime = Path(args.runtime)
    load_trial(runtime)  # Check initialization before waiting for a forecast window.
    print('Watching until the trial ends. Keep the computer powered on. Ctrl+C stops the watcher.')
    last_postclose_check = None
    try:
        while True:
            asof = now_cairo()
            state = read_json(runtime / 'state.json')
            stats = summarize(state, asof)
            reached_end = stats['status'] == 'COMPLETE'
            today = asof.date().isoformat()
            preopen = (schedule_candidate(asof) is not None
                       and stats['remaining_to_issue'] != 0
                       and (not stats['end_exclusive_date'] or today < stats['end_exclusive_date'])
                       and not any(p['date'] == today for p in state['predictions']))
            postclose = (stats['pending'] > 0 and asof.hour < 21
                         and market_time_passed(asof.date(), 'Africa/Cairo', EGX_CLOSE.strftime('%H:%M'), asof)
                         and (last_postclose_check is None or asof - last_postclose_check >= timedelta(minutes=20)))
            if reached_end or preopen or postclose:
                try:
                    if postclose:
                        last_postclose_check = asof
                    report = daily_once(args, asof)
                    if report['status'] == 'COMPLETE':
                        print('Trial complete. Reports saved.')
                        return
                except (OSError, RuntimeError, ValueError) as exc:
                    print('ERROR (next scheduled poll will retry):', exc, file=sys.stderr)
            clock.sleep(args.interval)
    except KeyboardInterrupt:
        print('Watcher stopped. Model and results are saved.')


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    asof = now_cairo()
    if args.command == 'init':
        init(args, asof)
    elif args.command == 'start':
        if not 20 <= args.interval <= 300:
            parser.error('--interval must be between 20 and 300 seconds')
        start_service(args)
    elif args.command == 'status':
        status(args)
    elif args.command == 'opening-report':
        opening.print_report(Path(args.runtime) / 'opening')
    elif args.command == 'intraday-report':
        global_move_status(Path(args.runtime))
    elif args.command == 'check-markets':
        check_markets(args, asof)
    elif args.command == 'check-saudi':
        os.environ['SAUDI_SOURCE'] = args.source
        m = next(m for m in MARKETS if m.key == 'SAUDI')
        p = fetch_market(m, asof, Path(args.csv_dir) if args.csv_dir else None,
                         lookback_days=1350, min_sessions=600,
                         cache_dir=DEFAULT_MARKET_CACHE, prefer_cache=False)
        if (asof.date() - p.index[-1].date()).days > 14:
            raise RuntimeError(f'SAUDI history is stale (last session {p.index[-1].date()}); refuse training.')
        print(f'SAUDI: {len(p)} real completed sessions; '
              f'first={p.index[0].date()}, last={p.index[-1].date()}; READY')
        print(f'SAUDI cache location: {cache_file_for(m, DEFAULT_MARKET_CACHE)}')
    elif args.command == 'check-egx':
        print('Checking for a fresh EGX30 source with at least 600 closes...')
        prices = fetch_egx(asof, Path(args.csv_dir) if args.csv_dir else None,
                           Path(args.egx_csv) if args.egx_csv else None, lookback_days=1350,
                           min_sessions=600, max_age_days=HISTORY_MAX_AGE)
        latest = prices.index[-1].date()
        age = (asof.date() - latest).days
        print(f'EGX30 feed: {len(prices)} finalized sessions, first={prices.index[0].date()}, '
              f'last={latest}, last close={prices.iloc[-1]:.2f}, age={age} calendar days. READY.')
    elif args.command == 'void':
        mark_void(args, asof)
    elif args.command == 'once':
        daily_once(args, asof)
    elif args.command == 'watch':
        if not 20 <= args.interval <= 300:
            parser.error('--interval must be between 20 and 300 seconds')
        watch(args)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print('ERROR:', error, file=sys.stderr)
        sys.exit(1)
