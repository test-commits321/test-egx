"""Bounded price requests through the optional TradingView data client."""
from __future__ import annotations

import logging
from contextlib import suppress
from threading import Timer, get_ident
from zoneinfo import ZoneInfo

import pandas as pd

from predictor import clean_daily_prices


def tradingview_bars(symbol: str, exchange: str, timezone: str, n_bars: int,
                     interval='in_daily') -> pd.DataFrame:
    try:
        from tvDatafeed import Interval, TvDatafeed
    except ImportError as exc:
        raise RuntimeError('TradingView reader missing: python -m pip install tradingview-datafeed') from exc

    thread = get_ident()
    messages = []

    def capture(record):
        if record.thread != thread or record.levelno < logging.ERROR:
            return True
        text = record.getMessage()
        if text not in messages:
            messages.append(text)
        return False  # The caller logs one actionable diagnostic and decides whether to retry.

    reader = TvDatafeed()
    code = f'{exchange}:{symbol}'

    def abort_request():
        socket = getattr(reader, 'ws', None)
        if socket is not None:
            with suppress(OSError, RuntimeError, AttributeError):
                socket.abort()

    logger = logging.getLogger('tvDatafeed.main')
    timeout = Timer(14, abort_request)
    timeout.daemon = True
    logger.addFilter(capture)
    try:
        timeout.start()
        frame = reader.get_hist(symbol=symbol, exchange=exchange, interval=getattr(Interval, interval),
                                n_bars=min(5000, max(10, n_bars)))
    except Exception as exc:
        raise RuntimeError(f'TradingView {code}: {exc}') from exc
    finally:
        timeout.cancel()
        logger.removeFilter(capture)
        socket = getattr(reader, 'ws', None)
        if socket is not None:
            with suppress(OSError, RuntimeError, AttributeError):
                socket.close(timeout=0.25)

    if messages:
        raise RuntimeError(f'TradingView {code}: {messages[0]}')
    if not isinstance(frame, pd.DataFrame) or frame.empty or 'close' not in frame.columns:
        raise ValueError(f'TradingView returned no usable candles for {code}')
    if 'symbol' in frame and not frame['symbol'].astype(str).str.upper().eq(code.upper()).all():
        raise ValueError(f'TradingView returned a different symbol from {code}')

    index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors='coerce'))
    zone = ZoneInfo(timezone)
    if index.tz is None:
        # tvDatafeed constructs naive timestamps in the host timezone.
        index = pd.DatetimeIndex([ts.to_pydatetime().astimezone(zone) if pd.notna(ts) else pd.NaT
                                  for ts in index], tz=zone)
    return frame.rename(columns={'open': 'Open', 'close': 'Close'}).set_axis(index.tz_convert(zone))


def tradingview_daily_prices(symbol: str, exchange: str, timezone: str, n_bars: int) -> pd.DataFrame:
    return clean_daily_prices(tradingview_bars(symbol, exchange, timezone, n_bars))
