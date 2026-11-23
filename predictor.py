#!/usr/bin/env python3
"""Historical, point-in-time global-market direction model for Egypt's EGX30.

Research tool. Not financial advice or an assurance of predictive accuracy.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CAIRO = ZoneInfo("Africa/Cairo")
EGX_OPEN = time(10, 0)  # Review this constant if EGX officially changes its opening time.
EGX_CLOSE = time(14, 30)
FORECAST_CUTOFF = time(9, 50)  # 10 min before the regular EGX session.
PUBLICATION_BUFFER_MINUTES = 30  # Conservative proxy, NOT a historical vendor timestamp.
MODEL_VERSION = 2
TRAINING_HALF_LIFE = 252  # Sessions. Fixed before evaluating the model.
RECENT_BASELINE_SESSIONS = 60
SIGNAL_THRESHOLD = 0.60
MOMENTUM_SUFFIX = "_MOM_5"
EGX_FEATURE_COLUMNS = ("EGX_PREV_1", "EGX_PREV_5", "EGX_VOL_5")
LEGACY_MARKET_KEYS = frozenset({"NASDAQ", "NYSE", "SHANGHAI", "JAPAN", "EURONEXT", "SHENZHEN",
                                "HONG_KONG", "CANADA", "TAIWAN", "BSE", "NSE", "KOREA", "GERMANY",
                                "SWITZERLAND", "SAUDI"})


@dataclass(frozen=True)
class Market:
    key: str
    country: str
    exchange: str
    ticker: str
    timezone: str
    close: str
    csv_alias: str | None = None

    @property
    def csv_name(self) -> str:
        return f"{self.csv_alias or self.key}.csv"

    def csv_override(self, csv_dir: Path | None) -> Path | None:
        if csv_dir is None:
            return None
        for name in (self.csv_name, f"{self.key}.csv"):
            path = csv_dir / name
            if path.is_file():
                return path
        return None


# Market indices used as EGX30 inputs. NYSE is a legacy key for S&P 500.
MARKETS = [
    Market("NASDAQ", "USA", "Nasdaq US", "^IXIC", "America/New_York", "16:00"),
    Market("NYSE", "USA", "S&P 500", "^GSPC", "America/New_York", "16:00", "SP500"),
    Market("SHANGHAI", "China", "Shanghai", "000001.SS", "Asia/Shanghai", "15:00"),
    Market("JAPAN", "Japan", "Japan Exchange Group", "^N225", "Asia/Tokyo", "15:30"),
    Market("EURONEXT", "Europe", "Euronext (CAC 40 proxy)", "^FCHI", "Europe/Paris", "17:30"),
    Market("SHENZHEN", "China", "Shenzhen", "399001.SZ", "Asia/Shanghai", "15:00"),
    Market("HONG_KONG", "Hong Kong", "HKEX", "^HSI", "Asia/Hong_Kong", "16:00"),
    Market("CANADA", "Canada", "TMX", "^GSPTSE", "America/Toronto", "16:00"),
    Market("TAIWAN", "Taiwan", "TWSE", "^TWII", "Asia/Taipei", "13:30"),
    Market("BSE", "India", "BSE", "^BSESN", "Asia/Kolkata", "15:30"),
    Market("NSE", "India", "NSE", "^NSEI", "Asia/Kolkata", "15:30"),
    Market("KOREA", "South Korea", "KRX", "^KS11", "Asia/Seoul", "15:30"),
    Market("GERMANY", "Germany", "Deutsche Boerse", "^GDAXI", "Europe/Berlin", "17:30"),
    Market("SWITZERLAND", "Switzerland", "SIX", "^SSMI", "Europe/Zurich", "17:30"),
    Market("SAUDI", "Saudi Arabia", "Tadawul", "^TASI.SR", "Asia/Riyadh", "15:00"),
    Market("DOW", "USA", "Dow Jones Industrial Average", "^DJI", "America/New_York", "16:00"),
    Market("FTSE", "United Kingdom", "FTSE 100", "^FTSE", "Europe/London", "16:35"),
    Market("IBEX", "Spain", "IBEX 35", "^IBEX", "Europe/Madrid", "17:35"),
]

ADDED_MARKET_KEYS = frozenset({"DOW", "FTSE", "IBEX"})


def now_cairo():
    return datetime.now(tz=CAIRO)


def local_datetime(day: pd.Timestamp | datetime, hhmm: str, timezone: str) -> pd.Timestamp:
    hour, minute = map(int, hhmm.split(":"))
    d = pd.Timestamp(day).date()
    return pd.Timestamp(datetime(d.year, d.month, d.day, hour, minute, tzinfo=ZoneInfo(timezone)))


def parse_daily_csv(path: Path, timezone: str | None = None, *, require_open=False) -> pd.DataFrame:
    """Read ordinary date-based or TradingView-exported daily prices.

    TradingView can export Unix seconds/milliseconds or ISO 8601 UTC times.
    Date-only CSV values are treated as exchange session dates. UTC datetime
    values are converted to the local exchange trading day when supplied.
    """
    df = pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")
    names = {str(c).strip().lower(): c for c in df.columns}
    date_field = next((names[c] for c in ("date", "datetime", "datetime_utc", "time", "timestamp", "trading_date") if c in names), None)
    open_field = names.get("open")
    close_names = ("close",) if require_open else ("close", "adj close", "adjusted close", "price", "last", "value")
    close_field = next((names[c] for c in close_names if c in names), None)
    if date_field is None or close_field is None or (require_open and open_field is None):
        expected = "Date,Open,Close" if require_open else "Date,Close"
        raise ValueError(f"{path}: need {expected} daily columns. Found: {list(df.columns)}")
    raw_dates = df[date_field].astype("string").str.strip()
    is_epoch = raw_dates.str.fullmatch(r"\d{9,13}(?:\.\d+)?", na=False)
    has_zone = raw_dates.str.contains(r"(?:Z|UTC|[+-]\d{2}:?\d{2})$", case=False, na=False)
    local_rows = ~(is_epoch | has_zone)
    dates = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if local_rows.any():
        local_dates = pd.to_datetime(raw_dates[local_rows], format="mixed", errors="coerce", dayfirst=False)
        if local_dates.notna().mean() < .7:
            local_dates = pd.to_datetime(raw_dates[local_rows], errors="coerce", dayfirst=True, format="mixed")
        dates.loc[local_rows] = local_dates
    if has_zone.any():
        utc_dates = pd.to_datetime(raw_dates[has_zone], utc=True, errors="coerce", format="mixed")
        dates.loc[has_zone] = utc_dates.dt.tz_convert(timezone or "UTC").dt.tz_localize(None)
    if is_epoch.any():
        epochs = pd.to_numeric(raw_dates[is_epoch], errors="coerce").astype(float)
        seconds = epochs.where(epochs <= 10**11, epochs / 1000)
        utc_dates = pd.to_datetime(seconds, unit="s", errors="coerce", utc=True)
        dates.loc[is_epoch] = utc_dates.dt.tz_convert(timezone or "UTC").dt.tz_localize(None)
    prices = {"Close": df[close_field].astype(str).str.replace(",", "", regex=False).to_numpy()}
    if open_field is not None:
        prices["Open"] = df[open_field].astype(str).str.replace(",", "", regex=False).to_numpy()
    return clean_daily_prices(pd.DataFrame(prices, index=pd.DatetimeIndex(dates)))


def parse_close_csv(path: Path, timezone: str | None = None) -> pd.Series:
    return clean_close(parse_daily_csv(path, timezone)["Close"])


def clean_daily_prices(prices: pd.DataFrame) -> pd.DataFrame:
    result = prices.reindex(columns=["Open", "Close"]).apply(pd.to_numeric, errors="coerce")
    result = result.where(np.isfinite(result) & (result > 0)).dropna(how="all")
    result.index = pd.DatetimeIndex(result.index).tz_localize(None).normalize()
    return result.loc[result.index.notna()].sort_index(kind="stable").astype(float)


def clean_close(series: pd.Series) -> pd.Series:
    result = series.copy()
    result.index = pd.DatetimeIndex(result.index).tz_localize(None).normalize()
    result = pd.to_numeric(result, errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan).dropna()
    result = result[(result > 0) & result.index.notna()].sort_index(kind="stable")
    result = result[~result.index.duplicated(keep="last")]
    return result.astype(float)


def yahoo_daily_prices(ticker: str, start: datetime, end: datetime, *, required_field='Open') -> pd.DataFrame:
    """Fetch raw daily opening and closing prices from Yahoo's chart endpoint."""
    start_day, end_day = start.date(), end.date()
    if start_day >= end_day:
        raise ValueError(f"Yahoo {ticker}: end date must follow the start date")
    # Pad the UTC request bounds, then enforce the interval by exchange date.
    first = datetime.combine(start_day - timedelta(days=1), time.min, tzinfo=timezone.utc)
    last = datetime.combine(end_day + timedelta(days=1), time.min, tzinfo=timezone.utc)
    params = urlencode({"period1": int(first.timestamp()), "period2": int(last.timestamp()), "interval": "1d"})
    failures = []
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        request = Request(f"https://{host}/v8/finance/chart/{quote(ticker, safe='')}?{params}",
                          headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(request, timeout=8) as response:
                chart = json.load(response)["chart"]
            if chart.get("error") or not chart.get("result"):
                raise ValueError(f"no daily chart data: {chart.get('error')}")
            result = chart["result"][0]
            meta = result["meta"]
            if str(meta.get("symbol", "")).casefold() != ticker.casefold() or meta.get("dataGranularity") != "1d":
                raise ValueError("response symbol or interval does not match the requested index")
            index = pd.to_datetime(result["timestamp"], unit="s", utc=True)
            index = pd.DatetimeIndex(index).tz_convert(ZoneInfo(meta["exchangeTimezoneName"])).tz_localize(None)
            quotes = result["indicators"]["quote"][0]
            prices = clean_daily_prices(pd.DataFrame({"Open": quotes.get("open"), "Close": quotes["close"]}, index=index))
            prices = prices[(prices.index >= pd.Timestamp(start_day)) & (prices.index < pd.Timestamp(end_day))]
            if prices[required_field].notna().sum() == 0:
                raise ValueError(f"no valid daily {required_field.lower()} prices in the requested date range")
            return prices
        except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            failures.append(f"{host}: {exc}")
    raise RuntimeError(f"Yahoo {ticker}: {'; '.join(failures)}")


def yahoo_close(ticker: str, start: datetime, end: datetime) -> pd.Series:
    return clean_close(yahoo_daily_prices(ticker, start, end, required_field='Close')["Close"])


def get_close(market: Market, csv_dir: Path | None, start: datetime, end: datetime) -> pd.Series:
    override = market.csv_override(csv_dir)
    if override:
        return parse_close_csv(override, timezone=market.timezone)
    return yahoo_close(market.ticker, start, end)


def available_returns(market: Market, closes: pd.Series, asof_now: datetime, delay_minutes: int) -> pd.DataFrame:
    """Timestamp returns and momentum at the scheduled close plus publication lag."""
    closes = clean_close(closes)
    returns = closes.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    momentum = closes.pct_change(5, fill_method=None).replace([np.inf, -np.inf], np.nan)
    delay = timedelta(minutes=delay_minutes)
    times = [local_datetime(day, market.close, market.timezone).tz_convert(CAIRO) + delay
             for day in returns.index]
    frame = pd.DataFrame({"available_at": pd.DatetimeIndex(times, tz=CAIRO), "return": returns.to_numpy(),
                          "momentum_5": momentum.reindex(returns.index).to_numpy(),
                          "session": returns.index})
    frame = frame[frame["available_at"] <= pd.Timestamp(asof_now)].sort_values("available_at")
    return frame.reset_index(drop=True)


def egypt_cutoffs(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"day": dates,
                         "cutoff": pd.DatetimeIndex([local_datetime(d, FORECAST_CUTOFF.strftime("%H:%M"), "Africa/Cairo")
                                                     for d in dates], tz=CAIRO)})


def market_feature_columns(keys) -> list[str]:
    """Return each market's one-session return and five-session momentum columns."""
    return [column for key in keys for column in (key, f"{key}{MOMENTUM_SUFFIX}")]


def assemble_market_features(days: pd.DatetimeIndex, all_closes: dict[str, pd.Series],
                             asof_now: datetime, delay_minutes: int, max_age_days: int = 8) -> pd.DataFrame:
    cutoffs = egypt_cutoffs(days)
    left = cutoffs[["cutoff"]].copy().sort_values("cutoff")
    restore_order = np.argsort(left.index)
    tolerance = timedelta(days=max_age_days)
    feature_frame = pd.DataFrame(index=days)
    for market in MARKETS:
        return_key, momentum_key = market_feature_columns([market.key])
        if market.key not in all_closes:
            feature_frame[return_key] = np.nan
            feature_frame[momentum_key] = np.nan
            continue
        right = available_returns(market, all_closes[market.key], asof_now, delay_minutes)
        if right.empty:
            feature_frame[return_key] = np.nan
            feature_frame[momentum_key] = np.nan
            continue
        joined = pd.merge_asof(left, right[["available_at", "return", "momentum_5"]],
                               left_on="cutoff", right_on="available_at", direction="backward",
                               tolerance=tolerance)
        feature_frame[return_key] = joined["return"].to_numpy()[restore_order]
        feature_frame[momentum_key] = joined["momentum_5"].to_numpy()[restore_order]
    return feature_frame


def egx_controls(days: pd.DatetimeIndex, close: pd.Series) -> pd.DataFrame:
    """Prior known EGX moves only: no use of the day's closing value."""
    close = clean_close(close)
    egx_r = close.pct_change(fill_method=None)
    rolling5 = close.pct_change(5, fill_method=None)
    volatility5 = egx_r.rolling(5).std(ddof=0)
    left = pd.DataFrame({"day": days}).sort_values("day")
    right = pd.DataFrame({"session": egx_r.index, "EGX_PREV_1": egx_r.to_numpy(),
                          "EGX_PREV_5": rolling5.to_numpy(), "EGX_VOL_5": volatility5.to_numpy()})
    # Strictly earlier sessions, including when forecasting after the last date in the CSV.
    match = pd.merge_asof(left, right, left_on="day", right_on="session", direction="backward",
                          allow_exact_matches=False)
    return match.set_index("day")[list(EGX_FEATURE_COLUMNS)].reindex(days)


def make_model() -> Pipeline:
    # Coefficients are interpretable only as model reliance, NOT causal country influence.
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scaler", StandardScaler()),
                     ("classifier", LogisticRegression(C=0.2, max_iter=3000))])


def fit_direction_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Fit the same regularized model for live prediction and historical evaluation."""
    ages = np.arange(len(X) - 1, -1, -1)
    weights = np.exp2(-ages / TRAINING_HALF_LIFE)
    weights /= weights.mean()
    return make_model().fit(X, y, scaler__sample_weight=weights, classifier__sample_weight=weights)


def walk_forward(X: pd.DataFrame, y: pd.Series, min_train: int, *, refit_every: int = 1,
                 close_only=None) -> pd.DataFrame:
    """Fit on earlier sessions and hold each fit fixed for the next test block."""
    if min_train < 2 or not X.index.is_monotonic_increasing or not X.index.is_unique or not X.index.equals(y.index):
        raise ValueError("Walk-forward requires aligned, unique, chronological rows and at least two training observations")
    if refit_every < 1:
        raise ValueError("Walk-forward refit interval must be one or more sessions")
    if len(X) < min_train + 50:
        raise ValueError(f"Need >= {min_train + 50} usable rows; got {len(X)}")
    if close_only is not None and not close_only.index.equals(X.index):
        raise ValueError('The comparison model requires the same training and test dates')
    # Original unweighted specification with one-day returns from the original
    # market feeds and the two EGX lag returns.
    legacy_features = LEGACY_MARKET_KEYS.union(EGX_FEATURE_COLUMNS[:2])
    legacy_base = X if close_only is None else close_only
    legacy_columns = [c for c in legacy_base.columns if c in legacy_features]
    legacy_X = legacy_base[legacy_columns] if legacy_columns else legacy_base
    predictions = []
    for start in range(min_train, len(X), refit_every):
        end = start + refit_every
        training_target = y.iloc[:start]
        if training_target.nunique() < 2:
            continue
        pipe = fit_direction_model(X.iloc[:start], training_target)
        probabilities = pipe.predict_proba(X.iloc[start:end])[:, 1]
        legacy = make_model().fit(legacy_X.iloc[:start], training_target)
        legacy_probabilities = legacy.predict_proba(legacy_X.iloc[start:end])[:, 1]
        if close_only is not None:
            comparator = fit_direction_model(close_only.iloc[:start], training_target)
            close_probabilities = comparator.predict_proba(close_only.iloc[start:end])[:, 1]
        else:
            close_probabilities = [None] * len(probabilities)
        recent_target = training_target.tail(RECENT_BASELINE_SESSIONS)
        baseline = float((training_target.sum() + 1) / (len(training_target) + 2))
        baseline_recent = float((recent_target.sum() + 1) / (len(recent_target) + 2))
        trained_through = training_target.index[-1]
        for (date, actual), p, legacy_p, close_p in zip(y.iloc[start:end].items(), probabilities,
                                                       legacy_probabilities, close_probabilities):
            row = {"date": date, "trained_through": trained_through, "actual_up": int(actual),
                   "probability_up": float(p), "predicted_up": int(p >= 0.5),
                   "baseline_majority": int(baseline >= 0.5), "baseline_probability_up": baseline,
                   "recent_baseline_probability_up": baseline_recent, "legacy_model_probability_up": float(legacy_p)}
            if close_p is not None:
                row["close_only_probability_up"] = float(close_p)
            predictions.append(row)
    return pd.DataFrame(predictions).set_index("date") if predictions else pd.DataFrame()


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denominator = 1 + z*z/n
    midpoint = (p + z*z/(2*n))/denominator
    width = (z/denominator) * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return midpoint - width, midpoint + width


def evaluate(preds: pd.DataFrame) -> dict:
    if preds.empty:
        raise ValueError("Backtest yielded no predictions")
    actual = preds["actual_up"]
    p = preds["probability_up"]
    has_both_classes = actual.nunique() == 2
    metrics = {"test_days": len(preds), "accuracy": float(accuracy_score(actual, preds["predicted_up"])),
               "balanced_accuracy": float(balanced_accuracy_score(actual, preds["predicted_up"])) if has_both_classes else None,
               "majority_baseline_accuracy": float(accuracy_score(actual, preds["baseline_majority"])),
               "brier_score": float(brier_score_loss(actual, p)),
               "auc": float(roc_auc_score(actual, p)) if has_both_classes else None,
               "high_confidence_groups": {}}
    metrics["accuracy_gain_vs_majority"] = metrics["accuracy"] - metrics["majority_baseline_accuracy"]
    if "baseline_probability_up" in preds:
        majority_brier = float(brier_score_loss(actual, preds["baseline_probability_up"]))
        metrics["majority_baseline_brier"] = majority_brier
        metrics["brier_skill_vs_majority"] = 1 - metrics["brier_score"] / majority_brier if majority_brier else None
    if "recent_baseline_probability_up" in preds:
        recent_prob = preds["recent_baseline_probability_up"]
        metrics["recent_baseline_accuracy"] = float(accuracy_score(actual, recent_prob >= .5))
        metrics["recent_baseline_brier"] = float(brier_score_loss(actual, recent_prob))
    if "legacy_model_probability_up" in preds:
        legacy_prob = preds["legacy_model_probability_up"]
        legacy_brier = float(brier_score_loss(actual, legacy_prob))
        metrics["legacy_model_accuracy"] = float(accuracy_score(actual, legacy_prob >= .5))
        metrics["legacy_model_brier"] = legacy_brier
        metrics["accuracy_gain_vs_legacy"] = metrics["accuracy"] - metrics["legacy_model_accuracy"]
        metrics["brier_skill_vs_legacy"] = 1 - metrics["brier_score"] / legacy_brier if legacy_brier else None
    if "close_only_probability_up" in preds:
        close_prob = preds["close_only_probability_up"]
        metrics["close_only_accuracy"] = float(accuracy_score(actual, close_prob >= .5))
        metrics["close_only_brier"] = float(brier_score_loss(actual, close_prob))
    for threshold in (0.55, SIGNAL_THRESHOLD, 0.65, 0.70):
        sel = preds[np.maximum(p, 1 - p) >= threshold]
        hits = int((sel["predicted_up"] == sel["actual_up"]).sum())
        lo, hi = wilson_interval(hits, len(sel))
        metrics["high_confidence_groups"][f"{threshold:.2f}"] = {
            "observations": len(sel), "coverage": len(sel) / len(preds),
            "accuracy": hits / len(sel) if len(sel) else None,
            "accuracy_95pct_wilson_interval": [lo, hi] if len(sel) else None}
    signal = metrics["high_confidence_groups"][f"{SIGNAL_THRESHOLD:.2f}"]
    metrics.update(signal_threshold=SIGNAL_THRESHOLD, signal_coverage=signal["coverage"],
                   signal_accuracy=signal["accuracy"], signal_observations=signal["observations"],
                   signal_accuracy_interval=signal["accuracy_95pct_wilson_interval"])
    return metrics


def country_weights(model: Pipeline, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    coef = model.named_steps["classifier"].coef_[0]
    coefficients = dict(zip(cols, coef))
    rows = []
    for m in MARKETS:
        daily = float(coefficients.get(m.key, 0.0))
        momentum = float(coefficients.get(f"{m.key}{MOMENTUM_SUFFIX}", 0.0))
        rows.append({"exchange": m.exchange, "key": m.key, "country": m.country, "ticker": m.ticker,
                     "model_weight": abs(daily) + abs(momentum), "signed_standardized_coefficient": daily,
                     "signed_momentum_standardized_coefficient": momentum})
    exchange_weights = pd.DataFrame(rows)
    total = exchange_weights["model_weight"].sum()
    if total:
        exchange_weights["model_weight"] /= total
    exchange_weights = exchange_weights.sort_values("model_weight", ascending=False)
    countries = exchange_weights.groupby("country", as_index=False)["model_weight"].sum().sort_values("model_weight", ascending=False)
    return exchange_weights, countries


def next_egx_weekday(date: pd.Timestamp) -> pd.Timestamp:
    # Egypt: Sunday..Thursday; exceptional EGX holidays must be verified separately.
    day = date + timedelta(days=1)
    while day.dayofweek in (4, 5):  # Friday/Saturday
        day += timedelta(days=1)
    return day


def default_forecast_day(now: datetime) -> pd.Timestamp:
    d = pd.Timestamp(now.date())
    if d.dayofweek in (4, 5):
        while d.dayofweek in (4, 5):
            d += timedelta(days=1)
        return d
    cutoff_close = local_datetime(d, EGX_CLOSE.strftime("%H:%M"), "Africa/Cairo")
    return next_egx_weekday(d) if pd.Timestamp(now) >= cutoff_close else d


def load_all(args, asof_now: datetime) -> tuple[pd.Series, dict[str, pd.Series], list[str]]:
    start = asof_now - timedelta(days=max(1350, args.days * 3))
    end = asof_now + timedelta(days=2)
    csv_dir = Path(args.csv_dir) if args.csv_dir else None
    egx_override = Path(args.egx_csv) if args.egx_csv else (csv_dir / "EGX30.csv" if csv_dir else None)
    if egx_override and egx_override.is_file():
        egx = parse_close_csv(egx_override)
        print(f"EGX30: imported {len(egx)} rows from {egx_override}")
    else:
        try:
            egx = yahoo_close("^CASE30", start, end)
        except Exception as exc:
            raise RuntimeError("EGX30 Yahoo data unavailable. Supply --egx-csv path/to/EGX30.csv ") from exc
        print(f"EGX30: downloaded {len(egx)} rows (^CASE30)")
    if len(egx) < args.days // 2:
        raise RuntimeError(f"EGX30 has just {len(egx)} observations. Yahoo may only serve one recent bar. "
                           "Provide a longer EGX30 daily history using --egx-csv.")
    all_closes = {}
    warnings = []
    for market in MARKETS:
        try:
            closes = get_close(market, csv_dir, start, end)
            if len(closes) < 80:
                raise ValueError(f"only {len(closes)} daily prices")
            all_closes[market.key] = closes
            print(f"  {market.key:12s} {len(closes):5d} daily prices")
        except Exception as exc:
            warnings.append(f"{market.key} ({market.ticker}) missing: {str(exc)[:160]}")
            print("  WARNING:", warnings[-1])
    return egx, all_closes, warnings


def main():
    parser = argparse.ArgumentParser(description="Point-in-time EGX30 directional model")
    parser.add_argument("--days", type=int, default=600, help="600 EGX trading sessions by default")
    parser.add_argument("--window", choices=["sessions", "calendar"], default="sessions",
                        help="sessions = last 600 EGX trading days, calendar = last 600 calendar days")
    parser.add_argument("--min-train", type=int, default=300, help="initial expanding-window observations")
    parser.add_argument("--egx-csv", type=str, default=None, help="EGX30 CSV with Date,Close columns")
    parser.add_argument("--csv-dir", type=str, default=None,
                        help="optional folder of EGX30.csv and market KEY.csv files; Yahoo used for missing files")
    parser.add_argument("--forecast-date", type=str, default=None, help="YYYY-MM-DD, EGX local session")
    parser.add_argument("--delay-minutes", type=int, default=PUBLICATION_BUFFER_MINUTES,
                        help="conservative lag after foreign closing time (default 30)")
    parser.add_argument("--out", default="output", help="where CSV/JSON results are written")
    args = parser.parse_args()
    if args.days < 200 or args.min_train < 100 or args.delay_minutes < 0:
        parser.error("days must be >=200, min-train >=100 and delay-minutes >=0")
    now = now_cairo()
    print(f"Analysis as of {now:%Y-%m-%d %H:%M %Z}, no post-cutoff returns are eligible.")
    egx, all_closes, warnings = load_all(args, now)
    closed = egx.loc[[local_datetime(d, EGX_CLOSE.strftime("%H:%M"), "Africa/Cairo") + timedelta(minutes=30)
                      <= pd.Timestamp(now) for d in egx.index]]
    if args.window == "calendar":
        closed = closed.loc[closed.index >= (pd.Timestamp(now.date()) - timedelta(days=args.days))]
    else:
        closed = closed.tail(args.days + 6)  # extra prior data for lagged target/EGX control
    if len(closed) < args.min_train + 55:
        raise RuntimeError(f"Only {len(closed)} closed EGX30 sessions; need >= {args.min_train + 55}. "
                           "Provide more observations or reduce --min-train.")
    days = closed.index[1:]
    X = assemble_market_features(days, all_closes, now, args.delay_minutes)
    X = X.join(egx_controls(days, closed))
    y = (closed.pct_change(fill_method=None).loc[days] > 0).astype(int)
    # Keep only reasonably covered feature columns, never hallucinate unavailable markets.
    present = [m.key for m in MARKETS if X[m.key].notna().mean() >= 0.65]
    if len(present) < 5:
        raise RuntimeError(f"Only {len(present)} markets with adequate point-in-time coverage. "
                           "Need 5 or more. Check Yahoo restrictions or import CSVs.")
    cols = market_feature_columns(present) + list(EGX_FEATURE_COLUMNS)
    X = X[cols]
    if len(X) > args.days:
        X, y = X.tail(args.days), y.loc[X.tail(args.days).index]
    print(f"Using {len(X)} EGX sessions; {len(present)}/{len(MARKETS)} index feeds available")
    preds = walk_forward(X, y, args.min_train)
    metrics = evaluate(preds)
    final_model = fit_direction_model(X, y)
    exchange_weights, countries = country_weights(final_model, cols)
    if len(present) != len(MARKETS):
        warnings.append(f"Coverage warning: only {len(present)} of {len(MARKETS)} markets included; missing weights = 0.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out / "walk_forward_predictions.csv", index_label="date")
    exchange_weights.to_csv(out / "market_weights.csv", index=False)
    countries.to_csv(out / "country_weights.csv", index=False)

    forecast_date = pd.Timestamp(args.forecast_date) if args.forecast_date else default_forecast_day(now)
    info = None
    forecast_cut = local_datetime(forecast_date, FORECAST_CUTOFF.strftime("%H:%M"), "Africa/Cairo")
    if forecast_date <= closed.index[-1]:
        warnings.append(f"Requested {forecast_date.date()} is within the training history; no live forecast output.")
    elif forecast_date.dayofweek in (4, 5):
        warnings.append("Forecast date falls on Friday/Saturday (Egyptian market normally closed).")
    elif forecast_cut > pd.Timestamp(now):
        warnings.append("Forecast cutoff 09:50 Cairo has not occurred: do not treat a future snapshot as known.")
    else:
        nextx = assemble_market_features(pd.DatetimeIndex([forecast_date]), all_closes, now,
                                         args.delay_minutes).join(egx_controls(pd.DatetimeIndex([forecast_date]), closed))
        if sum(pd.notna(nextx.loc[forecast_date, k]) for k in present) < 5:
            warnings.append("Insufficient recently available market returns for live forecast.")
        else:
            p = float(final_model.predict_proba(nextx[cols])[0, 1])
            info = {"date": str(forecast_date.date()), "cutoff_cairo": str(forecast_cut),
                    "probability_up": p, "probability_down": 1-p,
                    "direction": "UP" if p >= .5 else "DOWN",
                    "signal": "STRONGER" if max(p, 1-p) >= SIGNAL_THRESHOLD else "WEAK_OR_UNCLEAR"}

    report = {"generated_at_cairo": now.isoformat(), "window": args.window, "requested_days": args.days,
              "egypt_sessions_used": len(X), "training_start": str(X.index.min().date()),
              "training_end": str(X.index.max().date()), "markets_available": len(present),
              "included_market_keys": present, "backtest": metrics, "forecast": info,
              "warnings": warnings,
              "disclaimer": "Experimental historical probabilities; weights are predictive coefficients, not causality. "
                            "Daily market closes only, no same-day intraday foreign market moves."}
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nWALK-FORWARD BACKTEST (OUT OF SAMPLE)")
    print(f"Days: {metrics['test_days']} | Accuracy: {metrics['accuracy']:.1%} | "
          f"Baseline: {metrics['majority_baseline_accuracy']:.1%} | "
          f"Brier: {metrics['brier_score']:.4f} | AUC: {metrics['auc']}")
    print("\nCOUNTRY MODEL RELIANCE (not causal influence)")
    print(countries.to_string(index=False, formatters={"model_weight": "{:.1%}".format}))
    if info:
        print(f"\n{info['date']} signal: {info['direction']} | P(up)={info['probability_up']:.1%} "
              f"P(down)={info['probability_down']:.1%} | {info['signal']}")
    else:
        print("\nNo forecast published for selected day; see warnings in report.json")
    for w in warnings:
        print("WARNING:", w)
    print(f"\nResults saved to: {out.resolve()}")


if __name__ == "__main__":
    print("This is the legacy research model library, not the frozen 30-day runner.")
    print("Use: python app.py init   (first time); python app.py watch   (daily); python app.py status")
