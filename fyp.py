import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.stattools import pacf
from xgboost import XGBRegressor
import re
from sklearn.cluster import KMeans
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, Easter
import streamlit as st
import plotly.graph_objects as go
from datetime import date
from entsoe import EntsoePandasClient
import requests
from io import StringIO
from streamlit_autorefresh import st_autorefresh
import os
import time

# Set proxy for this Python session
os.environ['HTTP_PROXY'] = 'http://172.30.10.11:3128'
os.environ['HTTPS_PROXY'] = 'http://172.30.10.11:3128'

st.set_page_config(
    page_title="Austria Load Forecast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st_autorefresh(interval=1 * 60 * 1000, key="dataframerefresh")


plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

#%config InlineBackend.figure_format = 'retina'



    

def make_lags(ts, lagfirst, laglast):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(lagfirst, laglast + 1)
        },
        axis=1)


#d = pd.read_csv("C:/Users/as comp/Downloads/AT_2023.csv")
#e = pd.read_csv("C:/Users/as comp/Downloads/AT_2024.csv")

def smooth_between_zero_crossings(X, Pmax=1500, Emax=10000, dt=0.25, tol=20.0, max_iter=60):
    """
    Returns X with added columns:
      - pflow : raw flow = sw - load_actual
      - base  : found daily base (constant for smoothing interval)
      - SS    : smoothed storage flow = pflow + base (constrained later to [-Pmax, Pmax])
      - SoC   : battery state (kWh), continuous across days, clipped to [0, Emax]
      - peak_curve : load_actual - base
    Notes:
      - dt: hours per timestep (default 1.0). Adjust if your index timestep differs.
      - tol: acceptable energy mismatch (kWh) between available and required for that day.
    """
    X = X.copy()

    # Basic checks
    if not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("X must have DatetimeIndex")
    if "sw" not in X.columns or "load_actual" not in X.columns:
        raise ValueError("X must contain sw and load_actual")

    # raw flow (power)
    X['pflow'] = X['sw'] - X['load_actual']

    # prepare result columns
    X['base'] = np.nan
    X['SS'] = np.nan
    X['SoC'] = np.nan
    X['peak_curve'] = np.nan

    E = 0.0  # continuous SoC (kWh)
    # iterate day by day using groupby on date
    for day, grp in X.groupby(X.index.date):
        # use grp (subset) for per-day calculations
        idx = grp.index

        # compute daily mn/mx and initial base
        mn = float(grp['load_actual'].min())
        mx = float(grp['load_actual'].max())
        # handle flat-day
        if mx == mn:
            # no crossing possible, choose trivial base = mn
            base = mn
            zh1 = idx[0]
            zh2 = idx[-1]
        else:
            base = 0.5 * (mn + mx)
            # e.g. 5% of daily range
            
            # helper to find zh1, zh2 given a base
            def find_zhs(base_val):
                zc = grp['pflow'] + base_val
                # sign change detection: zc * zc.shift(1) < 0
                sc = (zc * zc.shift(1)) < 0
                zero_hours = grp.index[sc.fillna(False)]
                zh1 = zh2 = None
                if len(zero_hours) > 0:
                    morning = [t for t in zero_hours if t.hour < 12]
                    afternoon = [t for t in zero_hours if t.hour >= 12]
                    if morning:
                        zh1 = pd.Timestamp(morning[0])
                    if afternoon:
                        zh2 = pd.Timestamp(afternoon[-1])
                    # fallback if we only got one crossing
                    if zh1 is None or zh2 is None:
                        if len(zero_hours) > 2:
                            zh1, zh2 = pd.Timestamp(zero_hours[0]), pd.Timestamp(zero_hours[-1])
                        elif zh1 is None:
                            zh1 = idx[0]
                        else:
                            zh2 = idx[-1]
                else:
                    zh1 = idx[0]
                    zh2 = idx[-1]
                return zh1, zh2

            # bisection search for base so that available_energy - required_energy in [0, tol]
            lo, hi = mn, mx
            zh1, zh2 = find_zhs(base)
            # compute ED and available helper
            def compute_ED_and_available(base_val):
                zh1_loc, zh2_loc = find_zhs(base_val)
                mask = (grp.index >= zh1_loc) & (grp.index <= zh2_loc)
                # raw_interval: pflow + base (unit: kW). Energy over timestep = * dt
                raw_interval = (grp.loc[mask, 'pflow'] + base_val)  # kW
                ED = (-raw_interval.clip(upper=0)).sum() * dt      # kWh required (deficit)
                # compute available before zh1: cumulative sum from start to zh1-1
                # cumulative energy from day's start until just before zh1
                before_mask = grp.index < zh1_loc
                if before_mask.any():
                    energy_before = (grp.loc[before_mask, 'pflow'] + base_val).cumsum().iloc[-1] * dt
                else:
                    energy_before = 0.0
                total_available = min(Emax, max(0.0, E + energy_before))  # E is SoC entering the day
                return ED, total_available, zh1_loc, zh2_loc

            ED, total_available, zh1, zh2 = compute_ED_and_available(base)

            lo = mn
            hi = mx
            
            it = 0
            fell_to_min = False
            
            while it < max_iter:
                diff = total_available - ED
            
                if 0 <= diff <= tol:
                    break
            
                # --- standard bisection update ---
                if diff < 0:
                    lo = base
                else:
                    hi = base
            
                base = 0.5 * (lo + hi)
            
                # --- detect collapse toward minimum ---
                if abs(base - mn) < 1e-2:          # converged to min
                    fell_to_min = True
                    break
            
                # recompute energy quantities
                ED, total_available, zh1, zh2 = compute_ED_and_available(base)
                it += 1
            
            
            # ------------------------------------------------------------------
            #     FALLBACK CASE: base converged to mn → rescue by flipping range
            # ------------------------------------------------------------------
            if fell_to_min:
            
                # reverse and expand lower bound
                # new_hi = old mn
                # new_lo = mn - (mx - mn)  (push lower)
                new_hi = mn
                new_lo = mn - (mx - mn)
            
                # reset
                lo = new_lo
                hi = new_hi
                it = 0
            
                # restart bisection using extended interval
                while it < max_iter:
                    base = 0.5 * (lo + hi)
            
                    ED, total_available, zh1, zh2 = compute_ED_and_available(base)
                    diff = total_available - ED
            
                    if 0 <= diff <= tol:
                        break
            
                    if diff < 0:
                        lo = base
                    else:
                        hi = base
            
                    it += 1


            # done bisection for this day
        
        # now we have base, zh1, zh2
        # assign base for all timestamps of the day
        X.loc[idx, 'base'] = base

        # compute SS for the day (power), then limit by Pmax
        SS_day = (grp['pflow'] + base)
        X.loc[idx, 'SS'] = SS_day.values
        X.loc[idx, 'peak_curve'] = (grp['load_actual'] - base).values

        # integrate SoC sample-by-sample to maintain per-step clipping
        for t in idx:
            flow = float(X.loc[t, 'SS'])  # kW
            E = E + flow * dt             # kWh
            E = max(0.0, min(E, Emax))
            X.loc[t, 'SoC'] = E

    return X

client = EntsoePandasClient(api_key=st.secrets["api_keys"]["entsoe"])
start1 = pd.Timestamp('20230101', tz='UTC')
end1 = pd.Timestamp('202506302359', tz='UTC')
start2 = pd.Timestamp('20250701', tz='UTC')
end2 = pd.Timestamp('now', tz='UTC')

@st.cache_data(ttl=7200, show_spinner=False)
def main_data():
    data = client.query_load(
        country_code='AT',
        start=start1, end=end1
    )
    data.index.rename('datetime', inplace = True)
    data.index = pd.to_datetime(data.index)
    data.index = data.index.tz_convert('UTC').tz_localize(None)


    data2 = client.query_load(
        country_code='AT',
        start=start2, end=end2
    )

    data2.index.rename('datetime', inplace = True)
    data2.index = pd.to_datetime(data2.index)
    data2.index = data2.index.tz_convert('UTC').tz_localize(None)




    url = "https://dataset.api.hub.geosphere.at/v1/station/historical/klima-v2-1h"

    params = {
        "start": start1,
        "end": end1,
        "station_ids": "5904",
        "parameters": "TL,RR,ff",
        "output_format": "csv"
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    # Load CSV directly into pandas

    weather = pd.read_csv(StringIO(r.text))
    weather.time.rename('datetime', inplace = True)
    weather.index = pd.to_datetime(weather.time)
    weather = weather.drop(['time', 'station'], axis = 1)
    weather_15 = weather.resample("15min").interpolate().tz_localize(None)


    params1 = {
        "start": start2,
        "end": end2,
        "station_ids": "5904",
        "parameters": "TL,RR,SO_H,FF",
        "output_format": "csv"
    }

    r1 = requests.get(url, params=params1)
    r1.raise_for_status()
    # Load CSV directly into pandas

    weather1 = pd.read_csv(StringIO(r1.text))
    weather1.time.rename('datetime', inplace = True)
    weather1.index = pd.to_datetime(weather1.time)
    weather2_15 = weather1.drop(['time', 'station'], axis = 1).tz_localize(None)

    API_KEY = st.secrets["api_keys"]["weather_api"]   # <-- put your Meteosource key here

    lat = 48.249   # Austrian station 5904 (your coords)
    lon = 16.356

    url = "https://www.meteosource.com/api/v1/free/point"

    params = {
        "lat": lat,
        "lon": lon,
        "sections": "hourly",
        "timezone": "CET",
        "language": "en",
        "units": "metric",
        "key": API_KEY
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    datum = r.json()

    # extract next 7 days hourly
    hours = datum["hourly"]["data"]

    df = pd.DataFrame(hours)

    # Convert time
    df["date"] = pd.to_datetime(df["date"])

    df["rr"] = df["precipitation"].apply(lambda x: x["total"] if isinstance(x, dict) else x)
    df['tl'] = df['temperature']
    df["ff"] = df["wind"].apply(lambda x: x["speed"] if isinstance(x, dict) else x)
    df.index = pd.to_datetime(df.date)

    weather_forecast = pd.concat([df.tl, df.ff, df.rr], axis=1)
    weather2_15 = pd.concat([weather2_15, weather_forecast], axis = 0)
    weather2_15 = weather2_15[~weather2_15.index.duplicated(keep='last')]
    weather2_15 = weather2_15.resample("15min").interpolate().tz_localize(None)



    daily = CalendarFourier(freq='D', order=3)      # captures 24h, 12h, 8h cycles
    weekly = CalendarFourier(freq='W', order=2)     # weekly + harmonic
    annual = CalendarFourier(freq='A', order=6)     # annual + semiannual + quarterly + smaller peaks




    dp = DeterministicProcess(
        index=data.index,
        constant=True,      # intercept
        order=1,            # linear trend
        seasonal=False,     # you already use Fourier → don't want dummy explosion
        additional_terms=[daily, weekly, annual],
        drop=True
    )

    X = dp.in_sample()

    today = pd.Timestamp.today(tz='UTC').normalize()   # e.g. 2025-12-12 00:00

    start = pd.Timestamp("2025-07-01 00:00", tz='UTC')
    end = today + pd.Timedelta(hours=23, minutes=45)

    # 15-min index
    idx = pd.date_range(start=start, end=end, freq="15min").tz_localize(None)

    dp = DeterministicProcess(
        index=idx,
        constant=True,      # intercept
        order=1,            # linear trend
        seasonal=False,     # you already use Fourier → don't want dummy explosion
        additional_terms=[daily, weekly, annual],
        drop=True
    )

    X_test = dp.in_sample()


    class AustriaHolidays(AbstractHolidayCalendar):
        rules = [
            Holiday("New Year", month=1, day=1),
            Holiday("Epiphany", month=1, day=6),
            Holiday("Easter Monday", month=1, day=1, offset=[Easter(), pd.DateOffset(days=1)]),
            Holiday("Labour Day", month=5, day=1),
            Holiday("Ascension Day", month=1, day=1, offset=[Easter(), pd.DateOffset(days=39)]),
            Holiday("Pentecost Monday", month=1, day=1, offset=[Easter(), pd.DateOffset(days=50)]),
            Holiday("Corpus Christi", month=1, day=1, offset=[Easter(), pd.DateOffset(days=60)]),
            Holiday("Assumption Day", month=8, day=15),
            Holiday("National Day", month=10, day=26),
            Holiday("All Saints Day", month=11, day=1),
            Holiday("Immaculate Conception", month=12, day=8),
            Holiday("Christmas Day", month=12, day=25),
            Holiday("St Stephen's Day", month=12, day=26),
        ]

    # Generate holidays for 2023–2024
    holidays = AustriaHolidays().holidays(start=start1, end=end1)

    # Add to your dataframe
    X["is_holiday"] = X.index.normalize().isin(holidays).astype(int)


    holidays2 = AustriaHolidays().holidays(start=start2, end=end2)

    # Add to your dataframe
    X_test["is_holiday"] = X_test.index.normalize().isin(holidays2).astype(int)

    # Saturday = 5, Sunday = 6
    X["weekend_flag"] = X.index.dayofweek.map({5: 1, 6: 2}).fillna(0).astype(int)
    X_test["weekend_flag"] = X_test.index.dayofweek.map({5: 1, 6: 2}).fillna(0).astype(int)

    # Align X to weather_15 timestamps
    common_index = X.index.intersection(weather_15.index)
    
    X = X.loc[common_index]
    weather_aligned = weather_15.loc[common_index]

    X["temp"] = weather_aligned["tl"]
    X["precip"] = weather_aligned["rr"]
    X["windspeed"] = weather_aligned["ff"]


    # Align X to weather_15 timestamps
    common_index2 = X_test.index.intersection(weather2_15.index)

    X_test = X_test.loc[common_index2]
    weather2_aligned = weather2_15.loc[common_index2]

    X_test["temp"] = weather2_aligned["tl"]
    X_test["precip"] = weather2_aligned["rr"]
    X_test["windspeed"] = weather2_aligned["ff"]


    y=data['Actual Load']
    common_index = y.index.intersection(X.index)
    y = y.loc[common_index]

    y_test=data2['Actual Load']
    common_index2 = y_test.index.intersection(X_test.index)
    y_test = y_test.loc[common_index2]
    y_test = y_test.astype(float)


    #X1 = make_lags(y, 24*4+4, 24*4+8).fillna(0.0)
    #X2 = make_lags(y, 48*4+1, 48*4+8).fillna(0.0)
    #X3 = make_lags(y, 72*4+1, 72*4+8).fillna(0.0)
    #x1 = make_lags(y_test, 24*4+4, 24*4+8).reindex(X_test.index, fill_value=0.0)
    #x2 = make_lags(y_test, 48*4+1, 48*4+8).reindex(X_test.index, fill_value=0.0)
    #x3 = make_lags(y_test, 72*4+1, 72*4+8).reindex(X_test.index, fill_value=0.0)
    #X = pd.concat([X,X1,X2,X3], axis = 1).fillna(0.0)
    #X_test = pd.concat([X_test,x1,x2,x3], axis = 1).fillna(0.0)
    #X = X.astype(float)
    #X_test = X_test.astype(float)


    model = XGBRegressor()
    model.fit(X, y)
    y_pred = pd.Series(model.predict(X_test), index=X_test.index)

    #data3 = pd.read_csv("C:/Users/as comp/Downloads/SW.csv")
    # AGGREGATE Forecast (Wind + Solar)
    data3 = client.query_wind_and_solar_forecast(
        country_code='AT',
        start=start,
        end=end
    )

    # AGGREGATE Actual (Wind + Solar)
    data4 = client.query_generation(
        country_code='AT',
        start=start2,
        end=end2,
    )
    data3.index.rename('datetime', inplace = True)
    data3.index = pd.to_datetime(data3.index).tz_convert('UTC').tz_localize(None)
    data3['sw'] = data3['Solar'] + data3['Wind Onshore']

    DATA = pd.concat([y_pred, data3['sw']], axis=1)
    DATA.columns = ['load_actual', 'sw']
    DATA = DATA.astype(float)

    X2 = smooth_between_zero_crossings(DATA)
    return { "y_pred": y_pred,
             "DATA": DATA,
             'renew':data3,
             "X2": X2,
             "weather_15": weather_15,
             "weather2_15": weather2_15,
             "df_forecast": df, }



@st.cache_data(ttl=300, show_spinner=False)
def run_pipeline():
    data2 = client.query_load(
        country_code='AT',
        start=start2, end=end2
    )

    data2.index.rename('datetime', inplace = True)
    data2.index = pd.to_datetime(data2.index)
    data2.index = data2.index.tz_convert('UTC').tz_localize(None)
    y_test=data2['Actual Load']
    return { "y_test": y_test}    


if "results" not in st.session_state:
    st.session_state.results = None

if st.session_state.results is None:
    # first run only
    with st.spinner("⏳ Initial load..."):
        st.session_state.results = {**run_pipeline(), **main_data()}
        st.session_state.last_update = pd.Timestamp.utcnow()
else:
    # background refresh
    with st.spinner("🔄 Updating in background..."):
        try:
            new_results = {**run_pipeline(), **main_data()}
            st.session_state.results = new_results
            st.session_state.last_update = pd.Timestamp.utcnow()
        except Exception:
            st.warning("Using last successful results")



results = st.session_state.results

if results is None:
    st.warning("Waiting for first successful data load...")
    st.stop()

y_pred = results["y_pred"]
y_test = results["y_test"]
DATA = results["DATA"]
X2 = results["X2"]
weather_15 = results["weather_15"]
weather2_15 = results["weather2_15"]
df = results["df_forecast"]
re_gen = results['renew']

st.caption(
    f"🕒 Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M UTC')}"
)




st.title("⚡ DSM Dashboard")
st.markdown("An interactive interface for **load forecasting, storage flow, and base load analysis**.")


# -------------------------------------------------------
# SIDEBAR: SELECT A DAY
# -------------------------------------------------------
all_days = sorted(list(set(y_pred.index.date)))
default_day = all_days[-1]

selected_day = st.sidebar.date_input( "Today", value=default_day,
                                     min_value=min(all_days),
                                     max_value=max(all_days)
                )
if "selected_day" not in st.session_state:
    st.session_state.selected_day = default_day

st.sidebar.markdown("### 📅 Select a day")

col_prev, col_picker, col_next = st.sidebar.columns([1, 3, 1])

with col_prev:
    if st.button("◀", disabled=st.session_state.selected_day <= min(all_days)):
        st.session_state.selected_day -= pd.Timedelta(days=1)
        if "selected_day" not in st.session_state:
            st.session_state.selected_day = default_day

with col_picker:
    picked = st.date_input( " ", value=st.session_state.selected_day,
                            min_value=min(all_days), max_value=max(all_days),
                            label_visibility="collapsed" ) 
    st.session_state.selected_day = picked
    if "selected_day" not in st.session_state:
        st.session_state.selected_day = default_day

with col_next:
    if st.button("▶", disabled=st.session_state.selected_day >= max(all_days)):
        st.session_state.selected_day += pd.Timedelta(days=1)
        if "selected_day" not in st.session_state:
            st.session_state.selected_day = default_day


selected_day = st.session_state.selected_day



# convert to timestamp day
day_str = pd.to_datetime(selected_day).strftime("%Y-%m-%d")

# filter daily data
day_pred = y_pred.loc[day_str]
day_actual = y_test.loc[day_str]
day_sw = DATA.loc[day_str, 'sw']
day_X2 = X2.loc[day_str]
if (selected_day != all_days[-1]):
    day_weather = weather2_15.loc[day_str]


    # -------------------------------------------------------
    # WEATHER PANEL
    # -------------------------------------------------------
    st.subheader("🌤️ Weather Panel")

    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    with col1:
       st.metric("🌡 Temperature", f"{day_weather['tl'].mean():.1f} °C")

    with col2:
        st.metric("🌧 Precipitation", f"{day_weather['rr'].sum():.1f} mm")

    with col3:
        st.metric("☀ Sunshine", f"{day_weather['so_h'].sum()/4:.1f} hour")
    
    with col4:
        st.metric("🌬 Average Wind Speed", f"{day_weather['ff'].mean():.1f} m/s")

    with col5:
        cond = "☀ Clear" if day_weather["so_h"].mean() > 50 else "☁ Cloudy"
        st.metric("Condition", cond)

else:
    day_weather = df
    # -------------------------------------------------------
    # WEATHER PANEL
    # -------------------------------------------------------
    now_utc = pd.Timestamp.utcnow().tz_localize(None)
    row = df.loc[
    (df["date"] - now_utc).abs().idxmin()
    ]

    st.subheader("🌤️ Weather Panel")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.metric(
        "🕒 Current UTC time",
        now_utc.strftime("%Y-%m-%d %H:%M"),
        selected_day.strftime("%A")
        )

    with col2:
        st.metric(
        "🌡 Temperature",
        f"{row['temperature']:.1f} °C"
        )

    with col3:
        st.metric(
        "🌬 Wind",
        f"{row['wind']['speed']:.1f} m/s",
        row['wind']["dir"])

    col4, col5, col6 = st.columns([1, 1, 1])

    with col4:
        cloud = row["cloud_cover"]["total"]
        st.metric(
        "☁ Cloud Cover",
        f"{cloud:.0f} %"
        )

    with col5:
        precip = row["precipitation"]
        value = f"{precip['total']:.1f} mm"

        if precip["type"] != "none":
            value += f" ({precip['type']})"

        st.metric(
        "🌧 Precipitation",
        value
        )
    
    with col6:
        st.metric(
        "🌤 Condition",
        row["summary"]
        )


if (selected_day != all_days[-1]):
    

    # -------------------------------------------------------
    # Power PANEL
    # -------------------------------------------------------
    st.subheader("⏻ Power Panel")

    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    with col1:
       st.metric("⏻ Load", f"{day_actual.sum():.1f} MWh", f"{day_pred.sum():.1f} MWh(forcasted)")

    with col2:
        st.metric("☀ PV Generation", f"{re_gen['Solar'].sum()/1000:.1f} MWh")

    with col3:
        st.metric("🌬 Wind Generation", f"{re_gen['Wind Onshore'].sum()/1000:.1f} MWh")
    
    with col4:
        st.metric("Base Generation", f"{X2['base'].mean():.1f} MW")

    

else:
    # -------------------------------------------------------
    # Use latest COMMON available time
    # -------------------------------------------------------
    t_load = day_actual.index.max()
    t_pred = day_pred.index.max()
    t_re   = re_gen.index.max()
    t_x2   = day_X2.index.max()

    safe_time = min(t_load, t_pred, t_re, t_x2)

    row  = day_actual.loc[safe_time]
    row1 = day_pred.loc[safe_time]
    row2 = re_gen.loc[safe_time]
    row3 = day_X2.loc[safe_time]

    st.subheader("⏻ Power Panel")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "⏻ Load",
            f"{row:.1f} MW",
            f"{row1:.1f} MW (forecasted)"
        )

    with col2:
        st.metric(
            "☀ PV Generation",
            f"{row2['Solar']:.1f} MW"
        )

    with col3:
        st.metric(
            "🌬 Wind Generation",
            f"{row2['Wind Onshore']:.1f} MW"
        )

    col4, col5, col6 = st.columns(3)
    if row3['SoC'] < 10000:
        with col4:
            st.metric(
                "Base Generation",
                f"{row3['base'] + (row1 - row):.0f} MW"
            )

        with col5:
            st.metric(
                "Powerflow in Storage",
                f"{row3['SS']:.0f} MW"
            )
    else:
        with col4:
            st.metric(
                "Base Generation",
                f"{row3['base'] + (row1 - row) - row3['SS']:.0f} MW"
            )
        with col5:
            st.metric(
                "Powerflow in Storage",
                f"{0:.0f} MW"
            )

    with col6:
        st.metric(
            "SoC",
            f"{row3['SoC']*100/10000:.0f} %"
        )


# -------------------------------------------------------
# PLOT 1: FORECASTED vs ACTUAL LOAD
# -------------------------------------------------------
st.subheader("📈 Forecasted vs Actual Load")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=day_actual.index, y=day_actual, mode='lines', name='Actual Load', line=dict(width=2)
))
fig1.add_trace(go.Scatter(
    x=day_pred.index, y=day_pred, mode='lines', name='Forecasted Load', line=dict(width=2)
))
fig1.update_layout(height=350, xaxis_title="Time", yaxis_title="MW")
st.plotly_chart(fig1, use_container_width=True)


# -------------------------------------------------------
# PLOT 2: STORAGE FLOW + SOC
# -------------------------------------------------------
st.subheader("🔋 Storage Flow & SoC")

fig2 = go.Figure()

# Energy Flow (SS)
ss = []
for i in range(24*4):
    if day_X2['SoC'].iloc[i] < 10000:
       ss.append(-day_X2['SS'].iloc[i])
    else: ss.append(0)

fig2.add_trace(go.Bar(
    x=day_X2.index,
    y=ss,
    name="Storage Flow (charge−, discharge+)"
))

# SoC Line
fig2.add_trace(go.Scatter(
    x=day_X2.index,
    y=day_X2['SoC']*100/10000,
    name="State of Charge %",
    yaxis="y2",
))

fig2.add_trace(go.Scatter(
    x=day_pred.index, y=day_X2['peak_curve'], mode='lines', name='Forecasted Load - Base Gen', line=dict(width=2), yaxis='y1'
))

fig2.add_trace(go.Scatter(
    x=day_pred.index, y=day_sw, mode='lines', name='Solar + Wind Generation', line=dict(width=2), yaxis='y1'
))

fig2.update_layout(
    height=350,
    xaxis=dict(title="Time"),
    yaxis=dict(title="Flow (MW)"),
    yaxis2=dict(
        title="%SoC",
        overlaying='y',
        side='right'
    )
)

st.plotly_chart(fig2, use_container_width=True)




# -------------------------------------------------------
# END
# -------------------------------------------------------
st.success("Dashboard updated! Select another day from the sidebar.")




