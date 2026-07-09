"""
MLB HR Dashboard
A Streamlit app that pulls live MLB Stats API + Baseball Savant data to build
a per-game home run "prop read" board: composite scores, Statcast quality-of-contact
metrics, and a heatmap lineup table.

Data sources (both free, no API key required):
  - MLB Stats API   https://statsapi.mlb.com
  - Baseball Savant https://baseballsavant.mlb.com  (Statcast leaderboard CSV export)

Composite scores (TrueHRScore, MatchupScore, ZoneFit, HR Form) are estimates built
from public inputs, normalized within each night's lineup. They approximate — but do
not replicate — any specific commercial model. Not betting advice.
"""

import io
from datetime import date as date_cls, datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st

API_BASE = "https://statsapi.mlb.com/api/v1"
SAVANT_URL = (
    "https://baseballsavant.mlb.com/leaderboard/custom"
    "?year={year}&type=batter&min=1"
    "&selections=barrel_batted_rate,hard_hit_percent,xwoba,xwobacon,"
    "sweet_spot_percent,pull_percent,iso,launch_angle_avg"
    "&chart=false&csv=true"
)
SAVANT_PITCHER_URL = (
    "https://baseballsavant.mlb.com/leaderboard/custom"
    "?year={year}&type=pitcher&min=1"
    "&selections=whiff_percent,o_swing_percent,oz_swing_percent,k_percent,bb_percent,"
    "f_strike_percent,called_strike_percent,pitches,called_strikes,whiffs"
    "&chart=false&csv=true"
)

# Rough static historical HR park-factor approximations (1.00 = neutral).
# Not live/weather-adjusted — just a reasonable seed value per home park.
PARK_HR_FACTOR = {
    "COL": 1.28, "CIN": 1.15, "NYY": 1.12, "BAL": 1.10, "TEX": 1.08, "PHI": 1.07,
    "MIL": 1.05, "CHC": 1.04, "BOS": 1.02, "ARI": 1.02, "TOR": 1.01, "HOU": 1.00,
    "MIN": 1.00, "ATL": 0.99, "LAA": 0.99, "WSH": 0.98, "CWS": 0.98, "KC": 0.97,
    "STL": 0.97, "LAD": 0.97, "NYM": 0.96, "CLE": 0.95, "TB": 0.94, "DET": 0.94,
    "PIT": 0.93, "SEA": 0.92, "SD": 0.90, "SF": 0.88, "MIA": 0.90, "OAK": 0.93,
    "ATH": 0.93,
}

st.set_page_config(page_title="MLB HR Dashboard", layout="wide", page_icon="⚾")


# ----------------------------- Data fetchers ------------------------------ #

@st.cache_data(ttl=300, show_spinner=False)
def get_schedule(date_str: str):
    r = requests.get(
        f"{API_BASE}/schedule",
        params={"sportId": 1, "date": date_str, "hydrate": "probablePitcher,team"},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    games = []
    for g in data.get("dates", [{}])[0].get("games", []):
        games.append(
            {
                "key": f"{g['teams']['away']['team']['id']}-{g['teams']['home']['team']['id']}",
                "away": g["teams"]["away"]["team"],
                "home": g["teams"]["home"]["team"],
                "away_pitcher": g["teams"]["away"].get("probablePitcher"),
                "home_pitcher": g["teams"]["home"].get("probablePitcher"),
                "venue": g.get("venue", {}).get("name", ""),
                "game_time": g.get("gameDate"),
            }
        )
    return games


@st.cache_data(ttl=300, show_spinner=False)
def get_roster_with_stats(team_id: int, season: int):
    """Active roster hydrated with each player's season hitting stats in one call.

    (The old approach called /teams/{id}/stats expecting per-player rows, but that
    endpoint actually returns team-level totals — so it silently produced an empty
    season_stats dict and the board always showed 'no hitter data'.)
    """
    r = requests.get(
        f"{API_BASE}/teams/{team_id}/roster",
        params={
            "rosterType": "active",
            "hydrate": f"person(stats(group=hitting,type=season,season={season}))",
        },
        timeout=10,
    )
    r.raise_for_status()
    return r.json().get("roster", [])


def extract_season_stat(player_entry: dict):
    """Pull the season hitting stat dict out of a hydrated roster entry, or None."""
    for grp in player_entry.get("person", {}).get("stats", []):
        if grp.get("group", {}).get("displayName") == "hitting":
            splits = grp.get("splits", [])
            if splits:
                return splits[0].get("stat")
    return None


@st.cache_data(ttl=300, show_spinner=False)
def get_game_log(player_id: int, season: int):
    r = requests.get(
        f"{API_BASE}/people/{player_id}/stats",
        params={"stats": "gameLog", "group": "hitting", "season": season},
        timeout=10,
    )
    r.raise_for_status()
    splits = r.json().get("stats", [{}])[0].get("splits", [])
    splits.sort(key=lambda s: s.get("date", ""), reverse=True)
    return splits[:15]


@st.cache_data(ttl=300, show_spinner=False)
def get_vs_pitcher(batter_id: int, pitcher_id: int):
    try:
        r = requests.get(
            f"{API_BASE}/people/{batter_id}/stats",
            params={"stats": "vsPlayer", "opposingPlayerId": pitcher_id, "group": "hitting", "sportId": 1},
            timeout=10,
        )
        r.raise_for_status()
        splits = r.json().get("stats", [{}])[0].get("splits", [])
        return splits[0]["stat"] if splits else None
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def get_savant_leaderboard(season: int):
    """Returns a DataFrame keyed by MLBAM player_id, or None if unreachable."""
    try:
        r = requests.get(SAVANT_URL.format(year=season), timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        id_col = next((c for c in ("player_id", "mlbam_id", "id") if c in df.columns), None)
        if id_col is None:
            return None
        df = df.set_index(id_col)
        return df
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def get_pitcher_season_stats(pitcher_id: int, season: int):
    """Season pitching line (ERA, WHIP, HR/9, K/9, BB/9, IP, etc.) for one pitcher."""
    try:
        r = requests.get(
            f"{API_BASE}/people/{pitcher_id}/stats",
            params={"stats": "season", "group": "pitching", "season": season},
            timeout=10,
        )
        r.raise_for_status()
        splits = r.json().get("stats", [{}])[0].get("splits", [])
        return splits[0]["stat"] if splits else {}
    except Exception:
        return {}


@st.cache_data(ttl=600, show_spinner=False)
def get_savant_pitcher_leaderboard(season: int):
    """Returns a DataFrame of pitcher Statcast plate-discipline metrics, or None if unreachable."""
    try:
        r = requests.get(SAVANT_PITCHER_URL.format(year=season), timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        id_col = next((c for c in ("player_id", "mlbam_id", "id") if c in df.columns), None)
        if id_col is None:
            return None
        df = df.set_index(id_col)
        return df
    except Exception:
        return None


# ----------------------------- Helpers ------------------------------ #

def normalize(val, lo, hi):
    if val is None or pd.isna(val):
        return 0.5
    if hi == lo:
        return 0.5
    return min(1.0, max(0.0, (val - lo) / (hi - lo)))


def heat_style(pct):
    if pct is None or pd.isna(pct):
        return "background-color:#F3F4F6;color:#9AA5B1;"
    pct = min(1.0, max(0.0, pct))
    red, yellow, green = (233, 150, 122), (245, 224, 130), (140, 198, 140)
    if pct < 0.5:
        t = pct / 0.5
        c = tuple(int(red[i] + (yellow[i] - red[i]) * t) for i in range(3))
    else:
        t = (pct - 0.5) / 0.5
        c = tuple(int(yellow[i] + (green[i] - yellow[i]) * t) for i in range(3))
    return f"background-color:#{c[0]:02x}{c[1]:02x}{c[2]:02x};color:#1B2A41;"


def build_board(game: dict, view_team: str, season: int):
    batting_team = game["away"] if view_team == "away" else game["home"]
    pitching_side = game["home_pitcher"] if view_team == "away" else game["away_pitcher"]
    park_abbr = game["home"]["abbreviation"]
    park_factor = PARK_HR_FACTOR.get(park_abbr, 1.0)

    roster = get_roster_with_stats(batting_team["id"], season)
    hitter_ids = [p["person"]["id"] for p in roster if p.get("position", {}).get("abbreviation") != "P"]

    season_stats = {}
    for p in roster:
        stat = extract_season_stat(p)
        if stat:
            season_stats[p["person"]["id"]] = stat

    ranked = sorted(
        [pid for pid in hitter_ids if pid in season_stats],
        key=lambda pid: season_stats[pid].get("plateAppearances", 0),
        reverse=True,
    )[:10]

    savant = get_savant_leaderboard(season)
    savant_ok = savant is not None

    rows = []
    progress = st.progress(0.0, text="Building board…")
    for i, pid in enumerate(ranked):
        info = next((p for p in roster if p["person"]["id"] == pid), None)
        stat = season_stats[pid]
        gl = get_game_log(pid, season)
        vs_p = get_vs_pitcher(pid, pitching_side["id"]) if pitching_side else None

        sv_row = savant.loc[pid] if (savant is not None and pid in savant.index) else None

        def sv(col):
            if sv_row is None or col not in sv_row:
                return None
            try:
                return float(sv_row[col])
            except (TypeError, ValueError):
                return None

        barrel = sv("barrel_batted_rate")
        hard_hit = sv("hard_hit_percent")
        xwoba = sv("xwoba")
        xwobacon = sv("xwobacon")
        sweet_spot = sv("sweet_spot_percent")
        pull = sv("pull_percent")
        la = sv("launch_angle_avg")
        iso = sv("iso")
        if iso is None or pd.isna(iso):
            slg, avg = stat.get("slg"), stat.get("avg")
            try:
                iso = (float(slg) - float(avg)) if slg and avg else 0.0
            except (TypeError, ValueError):
                iso = 0.0

        season_hr = float(stat.get("homeRuns", 0))
        season_games = float(stat.get("gamesPlayed", 1)) or 1.0
        season_hr_rate = season_hr / season_games
        recent_hr = sum(float(g.get("stat", {}).get("homeRuns", 0)) for g in gl)
        recent_games = len(gl) or 1
        recent_hr_rate = recent_hr / recent_games
        hr_form_pct = round(
            min(100, max(0, ((recent_hr_rate * 0.65 + season_hr_rate * 0.35) / 0.06) * 100))
        )
        if recent_hr_rate > season_hr_rate * 1.15:
            hr_trend = "up"
        elif recent_hr_rate < season_hr_rate * 0.75:
            hr_trend = "down"
        else:
            hr_trend = "flat"

        rows.append(
            {
                "id": pid,
                "name": (info.get("person", {}) or {}).get("fullName") or f"Player #{pid}" if info else "Unknown",
                "pos": info.get("position", {}).get("abbreviation", "") if info else "",
                "barrel": barrel,
                "hard_hit": hard_hit,
                "xwoba": xwoba,
                "xwobacon": xwobacon,
                "sweet_spot": sweet_spot,
                "pull": pull,
                "la": la,
                "iso": iso,
                "hr_form_pct": hr_form_pct,
                "hr_trend": hr_trend,
                "vs_pitcher_ab": vs_p.get("atBats") if vs_p else None,
                "vs_pitcher_hr": vs_p.get("homeRuns") if vs_p else None,
            }
        )
        progress.progress((i + 1) / max(1, len(ranked)), text=f"Building board… {i+1}/{len(ranked)}")
    progress.empty()

    df = pd.DataFrame(rows)
    if df.empty:
        return df, park_factor, savant_ok

    def col_range(col):
        vals = df[col].dropna()
        if vals.empty:
            return (0.0, 1.0)
        return (float(vals.min()), float(vals.max() if vals.max() != vals.min() else vals.min() + 0.001))

    b_lo, b_hi = col_range("barrel")
    i_lo, i_hi = col_range("iso")
    h_lo, h_hi = col_range("hard_hit")
    p_lo, p_hi = col_range("pull")
    x_lo, x_hi = col_range("xwoba")
    xc_lo, xc_hi = col_range("xwobacon")

    n_barrel = df["barrel"].apply(lambda v: normalize(v, b_lo, b_hi))
    n_iso = df["iso"].apply(lambda v: normalize(v, i_lo, i_hi))
    n_hard = df["hard_hit"].apply(lambda v: normalize(v, h_lo, h_hi))
    n_pull = df["pull"].apply(lambda v: normalize(v, p_lo, p_hi))
    n_xwoba = df["xwoba"].apply(lambda v: normalize(v, x_lo, x_hi))
    n_xwobacon = df["xwobacon"].apply(lambda v: normalize(v, xc_lo, xc_hi))

    df["zone_fit"] = (n_barrel * 0.18 + n_pull * 0.12).round(3)
    df["matchup_score"] = (
        (n_xwobacon * 0.4 + n_barrel * 0.35 + n_hard * 0.25) * 100 * (0.9 + 0.2 * park_factor - 0.1)
    ).clip(upper=99.9).round(1)
    df["true_hr_score"] = (
        (n_barrel * 0.3 + n_iso * 0.25 + n_hard * 0.15 + n_xwoba * 0.15 + (df["hr_form_pct"] / 100) * 0.15)
        * 100
        * park_factor
    ).clip(upper=99.9).round(1)

    df = df.sort_values("true_hr_score", ascending=False).reset_index(drop=True)
    return df, park_factor, savant_ok


def trend_arrow(trend):
    return {"up": "▲", "down": "▼", "flat": "→"}.get(trend, "")


def trend_color(trend):
    return {"up": "#3A8F5C", "down": "#C6483C", "flat": "#9AA5B1"}.get(trend, "#9AA5B1")


def render_top_reads(df, batting_abbr, pitching_abbr):
    top4 = df.head(4)
    cols = st.columns(len(top4)) if len(top4) else []
    for col, (_, p) in zip(cols, top4.iterrows()):
        with col:
            st.markdown(
                f"""
                <div style="background:white;border:1px solid #E4E7EC;border-radius:12px;padding:16px;">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
                    <div>
                      <div style="font-weight:600;color:#1B2A41;">{p['name']}</div>
                      <div style="font-size:11px;color:#6B7789;font-family:monospace;">{batting_abbr} vs {pitching_abbr}</div>
                    </div>
                    <div style="font-family:'Sora',sans-serif;font-weight:800;font-size:24px;color:#E8622C;">{p['true_hr_score']:.1f}</div>
                  </div>
                  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;text-align:center;">
                    {mini_stat("Matchup", f"{p['matchup_score']:.1f}")}
                    {mini_stat("ZoneFit", f"{p['zone_fit']:.3f}")}
                    {mini_stat("HR Form", f"{p['hr_form_pct']}% <span style='color:{trend_color(p['hr_trend'])}'>{trend_arrow(p['hr_trend'])}</span>")}
                    {mini_stat("Pulled Brl", f"{p['pull']:.1f}%" if pd.notna(p['pull']) else "—")}
                    {mini_stat("Brl/BIP", f"{p['barrel']:.1f}%" if pd.notna(p['barrel']) else "—")}
                    {mini_stat("ISO", f"{p['iso']:.3f}" if pd.notna(p['iso']) else "—")}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def mini_stat(label, value):
    return (
        f'<div style="background:#F6F7FA;border-radius:8px;padding:6px 2px;">'
        f'<div style="font-size:9px;text-transform:uppercase;color:#9AA5B1;margin-bottom:2px;">{label}</div>'
        f'<div style="font-family:monospace;font-weight:600;font-size:13px;color:#1B2A41;">{value}</div>'
        f'</div>'
    )


def render_lineup_table(df):
    # Columns to heat-map, relative to tonight's lineup (min-max normalized).
    # LA is included for visual consistency but "hotter" here just means
    # "higher relative to this lineup" — it isn't a goodness signal like the others.
    heat_cols = ["zone_fit", "iso", "xwoba", "xwobacon", "pull", "barrel", "sweet_spot", "hard_hit", "la"]
    ranges = {c: (df[c].min(skipna=True), df[c].max(skipna=True)) for c in heat_cols}

    def badge(val, fmt_text):
        lo, hi = ranges[val[0]]
        raw = val[1]
        pct = normalize(raw, lo, hi) if pd.notna(raw) else None
        text = "—" if pd.isna(raw) else fmt_text
        return f"<td style='padding:6px 10px;border-top:1px solid #EEF0F3;text-align:right;font-family:monospace;{heat_style(pct)}'>{text}</td>"

    rows_html = ""
    for _, r in df.iterrows():
        vs_p = f"{int(r['vs_pitcher_hr'] or 0)}HR/{int(r['vs_pitcher_ab'])}AB" if pd.notna(r["vs_pitcher_ab"]) else "—"
        rows_html += f"""
        <tr>
          <td style="padding:6px 10px;border-top:1px solid #EEF0F3;">
            <div style="font-weight:500;color:#1B2A41;font-size:13px;">{r['name']}</div>
            <div style="font-size:10px;color:#9AA5B1;font-family:monospace;">{r['pos']}</div>
          </td>
          <td style="padding:6px 10px;border-top:1px solid #EEF0F3;text-align:right;font-family:monospace;font-weight:600;{heat_style(r['true_hr_score']/100)}">{r['true_hr_score']:.1f}</td>
          <td style="padding:6px 10px;border-top:1px solid #EEF0F3;text-align:right;font-family:monospace;{heat_style(r['matchup_score']/100)}">{r['matchup_score']:.1f}</td>
          {badge(('zone_fit', r['zone_fit']), f"{r['zone_fit']:.3f}")}
          <td style="padding:6px 10px;border-top:1px solid #EEF0F3;text-align:right;font-family:monospace;color:#1B2A41;">{r['hr_form_pct']}% <span style="color:{trend_color(r['hr_trend'])}">{trend_arrow(r['hr_trend'])}</span></td>
          {badge(('iso', r['iso']), f"{r['iso']:.3f}")}
          {badge(('xwoba', r['xwoba']), f"{r['xwoba']:.3f}")}
          {badge(('xwobacon', r['xwobacon']), f"{r['xwobacon']:.3f}")}
          {badge(('pull', r['pull']), f"{r['pull']:.1f}%")}
          {badge(('barrel', r['barrel']), f"{r['barrel']:.1f}%")}
          {badge(('sweet_spot', r['sweet_spot']), f"{r['sweet_spot']:.1f}%")}
          {badge(('hard_hit', r['hard_hit']), f"{r['hard_hit']:.1f}%")}
          {badge(('la', r['la']), f"{r['la']:.1f}")}
          <td style="padding:6px 10px;border-top:1px solid #EEF0F3;text-align:right;font-family:monospace;color:#6B7789;">{vs_p}</td>
        </tr>
        """

    headers = [
        "Player", "TrueHRScore", "MatchupScore", "ZoneFit", "HR Form", "ISO", "xwOBA",
        "xwOBAcon", "PulledBrl", "Brl/BIP%", "SweetSpot%", "HardHit%", "LA", "vs Pitcher",
    ]
    head_html = "".join(
        f"<th style='padding:10px 12px;text-align:{'left' if h=='Player' else 'right'};font-size:11px;text-transform:uppercase;color:#6B7789;'>{h}</th>"
        for h in headers
    )

    st.markdown(
        f"""
        <div style="border:1px solid #E4E7EC;border-radius:12px;overflow-x:auto;background:white;">
          <table style="width:100%;border-collapse:collapse;font-size:14px;min-width:1200px;">
            <thead style="background:#F6F7FA;"><tr>{head_html}</tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _sv_get(row, candidates):
    """Pull the first matching column from a Savant leaderboard row (names shift slightly year to year)."""
    if row is None:
        return None
    for c in candidates:
        if c in row and pd.notna(row[c]):
            try:
                return float(row[c])
            except (TypeError, ValueError):
                continue
    return None


def render_pitcher_report(pitcher: dict, season: int, team_abbr: str, opp_abbr: str):
    stat = get_pitcher_season_stats(pitcher["id"], season)
    sv_df = get_savant_pitcher_leaderboard(season)
    sv_row = sv_df.loc[pitcher["id"]] if sv_df is not None and pitcher["id"] in sv_df.index else None

    era = stat.get("era", "—")
    whip = stat.get("whip", "—")
    hr9 = stat.get("homeRunsPer9", "—")
    k9 = stat.get("strikeoutsPer9Inn", "—")
    bb9 = stat.get("walksPer9Inn", "—")
    ip = stat.get("inningsPitched", "—")

    csw = _sv_get(sv_row, ["csw_percent"])
    if csw is None:
        called_strike_pct = _sv_get(sv_row, ["called_strike_percent"])
        whiff_pct_raw = _sv_get(sv_row, ["whiff_percent"])
        if called_strike_pct is not None and whiff_pct_raw is not None:
            csw = called_strike_pct + whiff_pct_raw
        else:
            pitches = _sv_get(sv_row, ["pitches"])
            called_strikes = _sv_get(sv_row, ["called_strikes"])
            whiffs = _sv_get(sv_row, ["whiffs", "swing_miss"])
            if pitches and called_strikes is not None and whiffs is not None:
                csw = (called_strikes + whiffs) / pitches * 100
    whiff = _sv_get(sv_row, ["whiff_percent"])
    chase = _sv_get(sv_row, ["oz_swing_percent", "o_swing_percent", "chase_percent"])
    k_pct = _sv_get(sv_row, ["k_percent"])
    bb_pct = _sv_get(sv_row, ["bb_percent"])
    kbb = f"{k_pct - bb_pct:.1f}%" if k_pct is not None and bb_pct is not None else "—"
    fstrike = _sv_get(sv_row, ["f_strike_percent"])

    def pct(v):
        return f"{v:.1f}%" if v is not None else "—"

    st.markdown(
        f"""
        <div style="background:white;border:1px solid #E4E7EC;border-radius:12px;padding:16px;margin-bottom:16px;">
          <div style="font-weight:700;font-size:16px;color:#1B2A41;">🎯 Pitcher Report — {pitcher['fullName']}</div>
          <div style="font-size:11px;color:#6B7789;font-family:monospace;margin-bottom:10px;">{opp_abbr} vs {team_abbr}</div>
          <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:6px;text-align:center;">
            {mini_stat("ERA", era)}
            {mini_stat("WHIP", whip)}
            {mini_stat("HR/9", hr9)}
            {mini_stat("K/9", k9)}
            {mini_stat("BB/9", bb9)}
            {mini_stat("IP", ip)}
            {mini_stat("CSW%", pct(csw))}
            {mini_stat("Whiff%", pct(whiff))}
            {mini_stat("Chase%", pct(chase))}
            {mini_stat("K-BB%", kbb)}
            {mini_stat("1st-Pitch K%", pct(fstrike))}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if sv_row is None:
        st.caption("Savant plate-discipline data (CSW%, Whiff%, Chase%, K-BB%, 1st-Pitch K%) wasn't available for this pitcher — showing MLB Stats API season line only.")


# ----------------------------- App layout ------------------------------ #

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; max-width: 1400px; }
      #MainMenu, footer, header { visibility: hidden; }
      .block-container div, .block-container td, .block-container th, .block-container span {
        color-scheme: light;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

top_l, top_r = st.columns([3, 2])
with top_l:
    st.markdown("### ⚾ MLB HR Dashboard")
with top_r:
    c1, c2 = st.columns(2)
    with c1:
        sel_date = st.date_input("Date", value=date_cls.today(), label_visibility="collapsed")
    with c2:
        pass

date_str = sel_date.strftime("%Y-%m-%d")
season = sel_date.year

games = get_schedule(date_str)

if not games:
    st.warning(f"No MLB games scheduled for {date_str}. Pick a different date.")
    st.stop()

game_labels = {g["key"]: f"{g['away']['abbreviation']} @ {g['home']['abbreviation']}" for g in games}
sel_key = st.selectbox("Game", options=list(game_labels.keys()), format_func=lambda k: game_labels[k])
game = next(g for g in games if g["key"] == sel_key)

view_team = st.radio(
    "Viewing hitters for",
    options=["away", "home"],
    format_func=lambda v: game["away"]["abbreviation"] if v == "away" else game["home"]["abbreviation"],
    horizontal=True,
)

batting_team = game["away"] if view_team == "away" else game["home"]
pitching_team = game["home"] if view_team == "away" else game["away"]
opposing_pitcher = game["home_pitcher"] if view_team == "away" else game["away_pitcher"]
park_factor = PARK_HR_FACTOR.get(game["home"]["abbreviation"], 1.0)

game_time = ""
if game["game_time"]:
    try:
        game_time = datetime.fromisoformat(game["game_time"].replace("Z", "+00:00")).strftime("%I:%M %p UTC")
    except ValueError:
        pass

st.markdown(
    f"""
    <div style="background:white;border:1px solid #E4E7EC;border-radius:12px;padding:18px;margin:14px 0;">
      <div style="font-weight:800;font-size:26px;">{game['away']['abbreviation']} @ {game['home']['abbreviation']}</div>
      <div style="color:#6B7789;font-family:monospace;font-size:13px;margin-top:4px;">
        {game_time} · {game['venue']} · <span style="color:#E8622C;font-weight:600;">Park factor {park_factor:.2f}×</span>
      </div>
      <div style="color:#6B7789;font-size:13px;margin-top:8px;">
        Showing <b>{batting_team['name']}</b> hitters vs <b>{opposing_pitcher['fullName'] if opposing_pitcher else 'TBD'}</b>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Loading lineup data…"):
    df, park_factor, savant_ok = build_board(game, view_team, season)

if opposing_pitcher:
    render_pitcher_report(opposing_pitcher, season, batting_team["abbreviation"], pitching_team["abbreviation"])

if not savant_ok:
    st.warning(
        "Baseball Savant's Statcast feed didn't respond. Showing scores from MLB Stats API data only — "
        "Barrel%, xwOBA, Hard Hit%, Pull%, and Sweet Spot% will read as '—' until that connects."
    )

if df.empty:
    st.info("No hitter data available for this matchup yet.")
    st.stop()

st.markdown("#### Top Reads In This Game")
render_top_reads(df, batting_team["abbreviation"], pitching_team["abbreviation"])

st.markdown(
    f"#### Lineup Board — {batting_team['abbreviation']} hitters vs "
    f"{opposing_pitcher['fullName'] if opposing_pitcher else 'TBD'}"
)
render_lineup_table(df)

st.caption(
    "TrueHRScore, MatchupScore, ZoneFit, and HR Form are composite estimates built from public "
    "Statcast/MLB Stats API inputs and normalized within tonight's lineup — they approximate, but do not "
    "replicate, any specific commercial model. Park factor is a static historical approximation, not "
    "weather-adjusted. Not betting advice."
)
