# MLB HR Dashboard ⚾

A Streamlit app that pulls live MLB Stats API and Baseball Savant data to build a
per-game home run "prop read" board: composite scores, Statcast quality-of-contact
metrics, and a heatmap lineup table.

**Data sources (both free, no API key required):**
- [MLB Stats API](https://statsapi.mlb.com)
- [Baseball Savant](https://baseballsavant.mlb.com) (Statcast leaderboard CSV export)

> TrueHRScore, MatchupScore, ZoneFit, and HR Form are composite estimates built from
> public inputs, normalized within each night's lineup. They approximate — but do not
> replicate — any specific commercial model. Park factor is a static historical
> approximation, not weather-adjusted. **Not betting advice.**

## Setup

1. Clone the repo and move into it:
   ```bash
   git clone <your-repo-url>
   cd mlb-hr-dashboard
   ```

2. (Recommended) create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run

```bash
streamlit run app.py
```

Streamlit will open the app in your browser (default: `http://localhost:8501`).

## Usage

1. Pick a date — the app loads that day's MLB schedule.
2. Pick a game from the dropdown.
3. Toggle "away"/"home" to view hitters for either team, matched up against the
   opposing probable pitcher.
4. Review the Top Reads cards and the full lineup board with Statcast metrics.

If Baseball Savant's feed doesn't respond, the app still runs on MLB Stats API data
alone — Statcast-only columns (Barrel%, xwOBA, Hard Hit%, Pull%, Sweet Spot%) will
show as `—`.

## Notes

- Data is cached for 5–10 minutes (`st.cache_data`) to avoid hammering the public APIs.
- Park factors are static historical approximations hardcoded in `PARK_HR_FACTOR`.
- No API keys or secrets are required to run this app.

## License

Add a license of your choice (e.g. MIT) if you plan to make this repo public.
