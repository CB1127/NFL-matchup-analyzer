import datetime as dt
import warnings
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NFL Matchup Analyzer", page_icon="🏈", layout="wide")

# --------------------------- Configuration ---------------------------
TEAM_NAMES = {
    "ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens",
    "BUF":"Buffalo Bills","CAR":"Carolina Panthers","CHI":"Chicago Bears",
    "CIN":"Cincinnati Bengals","CLE":"Cleveland Browns","DAL":"Dallas Cowboys",
    "DEN":"Denver Broncos","DET":"Detroit Lions","GB":"Green Bay Packers",
    "HOU":"Houston Texans","IND":"Indianapolis Colts","JAX":"Jacksonville Jaguars",
    "KC":"Kansas City Chiefs","LA":"Los Angeles Rams","LAC":"Los Angeles Chargers",
    "LV":"Las Vegas Raiders","MIA":"Miami Dolphins","MIN":"Minnesota Vikings",
    "NE":"New England Patriots","NO":"New Orleans Saints","NYG":"New York Giants",
    "NYJ":"New York Jets","PHI":"Philadelphia Eagles","PIT":"Pittsburgh Steelers",
    "SEA":"Seattle Seahawks","SF":"San Francisco 49ers","TB":"Tampa Bay Buccaneers",
    "TEN":"Tennessee Titans","WAS":"Washington Commanders"
}
ALIASES = {"JAC":"JAX", "WSH":"WAS", "STL":"LA", "SD":"LAC", "OAK":"LV"}
POSITIONS = ["QB", "RB", "WR", "TE"]
STAT_LABELS = {
    "passing_yards":"Passing yards",
    "rushing_yards":"Rushing yards",
    "receiving_yards":"Receiving yards",
}
INJURY_MULTIPLIERS = {
    "Healthy / no limitation": 1.00,
    "Questionable": 0.90,
    "Doubtful": 0.75,
    "Out": 0.00,
    "Custom adjustment": 1.00,
}

st.markdown("""
<style>
.block-container{max-width:1180px;padding-top:1.1rem;padding-bottom:3rem}
.hero{padding:1.25rem 1.4rem;border-radius:18px;background:linear-gradient(120deg,#10243a,#1d4665);color:white;margin-bottom:1rem}
.hero h1{margin:0;font-size:2rem}.hero p{margin:.35rem 0 0;opacity:.86}
div[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.23);padding:12px;border-radius:14px}
.note{font-size:.83rem;opacity:.78}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🏈 NFL Matchup Analyzer</h1><p>Player production • Position-specific defense • Injury-aware projections</p></div>', unsafe_allow_html=True)

# --------------------------- Helpers ---------------------------
def first_col(df, names):
    return next((c for c in names if c in df.columns), None)

def as_num(df, names):
    c = first_col(df, names)
    return pd.to_numeric(df[c], errors="coerce").fillna(0) if c else pd.Series(0.0, index=df.index)

def norm_team(x):
    if pd.isna(x): return None
    x = str(x).strip().upper()
    return ALIASES.get(x, x)

def clean_pos(x):
    if pd.isna(x): return "UNK"
    x = str(x).upper().strip()
    if x in ("FB", "HB"): return "RB"
    if x in ("PK", "K", "P", "LS", "OL", "DL", "LB", "DB", "CB", "S", "NT", "DE", "DT", "OT", "OG", "C", "G", "T"): return x
    return x

@st.cache_data(ttl=1800, show_spinner=False)
def load_parquet(url):
    return pd.read_parquet(url)

@st.cache_data(ttl=1800, show_spinner=False)
def load_csv(url):
    return pd.read_csv(url, low_memory=False)

@st.cache_data(ttl=1800, show_spinner=False)
def try_load_injuries(season):
    urls = [
        "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries.csv",
        "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_weekly.csv",
    ]
    for url in urls:
        try:
            df = pd.read_csv(url, low_memory=False)
            if df is not None and len(df):
                return df
        except Exception:
            pass
    return pd.DataFrame()

def normalize_players(raw, season):
    d = raw.copy()
    required = {
        "player": first_col(d, ["player_display_name","player_name","name"]),
        "team": first_col(d, ["recent_team","team","team_abbr"]),
        "season": first_col(d, ["season"]),
        "week": first_col(d, ["week"]),
    }
    if any(v is None for v in required.values()):
        raise ValueError("Player feed is missing expected nflverse columns.")
    d["player"] = d[required["player"]].astype(str)
    d["team"] = d[required["team"]].map(norm_team)
    d["season_n"] = pd.to_numeric(d[required["season"]], errors="coerce")
    d["week_n"] = pd.to_numeric(d[required["week"]], errors="coerce")
    d["position"] = d[first_col(d, ["position","position_group"])].map(clean_pos) if first_col(d, ["position","position_group"]) else "UNK"
    d["opponent"] = d[first_col(d, ["opponent_team","opponent","opp"])].map(norm_team) if first_col(d, ["opponent_team","opponent","opp"]) else None
    d["game_id_n"] = d[first_col(d, ["game_id"])].astype(str) if "game_id" in d.columns else d["week_n"].astype(str)+"_"+d["team"].astype(str)
    for out, choices in {
        "passing_yards":["passing_yards","pass_yards"],
        "rushing_yards":["rushing_yards","rush_yards"],
        "receiving_yards":["receiving_yards","rec_yards"],
        "completions":["completions"],
        "attempts":["attempts","passing_attempts"],
        "interceptions":["interceptions","passing_interceptions"],
        "carries":["carries","rushing_attempts"],
        "receptions":["receptions"],
        "targets":["targets"],
        "passing_tds":["passing_tds","pass_touchdowns"],
        "rushing_tds":["rushing_tds","rush_touchdowns"],
        "receiving_tds":["receiving_tds","rec_touchdowns"],
    }.items():
        d[out] = as_num(d, choices)
    if "season_type" in d.columns:
        d = d[d["season_type"].astype(str).str.upper().isin(["REG","REGULAR","REGULAR SEASON"])]
    d = d[(d["season_n"] == season) & d["week_n"].notna()].copy()
    # Keep actual player-week rows and avoid duplicate rows when feeds contain multiple records.
    if "player_id" in d.columns:
        d = d.drop_duplicates(["player_id","game_id_n"], keep="last")
    else:
        d = d.drop_duplicates(["player","game_id_n"], keep="last")
    return d.sort_values(["week_n"])

def normalize_schedule(raw, season):
    d = raw.copy()
    date_c = first_col(d, ["gameday","game_date","date"])
    if not date_c or not {"home_team","away_team"}.issubset(d.columns):
        raise ValueError("Schedule feed missing expected columns.")
    d["date_n"] = pd.to_datetime(d[date_c], errors="coerce")
    d["home_n"] = d["home_team"].map(norm_team)
    d["away_n"] = d["away_team"].map(norm_team)
    if "season" in d.columns:
        d = d[pd.to_numeric(d["season"], errors="coerce") == season]
    if "game_type" in d.columns:
        d = d[d["game_type"].astype(str).str.upper().isin(["REG","REGULAR"])]
    return d[d["date_n"].notna()].copy()

def get_next_game(schedule, team, today):
    if not team: return None
    s = schedule[(schedule["date_n"].dt.date >= today) & ((schedule["home_n"] == team) | (schedule["away_n"] == team))].sort_values("date_n")
    if s.empty: return None
    g = s.iloc[0]
    home = g["home_n"] == team
    return {"opponent": g["away_n"] if home else g["home_n"], "date": g["date_n"], "venue": "Home" if home else "Away"}

def avg(s):
    return float(s.mean()) if len(s) else 0.0

def game_level_player(df):
    # Each row is a player's weekly line. Sum yards/usage across same player's game only if needed.
    return df.sort_values("week_n")

def player_window_stats(g):
    return {
        "games": int(g["game_id_n"].nunique()),
        "passing_yards": avg(g.groupby("game_id_n")["passing_yards"].sum()),
        "rushing_yards": avg(g.groupby("game_id_n")["rushing_yards"].sum()),
        "receiving_yards": avg(g.groupby("game_id_n")["receiving_yards"].sum()),
        "completions": avg(g.groupby("game_id_n")["completions"].sum()),
        "attempts": avg(g.groupby("game_id_n")["attempts"].sum()),
        "interceptions": avg(g.groupby("game_id_n")["interceptions"].sum()),
        "carries": avg(g.groupby("game_id_n")["carries"].sum()),
        "receptions": avg(g.groupby("game_id_n")["receptions"].sum()),
        "targets": avg(g.groupby("game_id_n")["targets"].sum()),
        "passing_tds": avg(g.groupby("game_id_n")["passing_tds"].sum()),
        "rushing_tds": avg(g.groupby("game_id_n")["rushing_tds"].sum()),
        "receiving_tds": avg(g.groupby("game_id_n")["receiving_tds"].sum()),
    }

def team_game_allowed(players):
    # Sum opponent offensive production to measure what each defense allowed in a game.
    d = players[players["opponent"].notna()].copy()
    if d.empty: return pd.DataFrame()
    g = d.groupby(["opponent","game_id_n","week_n","team"], as_index=False)[
        ["passing_yards","rushing_yards","receiving_yards"]
    ].sum()
    return g.rename(columns={"opponent":"defense"})

def defense_overall(allowed, team):
    d = allowed[allowed["defense"] == team]
    if d.empty: return None
    bygame = d.groupby(["game_id_n","defense"], as_index=False)[["passing_yards","rushing_yards","receiving_yards"]].sum()
    return {k: avg(bygame[k]) for k in ["passing_yards","rushing_yards","receiving_yards"]} | {"games":len(bygame)}

def defense_position_allowed(players, defense, position):
    # Aggregate a defense's opponent player production by position, per game.
    d = players[(players["opponent"] == defense) & (players["position"] == position)].copy()
    if d.empty: return None
    stat_map = {
        "QB": ["passing_yards"],
        "RB": ["rushing_yards","receiving_yards"],
        "WR": ["receiving_yards"],
        "TE": ["receiving_yards"],
    }
    stats = stat_map.get(position, ["receiving_yards"])
    bygame = d.groupby("game_id_n")[stats].sum()
    result = {s: avg(bygame[s]) for s in stats}
    result["games"] = len(bygame)
    return result

def position_rank_table(players):
    # For each defense, average opponent positional production per game; lower allowed = better defense.
    rows = []
    for pos in POSITIONS:
        d = players[(players["position"] == pos) & players["opponent"].notna()].copy()
        if d.empty: continue
        stat_map = {"QB":["passing_yards"],"RB":["rushing_yards","receiving_yards"],"WR":["receiving_yards"],"TE":["receiving_yards"]}
        stats = stat_map[pos]
        bygame = d.groupby(["opponent","game_id_n"])[stats].sum().reset_index()
        byteam = bygame.groupby("opponent")[stats].mean()
        for team, row in byteam.iterrows():
            vals = {s: float(row[s]) for s in stats}
            rows.append({"position":pos,"defense":team,**vals})
    return pd.DataFrame(rows)

def get_rank(rank_df, pos, defense):
    if rank_df.empty: return {}
    x = rank_df[rank_df["position"] == pos].copy()
    if x.empty: return {}
    stats = [c for c in ["passing_yards","rushing_yards","receiving_yards"] if c in x.columns]
    result = {}
    for s in stats:
        # Position rank based on the most relevant stat for that position.
        ranks = x[s].rank(method="min", ascending=True)
        row = x[x["defense"] == defense]
        if not row.empty:
            result[s] = int(ranks.loc[row.index[0]])
    return result

def home_away_split(g):
    if "venue" not in g.columns: return None, None
    return player_window_stats(g[g["venue"] == "Home"]), player_window_stats(g[g["venue"] == "Away"])

def injury_status_for_player(injuries, player, season):
    if injuries.empty: return None
    d = injuries.copy()
    name_c = first_col(d, ["full_name","player_name","player","display_name"])
    status_c = first_col(d, ["report_status","game_status","status","injury_status"])
    if not name_c or not status_c: return None
    d["_name"] = d[name_c].astype(str).str.casefold()
    d = d[d["_name"] == str(player).casefold()]
    if "season" in d.columns:
        d = d[pd.to_numeric(d["season"], errors="coerce") == season]
    if d.empty: return None
    status = str(d.iloc[-1][status_c]).strip().title()
    mapping = {"Out":"Out","Doubtful":"Doubtful","Questionable":"Questionable","Probable":"Healthy / no limitation","Active":"Healthy / no limitation"}
    for key, value in mapping.items():
        if key.casefold() in status.casefold(): return value
    return None

def fmt(v):
    return "—" if v is None or pd.isna(v) else f"{v:,.1f}"

# --------------------------- Sidebar / Data ---------------------------
with st.sidebar:
    st.header("Analyzer settings")
    today = st.date_input("Today's date", value=dt.date.today())
    season = st.selectbox("NFL season", [2026, 2025, 2024], index=0)
    st.caption("Season and weekly data are loaded from nflverse releases.")
    st.divider()
    if st.button("Refresh live data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown("**Data sources**")
    st.markdown("[nflverse weekly stats](https://github.com/nflverse/nflverse-data/releases)")
    st.markdown("[Pro Football Reference](https://www.pro-football-reference.com/)")

player_url = f"https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.parquet"
schedule_url = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"

try:
    with st.spinner(f"Loading {season} player statistics and schedule..."):
        raw_players = load_parquet(player_url)
        raw_schedule = load_csv(schedule_url)
        players = normalize_players(raw_players, season)
        schedule = normalize_schedule(raw_schedule, season)
except Exception as e:
    st.error(f"Could not load the {season} season data. This may mean the season file has not been published yet or the feed is temporarily unavailable.")
    st.code(str(e))
    st.markdown("Try **Refresh live data**, or select a season with a published release.")
    st.stop()

if players.empty:
    st.warning(f"No player game data is currently available for {season}.")
    st.stop()

# Add home/away from schedule using game IDs where possible.
if "game_id" in players.columns and "game_id" in schedule.columns:
    sched_cols = ["game_id","home_n","away_n"]
    merge = schedule[sched_cols].drop_duplicates("game_id")
    players = players.merge(merge, on="game_id", how="left")
    players["venue"] = players.apply(lambda r: "Home" if r["team"] == r.get("home_n") else ("Away" if r["team"] == r.get("away_n") else None), axis=1)
else:
    players["venue"] = None

with st.spinner("Preparing opponent defensive splits..."):
    allowed = team_game_allowed(players)
    pos_ranks = position_rank_table(players)

names = sorted(players["player"].dropna().unique().tolist())
st.subheader("Find a player")
player_name = st.selectbox("Player name", names, index=None, placeholder="Type a player name, e.g. CeeDee Lamb")
if not player_name:
    st.info("Select a player to see season and recent averages, next opponent, position-specific defense, splits, and projections.")
    st.stop()

pg = players[players["player"] == player_name].sort_values("week_n").copy()
if pg.empty:
    st.warning("No weekly statistics found for this player.")
    st.stop()

# Team and position from the most recent recorded game.
last_row = pg.iloc[-1]
player_team = last_row["team"]
position = last_row["position"]
next_game = get_next_game(schedule, player_team, today)
opponent = next_game["opponent"] if next_game else None
team_label = TEAM_NAMES.get(player_team, player_team or "Unknown")
opp_label = TEAM_NAMES.get(opponent, opponent or "No upcoming opponent")

st.markdown(f"### {player_name}")
st.caption(f"{team_label} · {position} · {season} season")
if next_game:
    st.success(f"Next game: {team_label} vs. {opp_label} · {next_game['date'].strftime('%A, %B %-d')} · {next_game['venue']}")
else:
    st.warning("No future regular-season matchup found for this team using the selected date.")

# --------------------------- Tabs: windows ---------------------------
season_stats = player_window_stats(pg)
last5_stats = player_window_stats(pg.tail(5))
last3_stats = player_window_stats(pg.tail(3))
windows = [
    ("Full season", season_stats, pg),
    ("Last 5 games", last5_stats, pg.tail(5)),
    ("Last 3 games", last3_stats, pg.tail(3)),
]

# --------------------------- Injury controls ---------------------------
injuries = try_load_injuries(season)
auto_injury = injury_status_for_player(injuries, player_name, season)
st.divider()
st.subheader("Injury adjustment")
if auto_injury:
    st.caption(f"Available injury feed match: {auto_injury}. Verify the latest official team report before relying on this.")
else:
    st.caption("No matching automated injury status was found. Use the manual adjustment below; this does not diagnose or verify a player's availability.")
injury_choices = list(INJURY_MULTIPLIERS.keys())
default_injury = auto_injury if auto_injury in injury_choices else "Healthy / no limitation"
injury_status = st.selectbox("Player availability", injury_choices, index=injury_choices.index(default_injury))
custom_pct = 100
if injury_status == "Custom adjustment":
    custom_pct = st.slider("Expected production retained", 0, 100, 90, 5)
injury_multiplier = custom_pct / 100 if injury_status == "Custom adjustment" else INJURY_MULTIPLIERS[injury_status]
if injury_status in ("Questionable","Doubtful","Out","Custom adjustment"):
    st.warning("This is a user-controlled scenario adjustment, not a medical or team-confirmed projection. The default percentages are illustrative.")

# --------------------------- Home / away ---------------------------
home_stats, away_stats = home_away_split(pg)
if home_stats is None:
    home_stats = player_window_stats(pg.iloc[0:0])
    away_stats = player_window_stats(pg.iloc[0:0])

# --------------------------- Display ---------------------------
tabs = st.tabs(["Full season", "Last 5 games", "Last 3 games", "Home / Away", "Defense rankings", "Game log"])
for tab, (label, stats, sample) in zip(tabs[:3], windows):
    with tab:
        st.markdown(f"#### Player averages · {label}")
        st.caption(f"{stats['games']} games in sample")
        a,b,c = st.columns(3)
        a.metric("Passing yards / game", fmt(stats["passing_yards"]))
        b.metric("Rushing yards / game", fmt(stats["rushing_yards"]))
        c.metric("Receiving yards / game", fmt(stats["receiving_yards"]))
        d,e,f = st.columns(3)
        d.metric("Carries / game", fmt(stats["carries"]))
        e.metric("Targets / game", fmt(stats["targets"]))
        f.metric("Receptions / game", fmt(stats["receptions"]))
        g,h,i = st.columns(3)
        g.metric("Passing TD / game", fmt(stats["passing_tds"]))
        h.metric("Rushing TD / game", fmt(stats["rushing_tds"]))
        i.metric("Receiving TD / game", fmt(stats["receiving_tds"]))
        if opponent:
            st.divider()
            st.markdown(f"#### {opp_label} defense · overall yards allowed")
            overall = defense_overall(allowed, opponent)
            if overall:
                x,y,z = st.columns(3)
                x.metric("Passing yards allowed / game", fmt(overall["passing_yards"]))
                y.metric("Rushing yards allowed / game", fmt(overall["rushing_yards"]))
                z.metric("Receiving yards allowed / game", fmt(overall["receiving_yards"]))
                st.caption(f"Based on {overall['games']} opponent team-games.")
            else:
                st.info("No defensive sample found for this opponent yet.")
            st.markdown(f"#### {opp_label} vs. {position}s")
            pos_def = defense_position_allowed(players, opponent, position)
            ranks = get_rank(pos_ranks, position, opponent)
            if pos_def:
                if position == "QB":
                    p,q = st.columns(2)
                    p.metric("Passing yards allowed to QBs / game", fmt(pos_def.get("passing_yards")))
                    q.metric("Defense rank vs. QBs", f"#{ranks.get('passing_yards','—')} of 32")
                elif position == "RB":
                    p,q = st.columns(2)
                    p.metric("Rushing yards allowed to RBs / game", fmt(pos_def.get("rushing_yards")))
                    q.metric("Receiving yards allowed to RBs / game", fmt(pos_def.get("receiving_yards")))
                else:
                    p,q = st.columns(2)
                    p.metric("Receiving yards allowed to "+position+"s / game", fmt(pos_def.get("receiving_yards")))
                    q.metric("Defense rank vs. "+position+"s", f"#{ranks.get('receiving_yards','—')} of 32")
                st.caption(f"Position-specific sample: {pos_def['games']} team-games. Lower rank number means fewer yards allowed.")
            else:
                st.info("Position-specific opponent sample is not available yet.")
            st.divider()
            st.markdown("#### Matchup projection")
            # A transparent blend of player window and opponent position allowance.
            proj = {}
            if position == "QB" and pos_def:
                proj["Passing yards"] = (stats["passing_yards"] + pos_def.get("passing_yards", stats["passing_yards"])) / 2
            elif position == "RB" and pos_def:
                proj["Rushing yards"] = (stats["rushing_yards"] + pos_def.get("rushing_yards", stats["rushing_yards"])) / 2
                proj["Receiving yards"] = (stats["receiving_yards"] + pos_def.get("receiving_yards", stats["receiving_yards"])) / 2
            elif position in ("WR","TE") and pos_def:
                proj["Receiving yards"] = (stats["receiving_yards"] + pos_def.get("receiving_yards", stats["receiving_yards"])) / 2
            else:
                st.info("A position-based projection is not available for this position or opponent sample.")
            if proj:
                cols = st.columns(len(proj))
                for (name, value), col in zip(proj.items(), cols):
                    adjusted = value * injury_multiplier
                    col.metric(name, f"{adjusted:,.1f}", f"{(injury_multiplier-1)*100:+.0f}% injury scenario" if injury_multiplier != 1 else "No injury adjustment")
                st.caption("Baseline = 50% player's selected-window average + 50% opponent's season-to-date average allowed to that position. Then the selected availability adjustment is applied. This is a simple transparent estimate, not a betting line or guaranteed outcome.")
        else:
            st.info("Opponent comparison will appear once a next matchup is identified.")

with tabs[3]:
    st.markdown("#### Home / away production")
    if home_stats["games"] == 0 and away_stats["games"] == 0:
        st.info("Home/away game data is not available in this feed.")
    else:
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**At home**")
            st.caption(f"{home_stats['games']} games")
            st.metric("Passing yards / game", fmt(home_stats["passing_yards"]))
            st.metric("Rushing yards / game", fmt(home_stats["rushing_yards"]))
            st.metric("Receiving yards / game", fmt(home_stats["receiving_yards"]))
            st.metric("Targets / game", fmt(home_stats["targets"]))
        with c2:
            st.markdown("**On the road**")
            st.caption(f"{away_stats['games']} games")
            st.metric("Passing yards / game", fmt(away_stats["passing_yards"]))
            st.metric("Rushing yards / game", fmt(away_stats["rushing_yards"]))
            st.metric("Receiving yards / game", fmt(away_stats["receiving_yards"]))
            st.metric("Targets / game", fmt(away_stats["targets"]))

with tabs[4]:
    st.markdown("#### League defensive rankings by position")
    st.caption("Rankings use current-season opponent player yardage per game. #1 allows the fewest yards; #32 allows the most.")
    if not pos_ranks.empty:
        for pos in POSITIONS:
            st.markdown(f"**{pos} defense**")
            sub = pos_ranks[pos_ranks["position"] == pos].copy()
            if sub.empty:
                st.caption("No data yet.")
                continue
            sort_stat = {"QB":"passing_yards","RB":"rushing_yards","WR":"receiving_yards","TE":"receiving_yards"}[pos]
            sub = sub.sort_values(sort_stat).reset_index(drop=True)
            sub["Rank"] = range(1, len(sub)+1)
            sub["Defense"] = sub["defense"].map(lambda x: TEAM_NAMES.get(x,x))
            display_cols = ["Rank","Defense",sort_stat]
            sub = sub[display_cols].rename(columns={sort_stat: f"{STAT_LABELS[sort_stat]} allowed / game"})
            st.dataframe(sub, use_container_width=True, hide_index=True)
    else:
        st.info("Defensive rankings will populate after enough weekly player data is available.")

with tabs[5]:
    st.markdown("#### Player game log")
    log_cols = ["week_n","opponent","venue","passing_yards","rushing_yards","receiving_yards","carries","targets","receptions","passing_tds","rushing_tds","receiving_tds"]
    log = pg[[c for c in log_cols if c in pg.columns]].copy()
    log = log.rename(columns={
        "week_n":"Week","opponent":"Opponent","venue":"Home/Away",
        "passing_yards":"Pass Yds","rushing_yards":"Rush Yds","receiving_yards":"Rec Yds",
        "carries":"Carries","targets":"Targets","receptions":"Receptions",
        "passing_tds":"Pass TD","rushing_tds":"Rush TD","receiving_tds":"Rec TD"
    }).sort_values("Week", ascending=False)
    log["Opponent"] = log["Opponent"].map(lambda x: TEAM_NAMES.get(x,x) if pd.notna(x) else "—")
    st.dataframe(log, use_container_width=True, hide_index=True)

st.divider()
st.markdown("### How to read the projections")
st.markdown("""
- **Position-specific defense** is the key matchup view: QB passing yards allowed to QBs; RB rushing and receiving yards allowed to RBs; WR/TE receiving yards allowed to those positions.
- **Overall defense** is also shown, but it is not substituted for the position-specific matchup.
- **Home/away** splits are based on the player's historical game location.
- **Injury adjustment** is a scenario control. The optional injury feed may not match the latest official report, so verify availability with the team/NFL before using it.
- **Projection method** is deliberately visible and simple. It does not model targets, carries, snap share, weather, game script, offensive line injuries, defensive injuries, or market lines.
""")
st.markdown("**Data source:** [nflverse weekly player statistics and schedules](https://github.com/nflverse/nflverse-data/releases). Pro Football Reference is linked for reference, but this app does not scrape PFR directly; automated PFR access may be restricted.")
st.caption("Unofficial fan analytics. Not affiliated with the NFL, nflverse, or Pro Football Reference. Not betting advice.")
