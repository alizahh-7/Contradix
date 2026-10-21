"""Custom CSS for the CONTRADIX dashboard (injected once per run with st.markdown).

Design tokens: near-black navy background, purple + electric-blue accents, rounded cards,
subtle borders. Result colours: red = potential contradiction, green = entailment,
grey-blue = neutral, amber = needs review. The font stack falls back to system fonts if
Google Fonts cannot be reached (offline use).
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #070B16; --surface: #0E1426; --surface-2: #121A31; --border: rgba(148,163,184,.16);
  --text: #E6EAF5; --muted: #94A3B8; --purple: #8B5CF6; --blue: #38BDF8; --blue-2: #3B82F6;
  --danger: #F43F5E; --success: #10B981; --warning: #F59E0B; --neutral: #64748B;
  --radius: 16px;
}

html, body, .stApp, [data-testid="stAppViewContainer"] { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; }
.stApp {
  background:
    radial-gradient(900px 500px at 12% -8%, rgba(139,92,246,.20), transparent 60%),
    radial-gradient(800px 500px at 92% 0%, rgba(56,189,248,.14), transparent 60%),
    var(--bg);
  color: var(--text);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: 1rem; }
.block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem; }
[data-testid="stHeaderActionElements"] { display: none; }
hr { border-color: var(--border); }

/* ---------- typography ---------- */
.stApp h1 {
  font-size: clamp(2.4rem, 6vw, 4rem); font-weight: 800; letter-spacing: -0.03em; line-height: 1.05;
  margin: .5rem 0 .2rem 0; padding: 0;
  background: linear-gradient(92deg, #C4B5FD 0%, #8B5CF6 38%, #38BDF8 100%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; color: transparent;
}
.stApp h2, .stApp h3 { color: var(--text); letter-spacing: -0.01em; }
.stApp p, .stApp li, .stApp label { color: var(--text); }
[data-testid="stCaptionContainer"], .stApp small { color: var(--muted) !important; }

.cx-chips { display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: .4rem; }
.cx-chip { font-size: .74rem; font-weight: 600; letter-spacing: .04em; color: #C4B5FD; background: rgba(139,92,246,.12);
  border: 1px solid rgba(139,92,246,.35); padding: .28rem .7rem; border-radius: 999px; }
.cx-tagline { font-size: clamp(1.15rem, 2.6vw, 1.6rem); font-weight: 600; color: var(--text); margin-bottom: .5rem; }
.cx-subtitle { max-width: 760px; font-size: 1.02rem; line-height: 1.65; color: var(--muted) !important; margin: 0 0 1.4rem 0; }

.cx-eyebrow { font-size: .7rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
.cx-section { margin: 1.6rem 0 .8rem 0; }
.cx-section-title { margin: .15rem 0 .1rem 0; padding: 0; font-size: 1.35rem; font-weight: 700; }
.cx-section-sub { color: var(--muted) !important; margin: .1rem 0 0 0; font-size: .95rem; }

/* ---------- cards ---------- */
.cx-grid { display: grid; gap: 1rem; margin: .5rem 0 1rem 0; }
.cx-grid-3 { grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); }
.cx-grid-2 { grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
.cx-card { background: linear-gradient(180deg, var(--surface-2), var(--surface)); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 1.15rem 1.25rem; transition: border-color .2s, transform .2s, box-shadow .2s; }
.cx-mode-card:hover, .cx-info:hover { border-color: rgba(139,92,246,.55); transform: translateY(-2px);
  box-shadow: 0 10px 30px -12px rgba(139,92,246,.45); }
.cx-icon { width: 44px; height: 44px; display: grid; place-items: center; font-size: 1.35rem; border-radius: 12px;
  background: linear-gradient(135deg, rgba(139,92,246,.35), rgba(56,189,248,.25)); border: 1px solid var(--border); margin-bottom: .8rem; }
.cx-card-title { font-weight: 700; font-size: 1.05rem; color: var(--text); margin-bottom: .25rem; }
.cx-card-text { color: var(--muted) !important; font-size: .92rem; line-height: 1.55; margin: 0 0 .7rem 0; }
.cx-tag { font-size: .72rem; font-weight: 600; color: var(--blue); background: rgba(56,189,248,.1); border: 1px solid rgba(56,189,248,.28);
  padding: .2rem .55rem; border-radius: 8px; }
.cx-info { padding: 1rem 1.15rem; }
.cx-info .cx-card-text { margin-bottom: 0; }
.cx-edge-blue { border-left: 3px solid var(--blue-2); }
.cx-edge-danger { border-left: 3px solid var(--danger); }
.cx-edge-success { border-left: 3px solid var(--success); }
.cx-edge-neutral { border-left: 3px solid var(--neutral); }
.cx-edge-warning { border-left: 3px solid var(--warning); }

/* ---------- classification card ---------- */
.cx-class { border-radius: 20px; padding: 1.4rem 1.6rem; border: 1px solid var(--border); margin: .4rem 0 1rem 0; position: relative; overflow: hidden; }
.cx-class::before { content: ""; position: absolute; inset: 0; opacity: .16; pointer-events: none; }
.cx-class > * { position: relative; }
.cx-danger { border-color: rgba(244,63,94,.55); box-shadow: 0 0 0 1px rgba(244,63,94,.12), 0 18px 50px -24px rgba(244,63,94,.55); }
.cx-danger::before { background: linear-gradient(135deg, var(--danger), transparent 65%); }
.cx-success { border-color: rgba(16,185,129,.55); box-shadow: 0 18px 50px -24px rgba(16,185,129,.5); }
.cx-success::before { background: linear-gradient(135deg, var(--success), transparent 65%); }
.cx-neutral { border-color: rgba(100,116,139,.6); }
.cx-neutral::before { background: linear-gradient(135deg, var(--blue-2), transparent 65%); }
.cx-warning { border-color: rgba(245,158,11,.55); box-shadow: 0 18px 50px -24px rgba(245,158,11,.45); }
.cx-warning::before { background: linear-gradient(135deg, var(--warning), transparent 65%); }
.cx-class-top { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: .5rem; }
.cx-class-label { font-size: clamp(1.5rem, 3.4vw, 2.2rem); font-weight: 800; letter-spacing: -0.02em; margin: .35rem 0 .1rem 0; }
.cx-danger .cx-class-label { color: #FB7185; } .cx-success .cx-class-label { color: #34D399; }
.cx-neutral .cx-class-label { color: #93C5FD; } .cx-warning .cx-class-label { color: #FBBF24; }
.cx-class-sub { color: var(--muted); font-size: .92rem; }
.cx-class-note { color: var(--muted) !important; font-size: .88rem; line-height: 1.55; margin: .7rem 0 0 0; }
.cx-badge { font-size: .74rem; font-weight: 600; padding: .25rem .65rem; border-radius: 999px; border: 1px solid var(--border); background: rgba(255,255,255,.04); color: var(--text); }
.cx-conf-low { border-color: rgba(245,158,11,.6); color: #FBBF24; }
.cx-conf-high { border-color: rgba(56,189,248,.5); color: #7DD3FC; }

/* ---------- probability bars ---------- */
.cx-bars { display: grid; gap: .95rem; }
.cx-bar-head { display: flex; justify-content: space-between; font-size: .88rem; margin-bottom: .3rem; color: var(--text); }
.cx-bar-head b { font-variant-numeric: tabular-nums; }
.cx-bar-track { height: 10px; border-radius: 999px; background: rgba(148,163,184,.14); overflow: hidden; }
.cx-bar-fill { height: 100%; border-radius: 999px; }
.cx-fill-danger { background: linear-gradient(90deg, #BE123C, var(--danger)); }
.cx-fill-success { background: linear-gradient(90deg, #047857, var(--success)); }
.cx-fill-neutral { background: linear-gradient(90deg, #475569, #7C93B5); }

/* ---------- results table ---------- */
.cx-table-wrap { max-height: 540px; overflow: auto; border: 1px solid var(--border); border-radius: 14px; background: var(--surface); margin: .4rem 0 1rem 0; }
.cx-table { width: 100%; border-collapse: collapse; font-size: .86rem; min-width: 880px; }
.cx-table th { position: sticky; top: 0; z-index: 1; background: #151D38; color: var(--muted); text-align: left; font-size: .68rem; font-weight: 700;
  letter-spacing: .09em; text-transform: uppercase; padding: .7rem .8rem; border-bottom: 1px solid var(--border); }
.cx-table td { padding: .7rem .8rem; border-bottom: 1px solid rgba(148,163,184,.09); vertical-align: top; color: var(--text); line-height: 1.5; }
.cx-table tr:hover td { background: rgba(139,92,246,.06); }
.cx-idx { color: var(--muted) !important; font-variant-numeric: tabular-nums; }
.cx-pill { display: inline-block; font-size: .72rem; font-weight: 700; padding: .2rem .6rem; border-radius: 999px; white-space: nowrap; border: 1px solid; }
.cx-pill-danger { color: #FB7185; background: rgba(244,63,94,.12); border-color: rgba(244,63,94,.45); }
.cx-pill-success { color: #34D399; background: rgba(16,185,129,.12); border-color: rgba(16,185,129,.45); }
.cx-pill-neutral { color: #93C5FD; background: rgba(100,116,139,.18); border-color: rgba(100,116,139,.55); }
.cx-pill-warning { color: #FBBF24; background: rgba(245,158,11,.12); border-color: rgba(245,158,11,.45); }
.cx-conf-small { font-size: .68rem; color: var(--muted); margin-top: .25rem; }
.cx-num { white-space: nowrap; min-width: 86px; } .cx-num b { font-variant-numeric: tabular-nums; font-size: .82rem; }
.cx-mini { height: 5px; border-radius: 999px; background: rgba(148,163,184,.16); margin-bottom: .3rem; overflow: hidden; }
.cx-mini-fill { display: block; height: 100%; border-radius: 999px; }
.cx-fill-blue { background: linear-gradient(90deg, var(--blue-2), var(--blue)); }

/* ---------- score strip + details ---------- */
.cx-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: .6rem; margin: .4rem 0 .8rem 0; }
.cx-strip-item { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: .55rem .8rem; }
.cx-strip-item span { display: block; font-size: .68rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.cx-strip-item b { font-size: 1.15rem; font-variant-numeric: tabular-nums; }
.cx-strip-danger b { color: #FB7185; } .cx-strip-success b { color: #34D399; } .cx-strip-neutral b { color: #93C5FD; }
.cx-details { margin: .4rem 0 .2rem 0; } .cx-details summary { cursor: pointer; color: var(--blue); font-size: .88rem; font-weight: 600; }

/* ---------- statements ---------- */
.cx-statement { padding: 1rem 1.2rem; }
.cx-quote { font-size: 1.02rem; line-height: 1.6; margin: .35rem 0 .6rem 0; color: var(--text) !important; }
.cx-src { font-size: .74rem; color: var(--muted); background: rgba(148,163,184,.1); border: 1px solid var(--border); padding: .15rem .5rem; border-radius: 6px; display: inline-block; }
.cx-context { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: .9rem 1rem; line-height: 1.65; font-size: .93rem; }
.cx-context mark { background: rgba(245,158,11,.28); color: #FDE68A; padding: .05rem .25rem; border-radius: 4px; }
.cx-note { border-radius: 12px; padding: .75rem 1rem; font-size: .88rem; line-height: 1.55; margin: .5rem 0; border: 1px solid var(--border); color: var(--text); }
.cx-note-info { background: rgba(59,130,246,.09); border-color: rgba(59,130,246,.3); }
.cx-note-warning { background: rgba(245,158,11,.09); border-color: rgba(245,158,11,.35); }

/* ---------- methodology ---------- */
.cx-flow { display: flex; flex-wrap: wrap; align-items: stretch; gap: .5rem; margin: .6rem 0 1.2rem 0; }
.cx-step { flex: 1 1 130px; background: linear-gradient(180deg, var(--surface-2), var(--surface)); border: 1px solid var(--border); border-radius: 14px; padding: .8rem .9rem; }
.cx-step b { display: block; font-size: .9rem; color: var(--text); } .cx-step span { font-size: .78rem; color: var(--muted); }
.cx-arrow { align-self: center; color: var(--purple); font-weight: 700; }
.cx-techs { display: flex; flex-wrap: wrap; gap: .6rem; margin: .4rem 0 1rem 0; }
.cx-tech { background: var(--surface-2); border: 1px solid var(--border); border-radius: 12px; padding: .5rem .8rem; display: flex; flex-direction: column; }
.cx-tech b { font-size: .86rem; color: var(--text); } .cx-tech i { font-style: normal; font-size: .74rem; color: var(--muted); }

/* ---------- Streamlit widgets ---------- */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0B1122, #080C19); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-size: 1.05rem; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid var(--border); }
.stTabs button[role="tab"] { height: 46px; padding: 0 1.1rem; border-radius: 10px 10px 0 0; color: var(--muted); font-weight: 600; }
.stTabs button[role="tab"][aria-selected="true"] { color: var(--text); background: rgba(139,92,246,.10); }
.stTabs [data-baseweb="tab-highlight"] { background: linear-gradient(90deg, var(--purple), var(--blue)); height: 3px; }

/* mode selector (horizontal radio -> pill segmented control) */
[data-testid="stRadio"] div[role="radiogroup"] { gap: .6rem; flex-wrap: wrap; }
[data-testid="stRadio"] div[role="radiogroup"] > label { background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: .6rem 1.1rem; cursor: pointer; transition: all .18s; margin: 0; }
[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child { display: none; }
[data-testid="stRadio"] div[role="radiogroup"] > label:hover { border-color: rgba(139,92,246,.6); }
[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) { background: linear-gradient(135deg, rgba(139,92,246,.35), rgba(59,130,246,.28));
  border-color: rgba(139,92,246,.8); box-shadow: 0 8px 24px -12px rgba(139,92,246,.7); }
[data-testid="stRadio"] div[role="radiogroup"] > label p { font-weight: 600; font-size: .95rem; }

/* buttons */
.stButton button, [data-testid="stDownloadButton"] button { border-radius: 12px; border: 1px solid var(--border); background: var(--surface-2);
  color: var(--text); font-weight: 600; padding: .55rem 1rem; transition: all .18s; }
.stButton button:hover, [data-testid="stDownloadButton"] button:hover { border-color: rgba(139,92,246,.75); color: #fff; transform: translateY(-1px); }
.stButton button[kind="primary"], .stButton button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(92deg, #7C3AED 0%, #3B82F6 100%); border: 0; color: #fff; width: 100%; padding: .85rem 1.2rem; font-size: 1.05rem;
  box-shadow: 0 14px 34px -14px rgba(99,102,241,.9); }
.stButton button[kind="primary"]:hover { filter: brightness(1.1); transform: translateY(-1px); }
[data-testid="stDownloadButton"] button { width: 100%; }

/* inputs */
.stTextArea textarea, .stTextInput input { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 12px !important;
  color: var(--text) !important; font-size: .98rem; line-height: 1.55; }
.stTextArea textarea:focus, .stTextInput input:focus { border-color: var(--purple) !important; box-shadow: 0 0 0 3px rgba(139,92,246,.22) !important; }
[data-testid="stFileUploaderDropzone"] { background: var(--surface); border: 1.5px dashed rgba(139,92,246,.45); border-radius: 14px; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--purple); background: var(--surface-2); }

/* metrics -> stat cards */
[data-testid="stMetric"] { background: linear-gradient(180deg, var(--surface-2), var(--surface)); border: 1px solid var(--border); border-radius: 14px; padding: .85rem 1rem; }
[data-testid="stMetricLabel"] p { color: var(--muted) !important; font-size: .78rem; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; }
[data-testid="stMetricValue"] { font-weight: 800; font-variant-numeric: tabular-nums; }

/* expanders, bordered containers, dataframes, alerts */
[data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 14px; background: var(--surface); overflow: hidden; }
[data-testid="stExpander"] summary:hover { background: rgba(139,92,246,.07); }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; }
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stSpinner"] { color: var(--blue); }

/* ---------- responsive ---------- */
@media (max-width: 768px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 1.2rem; }
  .cx-arrow { display: none; }
  .cx-class { padding: 1.1rem 1.1rem; }
  .stTabs button[role="tab"] { padding: 0 .7rem; }
}
</style>
"""
