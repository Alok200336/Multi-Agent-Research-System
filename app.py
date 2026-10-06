"""Research Desk: a Streamlit UI for the multi-agent research pipeline.

Run with:  streamlit run app.py
Needs Streamlit >= 1.40  (pip install -U streamlit)
Keep this file next to pipeline.py, agents.py and tools.py.
"""

import re
import time
import traceback
from datetime import datetime
from urllib.parse import urlparse

import streamlit as st
from langchain_core.output_parsers import StrOutputParser

from agents import search_agent, reader_agent, writer_chain, critic_chain

st.set_page_config(
    page_title="Research Desk",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ design tokens
STEPS = [
    {"key": "search", "icon": "🔍", "name": "Search", "does": "Finds sources", "color": "#4DD8F0"},
    {"key": "read", "icon": "📖", "name": "Read", "does": "Digests evidence", "color": "#8F7CFF"},
    {"key": "write", "icon": "✍️", "name": "Write", "does": "Drafts the report", "color": "#FF7AA8"},
    {"key": "review", "icon": "📝", "name": "Review", "does": "Checks the draft", "color": "#FFB547"},
]

EXAMPLES = [
    "How solid-state batteries could change EVs",
    "State of fusion energy in 2026",
    "Why sleep affects memory",
    "Rise of small language models",
    "Microplastics and human health",
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&display=swap');

:root{
  --ink:#0D0F1E; --ink-2:#151834; --ink-3:#22264A; --line:#2D3263;
  --text:#E9EAFB; --muted:#9A9DC8;
  --cyan:#4DD8F0; --violet:#8F7CFF; --rose:#FF7AA8; --amber:#FFB547;
  --paper:#F2F1F8; --paper-ink:#1A1A26;
}

html, body, [class*="css"], .stApp { font-family:'Bricolage Grotesque', system-ui, sans-serif; }
.stApp{
  background-color:var(--ink);
  background-image:radial-gradient(rgba(143,124,255,.13) 1px, transparent 1px);
  background-size:26px 26px;
  color:var(--text);
}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{ display:none !important; }
header[data-testid="stHeader"]{ background:transparent; }
.block-container{ padding-top:2.2rem; max-width:1100px; }

/* sidebar */
[data-testid="stSidebar"]{ background:var(--ink-2); border-right:1px solid var(--line); }
[data-testid="stSidebar"] *{ color:var(--text); }
[data-testid="stSidebar"] .stButton button{
  background:transparent; border:1px solid var(--line); text-align:left; justify-content:flex-start;
  border-radius:12px; color:var(--text); font-weight:400; transition:all .18s ease;
}
[data-testid="stSidebar"] .stButton button:hover{ border-color:var(--violet); background:rgba(143,124,255,.12); transform:translateX(3px); }

/* hero */
.hero h1{
  font-size:clamp(2.3rem, 5vw, 3.9rem); line-height:1.02; font-weight:800; letter-spacing:-.03em;
  margin:0 0 .7rem 0; color:var(--text);
}
.hero p{ color:var(--muted); font-size:1.08rem; max-width:60ch; line-height:1.55; margin:0 0 1.4rem 0; }

/* input */
.stTextInput input{
  background:var(--ink-2) !important; color:var(--text) !important;
  border:1.5px solid var(--line) !important; border-radius:16px !important;
  padding:1rem 1.15rem !important; font-size:1.1rem !important;
  font-family:'Bricolage Grotesque', sans-serif !important; transition:all .2s ease;
}
.stTextInput input:focus{ border-color:var(--violet) !important; box-shadow:0 0 0 4px rgba(143,124,255,.22) !important; }
.stTextInput input::placeholder{ color:#6E72A8 !important; }
[data-testid="InputInstructions"]{ display:none !important; }
.stTextInput label{ display:none; }

/* buttons */
.stButton button[kind="primary"]{
  background:var(--violet); color:#0D0F1E; border:none; border-radius:16px; height:3.35rem;
  font-weight:800; font-size:1.05rem; width:100%; transition:transform .15s ease, box-shadow .15s ease, background .15s ease;
}
.stButton button[kind="primary"]:hover{ background:#A596FF; transform:translateY(-2px); box-shadow:0 10px 28px rgba(143,124,255,.45); }
.stButton button[kind="primary"]:active{ transform:translateY(0); }
.stButton button[kind="secondary"]{
  background:var(--ink-2); color:var(--muted); border:1px solid var(--line); border-radius:999px;
  font-size:.86rem; padding:.2rem .9rem; transition:all .18s ease;
}
.stButton button[kind="secondary"]:hover{ color:var(--text); border-color:var(--cyan); background:rgba(77,216,240,.1); transform:translateY(-2px); }

/* pipeline rail */
.rail{ display:flex; align-items:flex-start; margin:1.8rem 0 .6rem 0; padding:1.4rem 1.2rem 1.1rem;
  background:var(--ink-2); border:1px solid var(--line); border-radius:22px; }
.node{ display:flex; flex-direction:column; align-items:center; width:104px; text-align:center; flex:0 0 auto; }
.orb{
  position:relative; width:54px; height:54px; border-radius:50%; display:grid; place-items:center;
  font-size:1.45rem; background:var(--ink); border:2px solid var(--line); transition:all .4s ease;
}
.node .nm{ margin-top:.6rem; font-weight:600; font-size:.95rem; color:var(--muted); transition:color .3s; }
.node .ds{ font-size:.76rem; color:#6E72A8; margin-top:.1rem; }
.node .tm{ font-size:.76rem; color:var(--muted); min-height:1.1rem; margin-top:.15rem; font-variant-numeric:tabular-nums; }
.node.idle .orb{ filter:grayscale(.9) opacity(.55); }
.node.active .orb{ border-color:var(--c); box-shadow:0 0 0 0 var(--c); animation:pulse 1.5s infinite; background:color-mix(in srgb, var(--c) 18%, var(--ink)); }
.node.active .nm{ color:var(--text); }
.node.done .orb{ border-color:var(--c); background:color-mix(in srgb, var(--c) 22%, var(--ink)); }
.node.done .orb::after{
  content:'✓'; position:absolute; right:-5px; bottom:-4px; width:20px; height:20px; border-radius:50%;
  background:var(--c); color:#0D0F1E; font-size:.72rem; font-weight:800; display:grid; place-items:center;
  animation:pop .35s cubic-bezier(.3,1.6,.5,1);
}
.node.done .nm{ color:var(--text); }
.node.error .orb{ border-color:#FF5C5C; background:rgba(255,92,92,.15); }
.link{ flex:1; height:3px; margin-top:26px; background:var(--line); border-radius:3px; position:relative; overflow:hidden; }
.link::after{ content:''; position:absolute; inset:0; width:0; background:var(--c); transition:width .6s ease; }
.link.done::after{ width:100%; }
.link.active::after{ width:100%; background:repeating-linear-gradient(90deg, var(--c) 0 10px, transparent 10px 20px); animation:flow .8s linear infinite; }
@keyframes pulse{ 0%{box-shadow:0 0 0 0 color-mix(in srgb, var(--c) 70%, transparent);} 70%{box-shadow:0 0 0 16px transparent;} 100%{box-shadow:0 0 0 0 transparent;} }
@keyframes pop{ from{transform:scale(0);} to{transform:scale(1);} }
@keyframes flow{ from{background-position:0 0;} to{background-position:20px 0;} }

/* stat tiles */
.stats{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.8rem; margin:1.2rem 0 .4rem; }
.stat{ background:var(--ink-2); border:1px solid var(--line); border-radius:16px; padding:.9rem 1.1rem; transition:transform .2s ease, border-color .2s ease; }
.stat:hover{ transform:translateY(-3px); border-color:var(--violet); }
.stat b{ display:block; font-size:1.7rem; font-weight:800; letter-spacing:-.02em; }
.stat span{ color:var(--muted); font-size:.85rem; }

/* tabs */
.stTabs [data-baseweb="tab-list"]{ gap:.4rem; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"]{ background:transparent; color:var(--muted); border-radius:12px 12px 0 0; padding:.6rem 1.1rem; font-weight:600; transition:color .2s, background .2s; }
.stTabs [data-baseweb="tab"]:hover{ color:var(--text); background:rgba(143,124,255,.1); }
.stTabs [aria-selected="true"]{ color:var(--text) !important; }
.stTabs [data-baseweb="tab-highlight"]{ background:var(--violet) !important; height:3px; }

/* report as a sheet of paper */
.st-key-paper{
  background:var(--paper); color:var(--paper-ink); border-radius:10px; padding:2.4rem 2.8rem;
  box-shadow:0 24px 60px rgba(0,0,0,.45); font-family:'Newsreader', Georgia, serif;
}
.st-key-paper p, .st-key-paper li{ font-family:'Newsreader', Georgia, serif; font-size:1.13rem; line-height:1.75; color:var(--paper-ink); max-width:70ch; }
.st-key-paper h1, .st-key-paper h2, .st-key-paper h3{ font-family:'Bricolage Grotesque', sans-serif; color:#14142B; letter-spacing:-.02em; }
.st-key-paper a{ color:#5B49E0; }
.st-key-paper code{ background:#E3E1F3; color:#2B2470; }

/* critique */
.st-key-critique{
  background:rgba(255,181,71,.07); border:1px solid rgba(255,181,71,.35); border-left:5px solid var(--amber);
  border-radius:14px; padding:1.6rem 2rem;
}
.st-key-critique p, .st-key-critique li{ color:var(--text); line-height:1.65; }

/* sources */
.src{ display:flex; align-items:center; gap:.9rem; padding:.85rem 1.1rem; margin:.5rem 0; background:var(--ink-2);
  border:1px solid var(--line); border-radius:14px; text-decoration:none !important; transition:all .18s ease; }
.src:hover{ border-color:var(--cyan); transform:translateX(5px); background:rgba(77,216,240,.07); }
.src img{ width:22px; height:22px; border-radius:6px; background:var(--ink-3); }
.src .d{ color:var(--text); font-weight:600; }
.src .u{ color:var(--muted); font-size:.82rem; display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:62ch; }

/* status / expanders / downloads */
[data-testid="stExpander"]{ background:var(--ink-2); border:1px solid var(--line); border-radius:14px; }
.stDownloadButton button{ background:var(--ink-3); color:var(--text); border:1px solid var(--line); border-radius:12px; transition:all .18s ease; }
.stDownloadButton button:hover{ border-color:var(--cyan); transform:translateY(-2px); }
.empty{ text-align:center; color:var(--muted); padding:2.2rem 1rem 1rem; }
.empty b{ color:var(--text); }

@media (max-width:700px){
  .rail{ padding:1rem .4rem; } .node{ width:70px; } .node .ds{ display:none; }
  .orb{ width:44px; height:44px; font-size:1.2rem; } .link{ margin-top:21px; }
  .st-key-paper{ padding:1.4rem 1.2rem; }
}
@media (prefers-reduced-motion:reduce){ *{ animation:none !important; transition:none !important; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def rail_html(states, times):
    parts = ['<div class="rail">']
    for i, s in enumerate(STEPS):
        t = f"{times[i]:.1f}s" if times[i] is not None else ""
        parts.append(
            f'<div class="node {states[i]}" style="--c:{s["color"]}">'
            f'<div class="orb">{s["icon"]}</div>'
            f'<div class="nm">{s["name"]}</div><div class="ds">{s["does"]}</div>'
            f'<div class="tm">{t}</div></div>'
        )
        if i < len(STEPS) - 1:
            link = "done" if states[i] == "done" else ("active" if states[i] == "active" else "")
            parts.append(f'<div class="link {link}" style="--c:{STEPS[i]["color"]}"></div>')
    parts.append("</div>")
    return "".join(parts)


def extract_sources(text):
    urls = re.findall(r"https?://[^\s\)\]>\"'<]+", text or "")
    seen, out = set(), []
    for u in urls:
        u = u.rstrip(".,;:*")
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def run_pipeline(topic, rail_slot):
    """Same four steps as pipeline.py, updating the rail between steps."""
    parser = StrOutputParser()
    states = ["idle"] * 4
    times = [None] * 4
    state = {"topic": topic}

    def show():
        rail_slot.markdown(rail_html(states, times), unsafe_allow_html=True)

    def step(i, fn):
        states[i] = "active"
        show()
        t0 = time.perf_counter()
        try:
            out = fn()
        except Exception:
            states[i] = "error"
            show()
            raise
        times[i] = time.perf_counter() - t0
        states[i] = "done"
        show()
        return out

    def do_search():
        r = search_agent.invoke(
            {"messages": [("user",
                f"Research this topic: {topic}\n"
                "Find relevant sources using web_search. "
                "Include findings and exact source URLs.")]},
            config={"recursion_limit": 20},
        )
        return parser.invoke(r["messages"][-1])

    def do_read():
        r = reader_agent.invoke(
            {"messages": [("user",
                f"Research topic: {topic}\n\n"
                f"Search findings:\n{state['search_results']}\n\n"
                "Use scrape_url to read up to three relevant URLs "
                "from these findings. Summarize the evidence and "
                "keep source URLs beside each finding.")]},
            config={"recursion_limit": 20},
        )
        return parser.invoke(r["messages"][-1])

    def do_write():
        return writer_chain.invoke({"topic": topic, "research": state["research"]})

    def do_review():
        return critic_chain.invoke(
            {"topic": topic, "research": state["research"], "report": state["report"]}
        )

    state["search_results"] = step(0, do_search)
    state["reader_notes"] = step(1, do_read)
    state["research"] = (
        f"SEARCH FINDINGS:\n{state['search_results']}\n\n"
        f"READER NOTES:\n{state['reader_notes']}"
    )
    state["report"] = step(2, do_write)
    state["critique"] = step(3, do_review)
    state["timings"] = times
    return state


def load_result(res):
    st.session_state["result"] = res


def use_example(text):
    st.session_state["topic_input"] = text


def request_run():
    st.session_state["go"] = True


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 🔭 Research Desk")
    st.caption("Four agents, one report.")
    if st.button("＋ New research", use_container_width=True):
        st.session_state.pop("result", None)
        st.session_state["topic_input"] = ""
        st.rerun()
    history = st.session_state.get("history", [])
    st.markdown("---")
    if history:
        st.markdown("**Earlier this session**")
        for i, item in enumerate(reversed(history)):
            st.button(
                f"{item['time']}  {item['topic'][:28]}",
                key=f"hist_{i}",
                on_click=load_result,
                args=(item["state"],),
                use_container_width=True,
            )
    else:
        st.caption("Finished reports will appear here so you can reopen them.")

# ------------------------------------------------------------------ hero + input
st.markdown(
    '<div class="hero"><h1>Four agents.<br>One researched answer.</h1>'
    "<p>Type a topic. A search agent finds sources, a reader digests them, "
    "a writer drafts the report and a critic checks it.</p></div>",
    unsafe_allow_html=True,
)

c_in, c_btn = st.columns([5, 1.4], vertical_alignment="center")
with c_in:
    st.text_input(
        "Research topic",
        key="topic_input",
        placeholder="What do you want to understand?",
        on_change=request_run,
    )
with c_btn:
    clicked = st.button("Start research", type="primary", use_container_width=True)

chip_cols = st.columns(len(EXAMPLES))
for col, ex in zip(chip_cols, EXAMPLES):
    with col:
        st.button(ex, key=f"ex_{ex}", on_click=use_example, args=(ex,), use_container_width=True)

go = clicked or st.session_state.pop("go", False)
topic = st.session_state.get("topic_input", "").strip()

# ------------------------------------------------------------------ rail + run
rail_slot = st.empty()
result = st.session_state.get("result")

if go:
    if not topic:
        st.warning("Enter a topic first, or pick one of the examples above.")
        rail_slot.markdown(rail_html(["idle"] * 4, [None] * 4), unsafe_allow_html=True)
    else:
        try:
            result = run_pipeline(topic, rail_slot)
            st.session_state["result"] = result
            st.session_state.setdefault("history", []).append(
                {"topic": topic, "time": datetime.now().strftime("%H:%M"), "state": result}
            )
            st.toast("Report ready", icon="✅")
        except Exception as e:
            st.error(f"The pipeline stopped: {e}. Check your API keys and network, then try again.")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())
            result = None
elif result:
    rail_slot.markdown(rail_html(["done"] * 4, result.get("timings", [None] * 4)), unsafe_allow_html=True)
else:
    rail_slot.markdown(rail_html(["idle"] * 4, [None] * 4), unsafe_allow_html=True)

# ------------------------------------------------------------------ results
if result:
    sources = extract_sources(result["search_results"] + "\n" + result["reader_notes"])
    total = sum(t for t in result.get("timings", []) if t)
    words = len(result["report"].split())

    st.markdown(
        f'<div class="stats">'
        f'<div class="stat"><b>{len(sources)}</b><span>sources found</span></div>'
        f'<div class="stat"><b>{words:,}</b><span>words in report</span></div>'
        f'<div class="stat"><b>{total:.0f}s</b><span>total time</span></div>'
        f'<div class="stat"><b>4</b><span>agents involved</span></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"#### {result['topic']}")

    tab_report, tab_critique, tab_sources, tab_evidence = st.tabs(
        ["📄 Report", "📝 Critique", "🔗 Sources", "🧪 Evidence"]
    )

    with tab_report:
        with st.container(key="paper"):
            st.markdown(result["report"])
        st.download_button(
            "⬇ Download report (.md)",
            data=result["report"],
            file_name="research_report.md",
            mime="text/markdown",
        )

    with tab_critique:
        with st.container(key="critique"):
            st.markdown(result["critique"])

    with tab_sources:
        if sources:
            html = ""
            for u in sources:
                host = urlparse(u).netloc.replace("www.", "")
                html += (
                    f'<a class="src" href="{u}" target="_blank" rel="noopener">'
                    f'<img src="https://www.google.com/s2/favicons?domain={host}&sz=64" alt="">'
                    f'<div><span class="d">{host}</span><span class="u">{u}</span></div></a>'
                )
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("No URLs were found in the agent output.")

    with tab_evidence:
        with st.expander("Search agent findings", expanded=False):
            st.markdown(result["search_results"])
        with st.expander("Reader agent notes", expanded=False):
            st.markdown(result["reader_notes"])
else:
    if not go:
        st.markdown(
            '<div class="empty"><b>Nothing here yet.</b><br>'
            "Pick an example above or type your own topic, then press Start research.</div>",
            unsafe_allow_html=True,
        )