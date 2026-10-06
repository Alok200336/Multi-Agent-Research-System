# Multi-Agent Research System

A Python research assistant that searches the web, reads source pages, writes a report, and reviews the result. Built with LangChain agents, LCEL chains, OpenAI, Tavily, BeautifulSoup, and Streamlit.

Enter a topic in the browser or terminal to generate a report with source references and AI-generated feedback.

## Features

- **Search agent:** Finds relevant web sources through Tavily.
- **Reader agent:** Uses a BeautifulSoup scraping tool to extract source text and summarize evidence.
- **Writer chain:** Produces a report from the gathered research using LCEL.
- **Critic chain:** Reviews evidence support, clarity, coverage, and citation quality.
- **Streamlit interface:** Displays the report, feedback, search findings, and source notes in separate tabs.
- **Markdown export:** Downloads the report together with feedback and supporting notes.
- **Terminal workflow:** Prints each component's output while the pipeline runs.
- **Session results:** Keeps the completed result available during ordinary Streamlit reruns.

## How It Works

The pipeline runs sequentially: search, read, write, then review. A shared Python dictionary passes outputs between stages.

| Component | Implementation | Input | Output |
| --- | --- | --- | --- |
| Search agent | `create_agent` with `web_search` | Research topic | Findings and source URLs |
| Reader agent | `create_agent` with `scrape_url` | Topic and search findings | Source notes with URLs |
| Writer | LCEL: prompt, model, output parser | Topic and combined evidence | Research report |
| Critic | LCEL: prompt, model, output parser | Topic, evidence, and report | Scores and improvement feedback |

There are **two tool-using agents and two LCEL chains**. The supervisor is the `run_research_pipeline` Python function. The outer workflow is sequential Python orchestration; the writer and critic use LCEL pipe composition.

## Technology Stack

| Technology | Purpose |
| --- | --- |
| Python | Application logic |
| LangChain | Agent creation and tool integration |
| LangChain Core / LCEL | Prompts, Runnables, and output parsing |
| LangChain OpenAI | OpenAI chat model integration |
| Tavily Python SDK | Web search |
| Requests | Fetching web pages |
| BeautifulSoup | HTML parsing and text extraction |
| Streamlit | Browser interface |
| python-dotenv | Local environment variable loading |

## Project Files

| File | Responsibility |
| --- | --- |
| `app.py` | Streamlit interface and Markdown download |
| `pipeline.py` | Runs the four stages and returns their shared state |
| `agents.py` | Defines the search agent, reader agent, writer chain, and critic chain |
| `tools.py` | Defines `web_search` and `scrape_url` |
| `requirements.txt` | Python dependencies |
| `.env` | Local API keys; excluded from Git |
| `.gitignore` | Excludes secrets, environments, and generated files |
| `README.md` | Project documentation |

## Setup

### 1. Get the project

The clone command below assumes the repository is named `Multi-Agent-Research-System` under `Alok200336`. Adjust it if your repository name differs. Skip cloning if you already have the project locally.

```bash
git clone https://github.com/Alok200336/Multi-Agent-Research-System.git
cd Multi-Agent-Research-System
```

### 2. Create a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Use a Python version supported by the installed dependencies. This project does not currently provide a tested Python-version matrix or pinned dependency lockfile.

### 3. Install dependencies

Ensure `requirements.txt` includes the following packages:

```text
langchain
langchain-core
langchain-openai
langchain-tavily
tavily-python
requests
beautifulsoup4
lxml
python-dotenv
pydantic
streamlit
```

```bash
python -m pip install -r requirements.txt
```

`tavily-python` provides `from tavily import TavilyClient`. The current search tool uses this SDK directly; `langchain-tavily` is only needed if using its LangChain integration. The current scraper uses Python's built-in `html.parser`, so `lxml` is optional for that implementation.

### 4. Configure API keys

Create a file named `.env` in the project folder:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
OPENAI_MODEL=gpt-5-nano
```

`OPENAI_MODEL` is optional: the supplied `agents.py` defaults to `gpt-5-nano`. Set it to a model available to your OpenAI API account that supports tool calling.

Obtain keys from [OpenAI Platform](https://platform.openai.com/) and [Tavily](https://app.tavily.com/). Requests use your provider accounts and may incur charges or count against usage limits.

Restart the Streamlit process after changing environment variables or API keys.

## Run the Browser Interface

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

1. Enter a research topic.
2. Click **Start research**.
3. Wait for the pipeline to finish.
4. Explore **Report**, **Review & feedback**, **Search findings**, and **Source notes**.
5. Click **Download research (.md)** to save the output.

The UI shows a loading indicator while the full pipeline runs. Detailed stage output remains in the terminal; the tabs populate after completion. **Clear results** removes the displayed result from the current session.

## Run from the Terminal

```bash
python pipeline.py
```

Enter a topic when prompted, for example:

```text
How is artificial intelligence used in Indian agriculture?
```

The terminal displays search findings, reader notes, the report, and the critique. The command-line version returns its result in memory; it does not automatically save a report file.

## Use from Python

```python
from pipeline import run_research_pipeline

result = run_research_pipeline(
    "Benefits and limitations of retrieval-augmented generation"
)

print(result["report"])
print(result["critique"])
```

The returned dictionary contains:

| Key | Contents |
| --- | --- |
| `topic` | Research question |
| `search_results` | Search agent's final findings |
| `reader_notes` | Reader agent's final notes |
| `research` | Combined search findings and reader notes |
| `report` | Writer chain's report |
| `critique` | Critic chain's feedback |

Keep standalone tool tests under a main guard in `tools.py` so importing the module does not execute test searches:

```python
if __name__ == "__main__":
    print(web_search.invoke("What is the capital of France?"))
    print(scrape_url.invoke({"url": "https://example.com"}))
```

## Troubleshooting

### `python` or `streamlit`: command not found

From the project folder on macOS or Linux, call the environment's interpreter directly:

```bash
./.venv/bin/python -m pip install streamlit
./.venv/bin/python -m streamlit run app.py
```

If the folder path contains spaces, enclose the full path in quotes. Use the environment's Python interpreter rather than relying on the shell prompt to confirm activation.

### `No module named pip`

```bash
./.venv/bin/python -m ensurepip --upgrade
./.venv/bin/python -m pip install -r requirements.txt
```

### `No module named tavily`

```bash
./.venv/bin/python -m pip install tavily-python
```

In VS Code, select the project's `.venv/bin/python` using **Python: Select Interpreter**.

### Cannot import `Tool` from `langchain.tools`

Use:

```python
from langchain_core.tools import Tool, tool
```

The supplied agents use `from langchain.agents import create_agent`; avoid mixing this implementation with older `AgentExecutor` tutorial code.

### `TypeError: string indices must be integers`

When using the full Tavily search response, iterate over its result list:

```python
for item in response["results"]:
    print(item["title"], item["url"])
```

### Missing keys, authentication errors, or failed research

Confirm that `.env` is beside the Python files and contains valid keys. For model-access, quota, or rate-limit errors, inspect the terminal traceback and the relevant provider account. The UI keeps the previous successful result when a new run fails.

### A page cannot be read

Some sites reject automated requests, require login, or render content with JavaScript. The scraper cannot read every search result. A failed request should be reported as missing evidence, not treated as a successful source read.

## Current Limitations

- The scraper returns only the first 3,000 characters of cleaned page text.
- Requests and BeautifulSoup do not execute JavaScript or parse PDF documents in this implementation.
- The reader is instructed to read up to three URLs; this is a prompt instruction, not a strict per-tool-call limit.
- Agent invocations use a recursion limit of 20 to bound execution steps; this is not a monetary budget.
- The critic evaluates supplied evidence and does not independently verify facts. Its scores are model-generated judgments, not benchmark accuracy measurements.
- Feedback does not automatically trigger a report rewrite.
- Retrieved pages may be incomplete, outdated, or misleading. Check original sources before relying on important claims.
- Session state is not a database. Results are not guaranteed to survive browser reloads, disconnection, or server restarts; download reports you want to keep.
- The app has no authentication, durable history, background job queue, or production deployment configuration.
- Dependency versions are unpinned. End-to-end operation depends on local configuration, provider access, and reachable source websites.

## Secrets and Git

Keep these patterns in `.gitignore`:

```gitignore
.env
.env.*
!.env.example
.venv/
venv/
__pycache__/
*.pyc
.DS_Store
.streamlit/secrets.toml
```

Never commit real API keys. If sharing an `.env.example`, include placeholder values only. Ignore rules do not remove files already tracked by Git; remove any accidentally tracked secret file from tracking and rotate exposed keys.

## Possible Improvements

These are future ideas, not implemented features:

- Live stage updates in the browser.
- Structured critic scores and an optional revision loop.
- PDF ingestion and longer-document processing.
- Persistent research history and richer exports.
- Automated tests and reproducible dependency versions.
- Usage tracking, caching, and deployment hardening.

## Author

**Alok Kumar** — [GitHub](https://github.com/Alok200336)

## License

No license is specified in this documentation. Add a `LICENSE` file to the repository if you want to grant explicit reuse permissions.
