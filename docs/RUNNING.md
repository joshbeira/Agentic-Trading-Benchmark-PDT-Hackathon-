# Running it from scratch

Four commands on a machine that has never seen this project. No API key, no network after
the clone, no config to edit, nothing to generate — the run data is committed, so cloning
*is* the setup.

Budget about **three minutes**, nearly all of it `pip` downloading Streamlit.

---

## Before you start

You need two things.

**Python 3.12 or newer.** Check it:

```
python3 --version      # Linux / macOS
py --version           # Windows
```

If that prints 3.11 or lower, or "command not found", install from
[python.org/downloads](https://www.python.org/downloads/). On Windows, tick
**"Add Python to PATH"** on the first screen of the installer — if you miss it, nothing
below will work and the error won't tell you why.

> **Windows note:** if you installed Python from the Microsoft Store, virtual environments
> can behave strangely because of how the Store sandboxes paths. The python.org installer
> is the one to use.

**Git.** [git-scm.com/downloads](https://git-scm.com/downloads). Accept the defaults.

---

## Linux and macOS

Open a terminal.

```bash
# 1. Get the code and the data (~64 MB, includes 600 episode logs)
git clone https://github.com/joshbeira/Agentic-Trading-Benchmark-PDT-Hackathon-.git
cd Agentic-Trading-Benchmark-PDT-Hackathon-

# 2. Make an isolated environment, so nothing here touches your system Python
python3 -m venv .venv
source .venv/bin/activate

# 3. Install everything (~2 min: this is the slow step)
pip install -e .

# 4. Start it
python -m streamlit run scripts/app.py
```

Your browser opens at **http://localhost:8501**. If it doesn't, open that address
yourself.

---

## Windows

Open **PowerShell** (press `Win`, type `powershell`, hit Enter).

```powershell
# 1. Get the code and the data (~64 MB, includes 600 episode logs)
git clone https://github.com/joshbeira/Agentic-Trading-Benchmark-PDT-Hackathon-.git
cd Agentic-Trading-Benchmark-PDT-Hackathon-

# 2. Make an isolated environment
py -3.12 -m venv .venv

# 3. Install everything (~2 min: this is the slow step)
.\.venv\Scripts\python.exe -m pip install -e .

# 4. Start it
.\.venv\Scripts\python.exe -m streamlit run scripts\app.py
```

Your browser opens at **http://localhost:8501**. If it doesn't, open that address
yourself.

> **Why call `.\.venv\Scripts\python.exe` instead of activating?** Because activating a
> virtual environment in PowerShell trips its script-execution policy on a lot of Windows
> machines, and the error it gives you is not obvious. Calling the environment's Python
> directly does the same job and cannot hit that problem. If you'd rather activate:
>
> ```powershell
> .\.venv\Scripts\Activate.ps1
> # blocked? then, for this window only:
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```
>
> In `cmd.exe` rather than PowerShell, the activate line is `.venv\Scripts\activate.bat`.

---

## What you should see

The sidebar offers two runs. It opens on **`_fixture_demo`**.

| Tab | What's there |
|---|---|
| **Headline** | A scatter of edge against identifiability. On `_fixture_demo` it reports **"No slope — the question is degenerate here"** and shows a single collapsed mark. **That is correct, not a bug** — the fixture's stub probe names the real series exactly as often as the twin, so identifiability is 0.000 on every window, and a regression against a constant is undefined. |
| **Scoreboard** | Agents ranked by median floored Sharpe, then split by regime. |
| **Learning** | Two lanes (the LLM arms), with the four baselines collapsed below. |
| **Reliability** | Every call the agents made, counted from the logs. |
| **Episode** | The equity curve, the position, all 89 calls, the fills, the memory notes. |

**Switch the sidebar to `baselines_dev`** for the real data: 240 episodes, 4 baselines × 2
tracks × 30 windows. Note `sma_10_50` sits at **−0.54** in red. That is the honest number —
the friction was never tuned to make the momentum baseline look good.

**Read [DEMO_GUIDE.md](DEMO_GUIDE.md) beside the app.** It explains what a twin is, what
every column means, and the two numbers you should refuse to quote on stage.

---

## Stopping it

Press `Ctrl+C` in the terminal.

---

## Running the tests

```bash
pip install pytest
python -m pytest -q
```

On Windows: `.\.venv\Scripts\python.exe -m pytest -q`

You should get **218 passing tests** and exit code 0. Nothing touches the network.

---

## When it doesn't work

**`python: command not found`, or it runs the wrong version.**
On Windows use `py -3.12` instead of `python`. On Linux/macOS use `python3`. Inside an
activated venv, plain `python` is correct.

**PowerShell: "running scripts is disabled on this system".**
You tried to activate the venv. Either use the `.\.venv\Scripts\python.exe -m ...` form
above, which sidesteps it, or run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first — that affects only
the current window.

**"Port 8501 is already in use."**
Something else is on that port, quite possibly an earlier copy of this app. Pick another:

```
python -m streamlit run scripts/app.py --server.port 8502
```

**"No run directories under .../runs. Nothing to read."**
The clone didn't bring the data. Almost always this means you downloaded the repo as a ZIP
from GitHub's web UI, or the clone was interrupted. Clone it properly with `git clone` and
check:

```
git status          # should say "working tree clean"
```

You should have 600 files under `runs/`. If you need to rebuild them from scratch, both
generators run offline:

```
python scripts/run_baselines.py       # -> runs/baselines_dev   (240 real episodes)
python scripts/make_fixture_run.py    # -> runs/_fixture_demo   (synthetic)
```

**`pip install -e .` fails to build a wheel.**
Your Python is probably older than 3.12. Check with `python --version`.

**The page loads but the first run takes a few seconds.**
Expected. It reads and revalidates 360 episode logs on first load, then caches them.
Switching tabs after that is instant.

---

## What was actually verified

Cloned fresh from GitHub into an empty directory, on a machine with none of this
project's dependencies installed, then run exactly as written above: install succeeded,
**218 tests passed**, and the demo rendered both runs across all five tabs with no console
errors, no clipped text, and no horizontal scrolling down to a 900px-wide window.

That was on **Linux**. The commands for Windows and macOS are the standard ones and there
is no OS-specific code in the project — every path goes through `pathlib`, nothing shells
out, and every dependency ships prebuilt wheels for all three platforms. But to be
straight with you: **Windows and macOS were not executed end-to-end.** If the demo is
going to run on a borrowed laptop, clone it there once, in advance, and watch it come up.
