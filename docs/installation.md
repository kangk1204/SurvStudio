# Installation and troubleshooting

[Back to quick start](../README.md)

## Recommended install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then open a new terminal:

```bash
uv tool install --python 3.12 "survstudio[all] @ https://github.com/kangk1204/SurvStudio/archive/refs/heads/main.zip"
survstudio
```

Open <http://127.0.0.1:8000>. The `[all]` option includes file readers, ML/DL models and figure export.
Python is downloaded automatically if needed. No Git clone is required.

On an Apple silicon Mac, run `xcode-select --install` if Apple's command-line tools are missing.
On an ARM Linux machine, a source dependency may need a compiler; on Ubuntu, install it with
`sudo apt install build-essential`.

## Smaller install

For CSV/TSV input and classical survival analysis:

```bash
uv tool install --python 3.12 "survstudio @ https://github.com/kangk1204/SurvStudio/archive/refs/heads/main.zip"
```

For Excel/Parquet input and ML, without deep learning:

```bash
uv tool install --python 3.12 "survstudio[formats,ml] @ https://github.com/kangk1204/SurvStudio/archive/refs/heads/main.zip"
```

Use `[all]` when you need DL and local figure-export dependencies.

## Update or remove

To fetch and reinstall the current `main` version:

```bash
uv tool install --force --refresh --python 3.12 "survstudio[all] @ https://github.com/kangk1204/SurvStudio/archive/refs/heads/main.zip"
```

To remove the app: `uv tool uninstall survstudio`.

## Developer install

Use Python 3.11 or newer and Git.

**macOS / Linux**

```bash
git clone https://github.com/kangk1204/SurvStudio.git
cd SurvStudio
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m survival_toolkit
```

**Windows PowerShell**

```powershell
git clone https://github.com/kangk1204/SurvStudio.git
cd SurvStudio
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m survival_toolkit
```

Use `[all]` instead of `[dev]` for runtime features without test tools.
Run `pytest -q` to check a developer installation. To reproduce the example outputs:

```bash
python examples/run_demo.py --models --output demo_results.json
```

For browser tests, add `python -m pip install -e ".[dev,e2e]"` and
`python -m playwright install chromium`. Linux may need
`python -m playwright install --with-deps chromium`.

## Docker

```bash
docker build -t survstudio https://github.com/kangk1204/SurvStudio.git
docker run --rm -p 127.0.0.1:8000:8000 survstudio
```

Open <http://127.0.0.1:8000>. The default image includes file formats and ML; build with
`--build-arg EXTRAS=all` to include DL. Stop the container with Ctrl+C.

## Common problems

| Problem | What to do |
|---|---|
| `uv` is not found | Close the terminal and open it again. |
| `survstudio` is not found | Run `uv tool update-shell`, then open a new terminal. |
| Port 8000 is busy | Run `survstudio serve --port 8001` and open <http://127.0.0.1:8001>. |
| The browser page stops responding | Keep the terminal running; restart with `survstudio` if it was closed. |
| A model package is missing | Reinstall with `[all]`. |
| DL takes too long | Start with one model, holdout evaluation and a small feature set. |
| An uploaded file is refused | Check the message, column names, numeric follow-up time and event coding. |

The local app has no login. Keep the default `127.0.0.1` address for use on your own computer.
Uploaded datasets are held in memory and are discarded when the app stops.
