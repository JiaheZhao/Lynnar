# Lynnar

A typographic agent for designing event posters. This project is the Figma plugin and the local API it calls.

## Layout

- `plugin/` — Figma plugin. Figma loads `manifest.json`, `code.js`, and `ui.html`.
- `server/` — Flask API for layout guidelines, prompt refinement, and image generation. It listens on `http://localhost:5000`.
- `.env` — API keys. Copy `env.example` if this file is missing.

## Start

From this folder:

```powershell
.\start.bat
```

That installs plugin dependencies on the first run, builds the plugin, and starts the server. Leave the window open.

In the Figma desktop app, open **Plugins → Development → Lynnar**.

The first time, import `plugin\manifest.json` with **Plugins → Development → Import plugin from manifest…**.

## Server endpoints

- `POST /predict` — layout guidelines for the selected frame
- `POST /chat` and `POST /chat/refine` — prompt refinement
- `POST /generate-image` — optional image generation

`server/primer.jsonl` is required for layout prediction.

## Figma Community

Listing files are in `community/`: the 128×128 icon, a 1920×1080 thumbnail, and three carousel slides made from the PosterDatabase deck. See `community/README.md` for the upload steps and the listing copy.

The plugin currently calls only `http://localhost:5000`. A Community install will not reach that server until the API is hosted and its URL is added to `plugin/manifest.json`.
