# Figma Community assets

Upload these from the Figma desktop app: **Plugins → Manage plugins → Publish**.

| Listing field | File | Size |
| --- | --- | --- |
| Icon | `icon.png` | 128×128 |
| Thumbnail | `thumbnail.jpg` | 1920×1080 |
| Carousel | `carousel/01-cover.jpg` | 1920×1080 |
| Carousel | `carousel/02-data.jpg` | 1920×1080 |
| Carousel | `carousel/03-prototype.jpg` | 1920×1080 |

`source/` keeps the original deck slides. The files above are those slides fitted to Figma’s 16:9 listing size. `icon.png` is the same mark as `plugin/icon.png`.

## Listing copy

**Name:** Lynnar

**Tagline:** A typographic agent for designing event posters.

**Description:**

Lynnar helps design event posters in Figma. Select a frame, choose the content to place, and generate layout guidelines for title, time, location, host, purpose, and description. A prompt conversation then refines the event brief and can generate an image.

## Before a public listing

The plugin only talks to `http://localhost:5000`. Someone who installs it from Community will not have this server, so layout guidelines, chat, and image generation will fail for them. Host the API on a public URL, add that URL to `plugin/manifest.json` under `networkAccess.allowedDomains`, and disclose that frame data and prompts are sent to your server and to OpenAI.
