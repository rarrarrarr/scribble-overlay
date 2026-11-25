# scribble-overlay

[scribble-overlay.user.js](scribble-overlay.user.js) is a WebGL2 port of the sketch look in [scribble.py](https://github.com/rarrarrarr/afha/blob/main/scripts/scribble.py), part of [scribble](https://github.com/rarrarrarr/scribble). It draws any video playing in the browser as a live sketch, laid over the video. Nothing is downloaded or saved unless you click **Save PNG**.

## Try it on a local file

Open [index.html](index.html) in Chrome or Edge, pick a video file, and the sketch turns on. The page loads the same script that Tampermonkey would install.

## Install it as a userscript

1. Install [Violentmonkey](https://violentmonkey.github.io/) or [Tampermonkey](https://www.tampermonkey.net/). In Chrome, turn on **Allow User Scripts** on the extension's details page (on older Chrome versions, turn on developer mode instead).
2. Open [scribble-overlay.user.js](https://raw.githubusercontent.com/rarrarrarr/scribble-overlay/main/scribble-overlay.user.js) and click **Install** when the extension asks. It checks this link for updates.
3. On a page with a video, press **Alt+S**.

## Controls

- **Alt+S** shows or hides the sketch and its panel.
- **Before / after** shows the original video to the left of the slider.
- The other sliders change the look. Their defaults match `scribble.py`. **Copy settings** copies the values as JSON so they can be carried back to the Python version.
- **Save PNG** saves the current sketched frame.

Settings are kept per site, in that site's local storage.

## Limits

- Videos are processed at 1080p at most.
- DRM-protected videos (rented movies, some music videos) show as black, because the browser doesn't let pages read their frames.
- Videos served from another site without CORS permission can't be read. The panel says so.
- If a site makes the `<video>` element itself fullscreen, the overlay isn't shown there. YouTube makes its player container fullscreen, so the overlay still works on YouTube.
- On the test page the overlay covers the browser's built-in video controls. They still work when clicked.
- The hatch and paper textures use a different random generator from NumPy, so the lines don't fall in the same places as in the Python output. The tone and line work match.
