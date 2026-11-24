# Plumbline brand assets (direction D: P monogram)

| File | Use |
| --- | --- |
| `plumbline-logo.svg` / `.png` | Icon plus wordmark, for light backgrounds (README header) |
| `plumbline-logo-dark.svg` / `.png` | The same, for dark backgrounds |
| `plumbline-icon.svg` | App icon, 48 px and up |
| `plumbline-icon-small.svg` | Simplified icon for 32 px and below (no string, larger bob) |
| `favicon.ico` | 16, 32 and 48 px, for browsers that want .ico |
| `favicon-16.png`, `favicon-32.png` | PNG favicons |
| `icon-192.png`, `icon-512.png` | Web app manifest icons |
| `apple-touch-icon.png` | iOS home screen, 180 px, square (iOS rounds the corners) |
| `social-preview.png` | GitHub social preview, 1280×640 (Settings > General > Social preview) |
| `social-preview.svg` | Source for the PNG. Its text needs Inter installed to render the same. |

Colours: ink `#15171C`, paper `#F6F5F1`, brass `#D9A441`, facet `#3A3F48`, muted `#A7ABB3`.

The wordmark is IBM Plex Sans SemiBold, converted to outlines, so the SVGs need no font.
IBM Plex is licensed under the SIL Open Font License 1.1, which allows this use.

## Where these are used

- `docs/brand/`: every file here (the source of truth).
- `frontend/public/`: copies of `favicon.ico`, `apple-touch-icon.png` and
  `plumbline-icon-small.svg` as `favicon.svg`, linked from `frontend/index.html`.
  Update the copies when the source changes.
- `README.md`: the header logo, switching with GitHub's light and dark themes.
- Not used in the app: the 16, 32, 192 and 512 px PNGs (there is no web manifest)
  and `social-preview.*`, which is uploaded by hand under Settings > General >
  Social preview.
