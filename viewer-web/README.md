# viewer-web

The browser reader and `<mipmap-viewer>` element for mipmap stores (`../docs/store-format.md`).

```bash
npm ci
npm run fixtures     # builds test/fixtures with the Python builder (needs uv)
npm test             # decodes every fixture and compares to Python
npm run e2e          # renders fixtures in Chromium and checks pixels (npx playwright install chromium once)
npm run lint && npm run typecheck
npm run build        # dist/mipmap-viewer.js
MIPMAP_STORE_DIR=/path/to/stores npm run dev   # standalone page; open ?store=/stores/<name>.zarr
```

TypeScript is pinned to 6.0, the newest release `typescript-eslint` supports; 7.0 waits on it.
