# viewer-web

The browser reader and `<mipmap-viewer>` element for mipmap stores (`../docs/store-format.md`).

```bash
npm ci
npm run fixtures     # builds test/fixtures with the Python builder (needs uv)
npm test             # decodes every fixture and compares to Python
npm run lint && npm run typecheck
npm run build        # dist/mipmap-viewer.js
```

TypeScript is pinned to 6.0, the newest release `typescript-eslint` supports; 7.0 waits on it.
