# Frontend (React + TypeScript + React Three Fiber + Plotly)

See the "Interactive digital phantom" section of the top-level README.md.

    npm install
    npm run dev      # http://localhost:5173, proxies /api to the backend on :8000
    npm run build    # writes dist/, which `python -m app` serves on :8000

Unit conversions (SI from the API -> mm / ms / uV on screen) live only in src/units.ts.
