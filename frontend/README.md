Frontend development notes

- Vite dev server: `npm run dev` (http://localhost:5173)
- The frontend proxies `/api` to the backend defined in `vite.config.js` (default: http://localhost:8010)
- If uploads fail with `ECONNREFUSED`, ensure the backend is running (backend folder):

  ```powershell
  cd backend
  .\.venv\Scripts\Activate.ps1
  python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
  ```

- The UI automatically pings `/api/health` and shows a warning if the backend is unreachable.
