import { useEffect, useState } from "react";

type HealthResponse = {
  status: string;
};

export function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/health")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<HealthResponse>;
      })
      .then((data) => {
        if (!cancelled) {
          setHealth(data);
          setError(null);
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setHealth(null);
          setError(e instanceof Error ? e.message : "Request failed");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <main className="layout">
      <h1>Monitoring</h1>
      <p className="muted">React UI with a Go API (proxied from Vite).</p>
      <section className="card">
        <h2>API health</h2>
        {health && <p className="ok">Connected: {health.status}</p>}
        {error && <p className="err">Could not reach API: {error}</p>}
        {!health && !error && <p>Loading…</p>}
      </section>
    </main>
  );
}
