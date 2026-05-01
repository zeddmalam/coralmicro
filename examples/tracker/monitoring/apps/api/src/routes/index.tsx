import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/")({
  component: () => (
    <main style={{ fontFamily: "system-ui, sans-serif", padding: "1.5rem", lineHeight: 1.5 }}>
      <h1 style={{ fontSize: "1.25rem", fontWeight: 600 }}>Monitoring API</h1>
      <p style={{ color: "#444", marginTop: "0.5rem" }}>
        tRPC: <code>/api/trpc</code>
        {" · "}
        Auth: <code>/api/auth</code>
        {" · "}
        Frame stream: <code>/frame/stream</code>
      </p>
    </main>
  ),
});
