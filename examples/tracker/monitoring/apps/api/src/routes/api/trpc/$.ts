import { createContext } from "@monitoring/api/context";
import { appRouter } from "@monitoring/api/routers/index";
import { createFileRoute } from "@tanstack/react-router";
import { fetchRequestHandler } from "@trpc/server/adapters/fetch";
import { ensureFrameStreamReady } from "@/lib/frame-stream";

async function handler({ request }: { request: Request }) {
  await ensureFrameStreamReady();
  return fetchRequestHandler({
    req: request,
    router: appRouter,
    createContext,
    endpoint: "/api/trpc",
  });
}

export const Route = createFileRoute("/api/trpc/$")({
  server: {
    handlers: {
      GET: handler,
      POST: handler,
    },
  },
});
