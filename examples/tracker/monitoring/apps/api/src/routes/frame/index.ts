import { createFileRoute } from "@tanstack/react-router";

import {
  ensureFrameStreamReady,
  getLatestFramePoses,
} from "@/lib/frame-stream";

async function getFramePayloadResponse() {
  try {
    await ensureFrameStreamReady();
  } catch (error) {
    return new Response(
      `Frame stream unavailable: ${error instanceof Error ? error.message : "unknown error"}`,
      { status: 503, headers: { "content-type": "text/plain; charset=utf-8" } },
    );
  }

  const poses = getLatestFramePoses();

  return Response.json(poses, {
    status: 200,
    headers: {
      "cache-control": "no-store",
    },
  });
}

export const Route = createFileRoute("/frame/")({
  server: {
    handlers: {
      GET: () => getFramePayloadResponse(),
    },
  },
});
