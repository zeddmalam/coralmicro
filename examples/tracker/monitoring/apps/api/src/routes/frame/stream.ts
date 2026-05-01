import { createFileRoute } from "@tanstack/react-router";

import { ensureFrameStreamReady, getLatestFrameBuffer } from "@/lib/frame-stream";

async function getFrameResponse() {
  try {
    await ensureFrameStreamReady();
  } catch (error) {
    return new Response(
      `Frame stream unavailable: ${error instanceof Error ? error.message : "unknown error"}`,
      { status: 503, headers: { "content-type": "text/plain; charset=utf-8" } },
    );
  }

  const frame = getLatestFrameBuffer();
  if (!frame) {
    return new Response("No frame available yet", {
      status: 204,
    });
  }

  return new Response(new Uint8Array(frame), {
    status: 200,
    headers: {
      "content-type": "application/octet-stream",
      "cache-control": "no-store",
    },
  });
}

export const Route = createFileRoute("/frame/stream")({
  server: {
    handlers: {
      GET: () => getFrameResponse(),
    },
  },
});
