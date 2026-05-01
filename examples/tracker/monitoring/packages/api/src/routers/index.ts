import { TRPCError } from "@trpc/server";
import { z } from "zod";
import {
  getLatestFramePayload,
  subscribeToFramePayload,
  type FramePayload,
} from "../frame-store";

import { protectedProcedure, publicProcedure, router } from "../index";

const describeFrameInputSchema = z
  .object({
    model: z.string().min(1).optional(),
    ollamaUrl: z.string().url().optional(),
  })
  .optional();

const objectBoxSchema = z.object({
  x: z.number(),
  y: z.number(),
  width: z.number(),
  height: z.number(),
});

const objectDetectionItemSchema = z.object({
  name: z.string(),
  confidence: z.number().optional(),
  box: objectBoxSchema,
});

const objectDetectionResultSchema = z.object({
  objects: z.array(objectDetectionItemSchema),
  summary: z.string(),
});

const objectDetectionResponseSchema = {
  type: "object",
  properties: {
    objects: {
      type: "array",
      items: {
        type: "object",
        properties: {
          name: { type: "string" },
          confidence: { type: "number" },
          box: {
            type: "object",
            properties: {
              x: { type: "number" },
              y: { type: "number" },
              width: { type: "number" },
              height: { type: "number" },
            },
            required: ["x", "y", "width", "height"],
          },
        },
        required: ["name", "box"],
      },
    },
    summary: { type: "string" },
  },
  required: ["objects", "summary"],
};

type OllamaChatResponse = {
  message?: { content?: string };
};

async function describeObjectsFromImageWithOllama(params: {
  imageData: string;
  model: string;
  ollamaUrl: string;
}): Promise<z.infer<typeof objectDetectionResultSchema>> {
  const response = await fetch(`${params.ollamaUrl}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model: params.model,
      stream: false,
      format: objectDetectionResponseSchema,
      messages: [
        {
          role: "user",
          content:
            "Describe which objects are visible in this image and include a bounding box for each object. Bounding boxes must be pixel coordinates in the image space with fields x, y, width, height. Return valid JSON only.",
          images: [params.imageData],
        },
      ],
    }),
  });

  if (!response.ok) {
    const bodyText = await response.text();
    throw new TRPCError({
      code: "BAD_REQUEST",
      message: `Ollama request failed (${response.status}): ${bodyText.slice(0, 500)}`,
    });
  }

  const json = (await response.json()) as OllamaChatResponse;
  const content = json.message?.content?.trim();
  if (!content) {
    throw new TRPCError({
      code: "BAD_REQUEST",
      message: "Ollama returned an empty response body.",
    });
  }

  try {
    const parsed = JSON.parse(content);
    return objectDetectionResultSchema.parse(parsed);
  } catch {
    throw new TRPCError({
      code: "BAD_REQUEST",
      message: "Ollama response was not valid object-detection JSON.",
    });
  }
}

async function* framePayloadUpdates(): AsyncGenerator<FramePayload> {
  yield getLatestFramePayload();

  const queue: FramePayload[] = [];
  let resolveNext: (() => void) | null = null;
  const unsubscribe = subscribeToFramePayload((payload) => {
    queue.push(payload);
    if (resolveNext) {
      resolveNext();
      resolveNext = null;
    }
  });

  try {
    while (true) {
      if (queue.length === 0) {
        await new Promise<void>((resolve) => {
          resolveNext = resolve;
        });
      }

      while (queue.length > 0) {
        const nextPayload = queue.shift();
        if (nextPayload) {
          yield nextPayload;
        }
      }
    }
  } finally {
    unsubscribe();
  }
}

export const appRouter = router({
  healthCheck: publicProcedure.query(() => {
    return "OK";
  }),
  describeLatestFrame: publicProcedure
    .input(describeFrameInputSchema)
    .output(
      z.object({
        model: z.string(),
        result: objectDetectionResultSchema,
      }),
    )
    .query(async ({ input }) => {
      const model = input?.model ?? "llava";
      const ollamaUrl = input?.ollamaUrl ?? "http://127.0.0.1:11434";
      const latest = getLatestFramePayload();
      if (!latest.imageData) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: "No frame image data available yet.",
        });
      }

      const description = await describeObjectsFromImageWithOllama({
        imageData: latest.imageData,
        model,
        ollamaUrl,
      });

      return {
        model,
        result: description,
      };
    }),
  framePayload: publicProcedure.subscription(async () => {
    try {
      return framePayloadUpdates();
    } catch (error) {
      throw new TRPCError({
        code: "INTERNAL_SERVER_ERROR",
        message: `Frame stream unavailable (${error instanceof Error ? error.message : "unknown error"})`,
      });
    }
  }),
  privateData: protectedProcedure.query(({ ctx }) => {
    return {
      message: "This is private",
      user: ctx.session.user,
    };
  }),
});
export type AppRouter = typeof appRouter;
