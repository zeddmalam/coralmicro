import { SerialPort } from "serialport";
import sharp from "sharp";
import {
  getLatestFramePayload,
  setLatestFramePayload,
} from "@monitoring/api/frame-store";

type PoseKeypoint = {
  name: string;
  x: number;
  y: number;
  score: number;
};

type Pose = {
  score: number;
  keypoints: PoseKeypoint[];
};

type IncomingFrame = {
  imageData?: string;
  poses?: Pose[];
};

type FrameState =
  | { status: "idle" }
  | { status: "starting"; promise: Promise<void> }
  | { status: "ready"; port: SerialPort }
  | { status: "error"; message: string };

let frameState: FrameState = { status: "idle" };
let parseBuffer = "";

function getSerialConfig() {
  const serialPath = process.env.FRAME_STREAM_SERIAL_PATH ?? "";
  const baudRate = Number(process.env.FRAME_STREAM_BAUD_RATE ?? "115200");
  return { serialPath, baudRate };
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

const KEYPOINT_SCORE_THRESHOLD = 0.2;
/** PoseNet output keypoints are in this square resolution. */
const MODEL_DIM = 324;
/** Tiny placeholder images (e.g. 1×1 WebP) are expanded for skeleton overlay. */
const PLACEHOLDER_MAX_DIM = 16;

const SKELETON_CONNECTIONS: Array<[string, string]> = [
  ["nose", "leftEye"],
  ["nose", "rightEye"],
  ["leftEye", "leftEar"],
  ["rightEye", "rightEar"],
  ["leftShoulder", "rightShoulder"],
  ["leftShoulder", "leftElbow"],
  ["leftElbow", "leftWrist"],
  ["rightShoulder", "rightElbow"],
  ["rightElbow", "rightWrist"],
  ["leftShoulder", "leftHip"],
  ["rightShoulder", "rightHip"],
  ["leftHip", "rightHip"],
  ["leftHip", "leftKnee"],
  ["leftKnee", "leftAnkle"],
  ["rightHip", "rightKnee"],
  ["rightKnee", "rightAnkle"],
];

function normalizeKeypointName(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function buildKeypointOverlaySvg(
  width: number,
  height: number,
  poses: Pose[],
  poseCoordSpace: number,
): string {
  const sx = width / poseCoordSpace;
  const sy = height / poseCoordSpace;
  const lines: string[] = [];
  const circles: string[] = [];
  const keypointRadius = Math.max(
    2,
    Math.round(Math.min(width, height) * 0.012),
  );
  const lineWidth = Math.max(2, Math.round(Math.min(width, height) * 0.005));

  for (const pose of poses) {
    const keyed = new Map<string, PoseKeypoint>();
    for (const keypoint of pose.keypoints) {
      keyed.set(normalizeKeypointName(keypoint.name), keypoint);
    }

    for (const [fromName, toName] of SKELETON_CONNECTIONS) {
      const from = keyed.get(normalizeKeypointName(fromName));
      const to = keyed.get(normalizeKeypointName(toName));
      if (!from || !to) continue;
      if (
        from.score < KEYPOINT_SCORE_THRESHOLD ||
        to.score < KEYPOINT_SCORE_THRESHOLD
      ) {
        continue;
      }

      const x1 = clamp(Math.round(from.x * sx), 0, width - 1);
      const y1 = clamp(Math.round(from.y * sy), 0, height - 1);
      const x2 = clamp(Math.round(to.x * sx), 0, width - 1);
      const y2 = clamp(Math.round(to.y * sy), 0, height - 1);
      lines.push(
        `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#34d399" stroke-width="${lineWidth}" stroke-linecap="round" stroke-opacity="0.85" />`,
      );
    }

    for (const keypoint of pose.keypoints) {
      if (keypoint.score < KEYPOINT_SCORE_THRESHOLD) continue;
      const x = clamp(Math.round(keypoint.x * sx), 0, width - 1);
      const y = clamp(Math.round(keypoint.y * sy), 0, height - 1);
      circles.push(
        `<circle cx="${x}" cy="${y}" r="${keypointRadius}" fill="#ff3355" fill-opacity="0.9" />`,
      );
      circles.push(
        `<circle cx="${x}" cy="${y}" r="${Math.max(1, Math.floor(keypointRadius / 2))}" fill="#ffffff" fill-opacity="0.95" />`,
      );
    }
  }

  return `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">${lines.join("")}${circles.join("")}</svg>`;
}

async function annotateFrameImage(
  imageData: string,
  poses: Pose[],
): Promise<string> {
  if (!imageData) return imageData;

  try {
    const imageBuffer = Buffer.from(imageData, "base64");
    let image = sharp(imageBuffer);
    const metadata = await image.metadata();
    let width = metadata.width ?? 0;
    let height = metadata.height ?? 0;
    if (!width || !height) return imageData;

    if (Math.max(width, height) < PLACEHOLDER_MAX_DIM) {
      image = sharp(imageBuffer).resize(MODEL_DIM, MODEL_DIM, { fit: "fill" });
      width = height = MODEL_DIM;
    }

    if (poses.length === 0) {
      const buf = await image.jpeg({ quality: 85 }).toBuffer();
      return buf.toString("base64");
    }

    const overlay = Buffer.from(
      buildKeypointOverlaySvg(width, height, poses, MODEL_DIM),
      "utf8",
    );
    const annotatedBuffer = await image
      .composite([{ input: overlay }])
      .jpeg({ quality: 85 })
      .toBuffer();
    return annotatedBuffer.toString("base64");
  } catch (error) {
    console.error(
      "Failed to annotate frame image:",
      error instanceof Error ? error.message : "unknown error",
    );
    return imageData;
  }
}

async function publishIncomingFrame(frame: IncomingFrame): Promise<void> {
  const nextPoses =
    Array.isArray(frame.poses) && frame.poses.length ? frame.poses : [];
  const rawImageData = frame.imageData ?? "";
  const annotatedImageData = await annotateFrameImage(rawImageData, nextPoses);

  setLatestFramePayload({
    imageData: annotatedImageData,
    poses: nextPoses,
  });
}

function parseIncomingChunk(chunk: Buffer): void {
  parseBuffer += chunk.toString("utf8");

  const parts = parseBuffer.split("\n\n").reverse();
  try {
    const json = JSON.parse(parts[0] ?? "");
    parseBuffer = "";
    void publishIncomingFrame(json as IncomingFrame);
    return;
  } catch {
    // Try partial + previous segment fallback.
  }

  try {
    const json = JSON.parse(parts[1] ?? "");
    parseBuffer = parts[0] ?? "";
    void publishIncomingFrame(json as IncomingFrame);
  } catch {
    // Keep buffering until a valid JSON frame arrives.
  }
}

function openSerialPort(port: SerialPort): Promise<void> {
  return new Promise((resolve, reject) => {
    port.open((err) => {
      if (err) {
        reject(err);
        return;
      }
      resolve();
    });
  });
}

async function initializeFrameStream(): Promise<void> {
  const { serialPath, baudRate } = getSerialConfig();
  if (!serialPath) {
    throw new Error(
      "FRAME_STREAM_SERIAL_PATH is not configured. Set it in monitoring/.env.local.",
    );
  }

  const port = new SerialPort({
    path: serialPath,
    baudRate: Number.isFinite(baudRate) ? baudRate : 115200,
    autoOpen: false,
  });

  port.on("data", parseIncomingChunk);
  port.on("error", (err: Error) => {
    console.error("Frame stream serial error:", err.message);
  });

  await openSerialPort(port);
  console.error(`Frame stream serial opened ${serialPath} @ ${baudRate} baud`);
  frameState = { status: "ready", port };
}

export async function ensureFrameStreamReady(): Promise<void> {
  if (frameState.status === "ready") return;
  if (frameState.status === "starting") {
    await frameState.promise;
    return;
  }
  if (frameState.status === "error") {
    throw new Error(frameState.message);
  }

  const promise = initializeFrameStream().catch((err: Error) => {
    frameState = { status: "error", message: err.message };
    throw err;
  });

  frameState = { status: "starting", promise };
  await promise;
}

export function getLatestFrameBuffer(): Buffer | null {
  const base64 = getLatestFramePayload().imageData;
  if (!base64) return null;
  return Buffer.from(base64, "base64");
}
