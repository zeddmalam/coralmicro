/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import express from "express";
import type { Request, Response } from "express";
import { SerialPort } from "serialport";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const publicDir = path.join(__dirname, "..", "public");

type CmdArgs = {
  serialPath: string;
  baudRate: number;
  listenPort: number;
  bindHost: string;
  drainMs: number;
  idleMs: number;
  maxMs: number;
};

const HOP_BY_HOP = new Set(
  [
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
  ].map((s) => s.toLowerCase()),
);

function parseArgs(argv: string[]): CmdArgs {
  let serialPath = "";
  let baudRate = 115200;
  let listenPort = 8787;
  let bindHost = "0.0.0.0";
  let drainMs = 50;
  let idleMs = 500;
  let maxMs = 10_000;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--device" || a === "-d") {
      serialPath = argv[++i] ?? "";
    } else if (a === "--baud" || a === "-b") {
      baudRate = Number(argv[++i]) || 115200;
    } else if (a === "--listen" || a === "-l") {
      listenPort = Number(argv[++i]) || 8787;
    } else if (a === "--bind") {
      bindHost = argv[++i] ?? "0.0.0.0";
    } else if (a === "--drain-ms") {
      drainMs = Number(argv[++i]) || 50;
    } else if (a === "--idle-ms") {
      idleMs = Number(argv[++i]) || 500;
    } else if (a === "--max-ms") {
      maxMs = Number(argv[++i]) || 120_000;
    } else if (a === "--help" || a === "-h") {
      printHelp();
      process.exit(0);
    }
  }
  return {
    serialPath,
    baudRate,
    listenPort,
    bindHost,
    drainMs,
    idleMs,
    maxMs,
  };
}

function printHelp(): void {
  console.log(`uart-http-proxy — static UI from public/; only GET /frame is proxied to UART.

Usage:
  npm run dev -- --device /dev/cu.usbmodem1101 [--listen 8787] [--bind 0.0.0.0]

Use http://127.0.0.1:PORT/ on macOS (localhost may use IPv6 and miss the server).

Defaults: --bind 0.0.0.0  --idle-ms 500  --max-ms 120000  --drain-ms 50
`);
}

function serializeHttpRequest(req: Request): Buffer {
  const lines: string[] = [];
  const target = req.originalUrl || "/";
  lines.push(`${req.method} ${target} HTTP/1.1`);
  for (const [name, value] of Object.entries(req.headers)) {
    if (value === undefined) continue;
    const lower = name.toLowerCase();
    if (HOP_BY_HOP.has(lower)) continue;
    const v = Array.isArray(value) ? value.join(", ") : String(value);
    lines.push(`${name}: ${v}`);
  }
  lines.push("");
  const head = lines.join("\r\n") + "\r\n";
  const body = Buffer.isBuffer(req.body) ? req.body : Buffer.alloc(0);
  return Buffer.concat([Buffer.from(head, "utf8"), body]);
}

type ParsedResponse = {
  status: number;
  headers: Record<string, string>;
  body: Buffer;
};

function parseHeadersBlock(head: string): Record<string, string> {
  const headers: Record<string, string> = {};
  const lines = head.split("\r\n");
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    const idx = line.indexOf(":");
    if (idx === -1) continue;
    const key = line.slice(0, idx).trim().toLowerCase();
    const val = line.slice(idx + 1).trim();
    headers[key] = val;
  }
  return headers;
}

function decodeChunkedBody(buf: Buffer): Buffer | null {
  let pos = 0;
  const parts: Buffer[] = [];
  while (pos < buf.length) {
    const lineEnd = buf.indexOf("\r\n", pos);
    if (lineEnd === -1) return null;
    const sizeLine = buf.subarray(pos, lineEnd).toString("latin1");
    const size = parseInt(sizeLine.split(";")[0].trim(), 16);
    if (Number.isNaN(size)) return null;
    pos = lineEnd + 2;
    if (size === 0) {
      if (buf.indexOf("\r\n", pos) === pos) return Buffer.concat(parts);
      return null;
    }
    if (pos + size > buf.length) return null;
    parts.push(buf.subarray(pos, pos + size));
    pos += size;
    if (buf[pos] !== 0x0d || buf[pos + 1] !== 0x0a) return null;
    pos += 2;
  }
  return null;
}

function tryParseHttpResponse(
  buf: Buffer,
): ParsedResponse | null | "need_more" {
  const sep = buf.indexOf("\r\n\r\n");
  if (sep === -1) return "need_more";

  const head = buf.subarray(0, sep).toString("latin1");
  const firstLineEnd = head.indexOf("\r\n");
  if (firstLineEnd === -1) return null;
  const statusLine = head.slice(0, firstLineEnd);
  const m = statusLine.match(/^HTTP\/\d\.\d (\d{3})/);
  if (!m) return null;

  const status = parseInt(m[1], 10);
  const headers = parseHeadersBlock(head);
  let body = buf.subarray(sep + 4);

  const te = (headers["transfer-encoding"] || "").toLowerCase();
  if (te.includes("chunked")) {
    const decoded = decodeChunkedBody(body);
    if (decoded === null) return "need_more";
    return { status, headers, body: decoded };
  }

  const cl = headers["content-length"];
  if (cl !== undefined) {
    const n = parseInt(cl, 10);
    if (Number.isNaN(n) || n < 0) return null;
    if (body.length < n) return "need_more";
    return { status, headers, body: body.subarray(0, n) };
  }

  return { status, headers, body };
}

function drainSerial(port: SerialPort, ms: number): Promise<void> {
  return new Promise((resolve) => {
    const onData = (): void => {
      /* discard */
    };
    port.on("data", onData);
    setTimeout(() => {
      port.off("data", onData);
      resolve();
    }, ms);
  });
}

/**
 * Read until we have a full HTTP response (per tryParseHttpResponse) or maxMs.
 * UART idle alone does not end the read while Content-Length is still short.
 */
function readSerialUntilHttpComplete(
  port: SerialPort,
  opts: { idleMs: number; maxMs: number },
): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let idleTimer: NodeJS.Timeout | undefined;
    let maxTimer: NodeJS.Timeout;
    let sawData = false;

    const cleanup = (): void => {
      clearTimeout(maxTimer);
      if (idleTimer) clearTimeout(idleTimer);
      port.removeListener("data", onData);
      port.removeListener("error", onErr);
    };

    const bufNow = (): Buffer => Buffer.concat(chunks);

    const tryResolveIfComplete = (): boolean => {
      const parsed = tryParseHttpResponse(bufNow());
      if (parsed && parsed !== "need_more") {
        cleanup();
        resolve(bufNow());
        return true;
      }
      return false;
    };

    const finishPartial = (): void => {
      cleanup();
      resolve(bufNow());
    };

    const onErr = (err: Error): void => {
      cleanup();
      reject(err);
    };

    maxTimer = setTimeout(finishPartial, opts.maxMs);

    const resetIdle = (): void => {
      if (!sawData) return;
      if (idleTimer) clearTimeout(idleTimer);
      idleTimer = setTimeout(() => {
        if (tryResolveIfComplete()) return;
        const parsed = tryParseHttpResponse(bufNow());
        if (parsed === "need_more") {
          resetIdle();
          return;
        }
        finishPartial();
      }, opts.idleMs);
    };

    const onData = (d: Buffer): void => {
      sawData = true;
      chunks.push(d);
      if (tryResolveIfComplete()) return;
      resetIdle();
    };

    port.on("data", onData);
    port.on("error", onErr);
  });
}

function writeAll(port: SerialPort, data: Buffer): Promise<void> {
  return new Promise((resolve, reject) => {
    port.write(data, (err) => {
      if (err) {
        reject(err);
        return;
      }
      port.drain((e) => (e ? reject(e) : resolve()));
    });
  });
}

class SerialGate {
  private tail: Promise<void> = Promise.resolve();

  run<T>(fn: () => Promise<T>): Promise<T> {
    const runNext = async (): Promise<T> => fn();
    const p = this.tail.then(runNext, runNext);
    this.tail = p.then(
      () => {},
      () => {},
    );
    return p;
  }
}

function filterResponseHeaders(
  h: Record<string, string>,
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(h)) {
    if (HOP_BY_HOP.has(k.toLowerCase())) continue;
    out[k] = v;
  }
  return out;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));

  if (!args.serialPath) {
    console.error("Missing --device /path/to/serial\n");
    const ports = await SerialPort.list();
    if (ports.length === 0) {
      console.error("No serial ports found.");
    } else {
      console.error("Available ports:");
      for (const p of ports) {
        const extra = [
          p.manufacturer,
          (p as { friendlyName?: string }).friendlyName,
        ]
          .filter(Boolean)
          .join(" · ");
        console.error(`  ${p.path}${extra ? "\t" + extra : ""}`);
      }
    }
    printHelp();
    process.exit(1);
  }

  const serial = new SerialPort({
    path: args.serialPath,
    baudRate: args.baudRate,
    autoOpen: false,
  });

  let dataToParse = "";
  let lastFrame: any = null;
  serial.on("data", (d: Buffer) => {
    dataToParse += d.toString("utf8");

    const parts = dataToParse.split("\n\n").reverse();
    try {
      const json = JSON.parse(parts[0]);
      dataToParse = "";
      lastFrame = json;
      return;
    } catch (e) {
      try {
        const json = JSON.parse(parts[1]);
        dataToParse = parts[0];
        lastFrame = json;
      } catch (e) {}
    }
  });

  await new Promise<void>((resolve, reject) => {
    serial.open((err) => (err ? reject(err) : resolve()));
  });

  console.error(`Opened ${args.serialPath} @ ${args.baudRate} baud`);
  console.error(
    `Idle ${args.idleMs}ms (after first RX byte), max wait ${args.maxMs}ms`,
  );

  serial.on("error", (err: Error) => {
    console.error("Serial error:", err.message);
  });

  const app = express();
  app.disable("x-powered-by");

  app.use(
    express.raw({
      type: () => true,
      limit: "100mb",
    }),
  );

  app.use(express.static(publicDir));

  const gate = new SerialGate();

  function forwardFrameToUart(req: Request, res: Response): void {
    gate
      .run(async () => {
        const payload = serializeHttpRequest(req);
        await drainSerial(serial, args.drainMs);
        await writeAll(serial, payload);
        const raw = await readSerialUntilHttpComplete(serial, {
          idleMs: args.idleMs,
          maxMs: args.maxMs,
        });

        console.log("raw", raw.toString("utf8"));

        if (raw.length === 0) {
          res
            .status(504)
            .type("text/plain")
            .send(
              "Empty UART response (wrong port or device not answering). Try --max-ms.",
            );
          return;
        }

        const parsed = tryParseHttpResponse(raw);
        if (parsed && parsed !== "need_more") {
          const hdrs = filterResponseHeaders(parsed.headers);
          delete hdrs["content-length"];
          delete hdrs["transfer-encoding"];
          res.status(parsed.status);
          for (const [k, v] of Object.entries(hdrs)) {
            res.setHeader(k, v);
          }
          res.send(parsed.body);
          return;
        }

        res.status(200);
        res.setHeader("Content-Type", "application/json");
        res.send(raw);
      })
      .catch((err: Error) => {
        console.error(err);
        if (!res.headersSent) {
          res
            .status(502)
            .type("text/plain")
            .send(`uart proxy error: ${err.message}`);
        } else {
          res.destroy();
        }
      });
  }

  const onFrameUart: express.RequestHandler = (req, res) => {
    res.status(200);
    res.setHeader("Content-Type", "application/octet-stream");
    res.send(Buffer.from(lastFrame?.imageData || "", "base64"));
  };
  // uart_camera_http.cc: /frame and /frame/...
  app.get("/frame", onFrameUart);
  app.get("/frame/*", onFrameUart);

  app.use((_req: Request, res: Response) => {
    res.status(404).type("text/plain").send("Not found");
  });

  const httpServer = http.createServer(app);
  const listenOpts: { port: number; host: string; ipv6Only?: boolean } = {
    port: args.listenPort,
    host: args.bindHost,
  };
  if (args.bindHost === "::") {
    listenOpts.ipv6Only = false;
  }

  httpServer.listen(listenOpts, () => {
    const addr = httpServer.address();
    console.error(
      `Listening ${JSON.stringify(addr)} (open http://127.0.0.1:${args.listenPort}/)`,
    );
  });

  const shutdown = (): void => {
    httpServer.close(() => {
      serial.close(() => process.exit(0));
    });
  };

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
