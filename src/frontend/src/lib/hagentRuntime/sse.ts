import type {
  JsonObject,
  RuntimeEvent,
  RuntimeEventType,
} from "./types.js";

const MAX_SSE_FRAME_CHARS = 1_048_576;
const SAFE_RUNTIME_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const EVENT_TYPES = new Set<RuntimeEventType>([
  "run_started",
  "plan_proposed",
  "artifact_produced",
  "check_completed",
  "approval_required",
  "action_completed",
  "evidence_added",
  "run_completed",
  "run_failed",
  "run_cancelled",
]);

export class RuntimeStreamProtocolError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "RuntimeStreamProtocolError";
    this.code = code;
  }
}

function protocolError(code: string, message: string): never {
  throw new RuntimeStreamProtocolError(code, message);
}

function isRecord(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requiredString(value: unknown, field: string): string {
  if (typeof value !== "string" || value.length === 0) {
    protocolError("INVALID_EVENT", `Runtime event thiếu ${field}`);
  }
  return value;
}

function requiredId(value: unknown, field: string): string {
  const id = requiredString(value, field);
  if (!SAFE_RUNTIME_ID.test(id)) {
    protocolError("INVALID_EVENT", `Runtime event có ${field} không hợp lệ`);
  }
  return id;
}

function requiredRecord(value: unknown, field: string): JsonObject {
  if (!isRecord(value)) {
    protocolError("INVALID_EVENT", `Runtime event thiếu object ${field}`);
  }
  return value;
}

function parseRuntimeEvent(
  raw: unknown,
  eventName: string,
  eventId: number,
): RuntimeEvent {
  if (!isRecord(raw) || !EVENT_TYPES.has(eventName as RuntimeEventType)) {
    protocolError("INVALID_EVENT", "Runtime event không thuộc contract");
  }
  if (raw.type !== eventName || raw.sequence !== eventId) {
    protocolError("EVENT_MISMATCH", "SSE id, event và payload không khớp");
  }

  const base = {
    run_id: requiredId(raw.run_id, "run_id"),
    command_id: requiredId(raw.command_id, "command_id"),
    sequence: eventId,
    created_at: requiredString(raw.created_at, "created_at"),
  };

  switch (eventName as RuntimeEventType) {
    case "run_started":
      return {
        ...base,
        type: "run_started",
        metadata: requiredRecord(raw.metadata, "metadata"),
      };
    case "plan_proposed":
      return {
        ...base,
        type: "plan_proposed",
        plan: requiredRecord(raw.plan, "plan"),
      };
    case "artifact_produced":
      return {
        ...base,
        type: "artifact_produced",
        artifact_type: requiredString(raw.artifact_type, "artifact_type"),
        artifact: requiredRecord(raw.artifact, "artifact"),
      };
    case "check_completed":
      return {
        ...base,
        type: "check_completed",
        checker: requiredString(raw.checker, "checker"),
        verdict: requiredString(raw.verdict, "verdict"),
        details: requiredRecord(raw.details, "details"),
      };
    case "approval_required":
      return {
        ...base,
        type: "approval_required",
        approval_id: requiredId(raw.approval_id, "approval_id"),
        proposal: requiredRecord(raw.proposal, "proposal"),
      };
    case "action_completed":
      return {
        ...base,
        type: "action_completed",
        action: requiredString(raw.action, "action"),
        outcome: requiredString(raw.outcome, "outcome"),
        details: requiredRecord(raw.details, "details"),
      };
    case "evidence_added":
      return {
        ...base,
        type: "evidence_added",
        evidence_type: requiredString(raw.evidence_type, "evidence_type"),
        evidence: requiredRecord(raw.evidence, "evidence"),
      };
    case "run_completed":
      return {
        ...base,
        type: "run_completed",
        result: requiredRecord(raw.result, "result"),
      };
    case "run_failed":
      return {
        ...base,
        type: "run_failed",
        error_code: requiredString(raw.error_code, "error_code"),
        message: requiredString(raw.message, "message"),
      };
    case "run_cancelled":
      return {
        ...base,
        type: "run_cancelled",
        reason: requiredString(raw.reason, "reason"),
      };
  }
}

function parseFrame(frame: string): RuntimeEvent {
  let eventName: string | undefined;
  let eventId: string | undefined;
  const dataLines: string[] = [];

  for (const line of frame.replace(/\r\n/g, "\n").split("\n")) {
    if (!line || line.startsWith(":")) continue;
    const separator = line.indexOf(":");
    if (separator < 0) {
      protocolError("MALFORMED_FRAME", "SSE field thiếu dấu phân cách");
    }
    const field = line.slice(0, separator);
    const value = line.slice(separator + 1).replace(/^ /, "");
    if (field === "event") {
      if (eventName !== undefined) {
        protocolError("MALFORMED_FRAME", "SSE frame lặp event field");
      }
      eventName = value;
    } else if (field === "id") {
      if (eventId !== undefined) {
        protocolError("MALFORMED_FRAME", "SSE frame lặp id field");
      }
      eventId = value;
    } else if (field === "data") {
      dataLines.push(value);
    } else {
      protocolError("MALFORMED_FRAME", "SSE frame có field ngoài contract");
    }
  }

  if (!eventName || !eventId || dataLines.length === 0) {
    protocolError("INCOMPLETE_FRAME", "SSE frame thiếu event, id hoặc data");
  }
  if (!/^\d+$/.test(eventId)) {
    protocolError("INVALID_SEQUENCE", "SSE id không phải số nguyên dương");
  }
  const sequence = Number(eventId);
  if (!Number.isSafeInteger(sequence) || sequence <= 0) {
    protocolError("INVALID_SEQUENCE", "SSE id nằm ngoài miền an toàn");
  }

  let payload: unknown;
  try {
    payload = JSON.parse(dataLines.join("\n"));
  } catch {
    protocolError("INVALID_JSON", "SSE data không phải JSON hợp lệ");
  }
  return parseRuntimeEvent(payload, eventName, sequence);
}

export class RuntimeSseParser {
  private buffer = "";
  private lastSequence: number;
  private readonly expectedRunId?: string;

  constructor(options: { afterSequence?: number; expectedRunId?: string } = {}) {
    const afterSequence = options.afterSequence ?? 0;
    if (!Number.isSafeInteger(afterSequence) || afterSequence < 0) {
      protocolError("INVALID_SEQUENCE", "afterSequence không hợp lệ");
    }
    if (options.expectedRunId && !SAFE_RUNTIME_ID.test(options.expectedRunId)) {
      protocolError("INVALID_RUN_ID", "expectedRunId không hợp lệ");
    }
    this.lastSequence = afterSequence;
    this.expectedRunId = options.expectedRunId;
  }

  feed(chunk: string): RuntimeEvent[] {
    this.buffer += chunk;
    const events: RuntimeEvent[] = [];

    while (true) {
      const boundary = /\r?\n\r?\n/.exec(this.buffer);
      if (!boundary) break;
      const frame = this.buffer.slice(0, boundary.index);
      this.buffer = this.buffer.slice(boundary.index + boundary[0].length);
      if (!frame.trim()) continue;
      if (frame.length > MAX_SSE_FRAME_CHARS) {
        protocolError("FRAME_TOO_LARGE", "SSE frame vượt giới hạn an toàn");
      }

      const event = parseFrame(frame);
      if (event.sequence <= this.lastSequence) {
        protocolError("NON_MONOTONIC_SEQUENCE", "SSE sequence không tăng");
      }
      if (this.expectedRunId && event.run_id !== this.expectedRunId) {
        protocolError("RUN_ID_MISMATCH", "Runtime event thuộc run khác");
      }
      this.lastSequence = event.sequence;
      events.push(event);
    }

    if (this.buffer.length > MAX_SSE_FRAME_CHARS) {
      protocolError("FRAME_TOO_LARGE", "SSE buffer vượt giới hạn an toàn");
    }
    return events;
  }

  finish(): void {
    if (this.buffer.trim()) {
      protocolError("INCOMPLETE_FRAME", "SSE stream kết thúc giữa frame");
    }
    this.buffer = "";
  }

  get sequence(): number {
    return this.lastSequence;
  }
}
