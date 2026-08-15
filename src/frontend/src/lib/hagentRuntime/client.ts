import { getSession } from "next-auth/react";

import { RuntimeSseParser } from "./sse";
import type {
  CancelRunInput,
  JsonObject,
  ResolveApprovalInput,
  RuntimeEvent,
  RuntimeRequestOptions,
  RuntimeStreamResult,
  StartRunInput,
} from "./types";

const RUNS_ENDPOINT = "/api/hagent/api/v1/runs";
const SAFE_RUNTIME_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const SAFE_ERROR_CODE = /^[A-Z][A-Z0-9_]{0,63}$/;

type SessionWithAccessToken = {
  user?: {
    access_token?: string;
  };
};

export class RuntimeApiError extends Error {
  readonly status?: number;
  readonly code: string;
  readonly cause?: unknown;

  constructor(
    code: string,
    message: string,
    options: { status?: number; cause?: unknown } = {},
  ) {
    super(message);
    this.name = "RuntimeApiError";
    this.code = code;
    this.status = options.status;
    this.cause = options.cause;
  }
}

function isRecord(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function validateRuntimeId(id: string, field: string): string {
  if (!SAFE_RUNTIME_ID.test(id)) {
    throw new RuntimeApiError("INVALID_RUNTIME_ID", `${field} không hợp lệ`);
  }
  return id;
}

export function createRuntimeId(): string {
  if (!globalThis.crypto?.randomUUID) {
    throw new RuntimeApiError(
      "SECURE_RANDOM_UNAVAILABLE",
      "Trình duyệt không hỗ trợ nguồn random an toàn",
    );
  }
  return globalThis.crypto.randomUUID().replaceAll("-", "");
}

async function authorizationHeaders(token?: string): Promise<Headers> {
  let accessToken = token?.trim();
  if (!accessToken) {
    const session = (await getSession()) as SessionWithAccessToken | null;
    accessToken = session?.user?.access_token?.trim();
  }
  if (!accessToken) {
    throw new RuntimeApiError("AUTH_REQUIRED", "Bạn cần đăng nhập để chạy HAgent");
  }

  return new Headers({
    Accept: "text/event-stream",
    Authorization: `Bearer ${accessToken}`,
  });
}

async function responseError(response: Response): Promise<RuntimeApiError> {
  let code = "RUNTIME_REQUEST_FAILED";
  try {
    const payload: unknown = await response.json();
    if (isRecord(payload) && isRecord(payload.detail)) {
      const candidate = payload.detail.code;
      if (typeof candidate === "string" && SAFE_ERROR_CODE.test(candidate)) {
        code = candidate;
      }
    }
  } catch {
    // Nội dung lỗi không theo hợp đồng JSON công khai; chỉ giữ mã an toàn.
  }
  return new RuntimeApiError(code, `HAgent trả HTTP ${response.status}`, {
    status: response.status,
  });
}

async function consumeRuntimeStream(
  response: Response,
  expectedRunId: string,
  options: RuntimeRequestOptions,
): Promise<RuntimeStreamResult> {
  if (!response.ok) throw await responseError(response);
  const contentType = response.headers.get("content-type")?.toLowerCase() ?? "";
  if (!contentType.startsWith("text/event-stream") || !response.body) {
    await response.body?.cancel();
    throw new RuntimeApiError(
      "INVALID_RUNTIME_STREAM",
      "HAgent không trả SSE stream hợp lệ",
      { status: response.status },
    );
  }

  const headerRunId = response.headers.get("x-run-id")?.trim();
  if (!headerRunId || validateRuntimeId(headerRunId, "X-Run-Id") !== expectedRunId) {
    await response.body.cancel();
    throw new RuntimeApiError(
      "RUN_ID_MISMATCH",
      "X-Run-Id không khớp request hiện tại",
    );
  }

  const parser = new RuntimeSseParser({
    afterSequence: options.afterSequence,
    expectedRunId,
  });
  const events: RuntimeEvent[] = [];
  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (value) {
        for (const event of parser.feed(decoder.decode(value, { stream: true }))) {
          events.push(event);
          options.onEvent?.(event);
        }
      }
      if (done) {
        const tail = decoder.decode();
        if (tail) {
          for (const event of parser.feed(tail)) {
            events.push(event);
            options.onEvent?.(event);
          }
        }
        break;
      }
    }
    parser.finish();
  } catch (error) {
    try {
      await reader.cancel();
    } catch {
      // Luồng đã lỗi; chỉ cố gắng hủy để giải phóng tài nguyên.
    }
    if (error instanceof Error && error.name === "AbortError") throw error;
    if (error instanceof RuntimeApiError) throw error;
    throw new RuntimeApiError(
      "INVALID_RUNTIME_STREAM",
      "Không thể đọc HAgent runtime stream",
      { cause: error },
    );
  } finally {
    reader.releaseLock();
  }

  return {
    runId: expectedRunId,
    lastSequence: parser.sequence,
    events,
  };
}

async function runRequest(
  method: "GET" | "POST",
  url: string,
  expectedRunId: string,
  options: RuntimeRequestOptions,
  body?: JsonObject,
): Promise<RuntimeStreamResult> {
  const headers = await authorizationHeaders(options.token);
  if (body) headers.set("Content-Type", "application/json");

  let response: Response;
  try {
    response = await fetch(url, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
      signal: options.signal,
      cache: "no-store",
    });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") throw error;
    throw new RuntimeApiError(
      "RUNTIME_UNAVAILABLE",
      "Không thể kết nối HAgent runtime",
      { cause: error },
    );
  }
  return consumeRuntimeStream(response, expectedRunId, options);
}

export async function startRun(
  input: StartRunInput,
  options: RuntimeRequestOptions = {},
): Promise<RuntimeStreamResult> {
  const runId = validateRuntimeId(input.run_id ?? createRuntimeId(), "run_id");
  const commandId = validateRuntimeId(
    input.command_id ?? createRuntimeId(),
    "command_id",
  );
  return runRequest("POST", RUNS_ENDPOINT, runId, options, {
    message: input.message,
    run_id: runId,
    command_id: commandId,
    history: input.history ? [...input.history] : [],
    ...(input.model ? { model: input.model } : {}),
  });
}

export async function replayRun(
  runId: string,
  afterSequence: number,
  options: Omit<RuntimeRequestOptions, "afterSequence"> = {},
): Promise<RuntimeStreamResult> {
  validateRuntimeId(runId, "run_id");
  if (!Number.isSafeInteger(afterSequence) || afterSequence < 0) {
    throw new RuntimeApiError("INVALID_SEQUENCE", "afterSequence không hợp lệ");
  }
  const headers = await authorizationHeaders(options.token);
  headers.set("Last-Event-ID", String(afterSequence));
  const url = `${RUNS_ENDPOINT}/${encodeURIComponent(runId)}/events?after_sequence=${afterSequence}`;

  let response: Response;
  try {
    response = await fetch(url, {
      method: "GET",
      headers,
      signal: options.signal,
      cache: "no-store",
    });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") throw error;
    throw new RuntimeApiError(
      "RUNTIME_UNAVAILABLE",
      "Không thể replay HAgent runtime",
      { cause: error },
    );
  }
  return consumeRuntimeStream(response, runId, {
    ...options,
    afterSequence,
  });
}

export function resolveApproval(
  runId: string,
  approvalId: string,
  input: ResolveApprovalInput,
  options: RuntimeRequestOptions = {},
): Promise<RuntimeStreamResult> {
  validateRuntimeId(runId, "run_id");
  validateRuntimeId(approvalId, "approval_id");
  const commandId = validateRuntimeId(
    input.command_id ?? createRuntimeId(),
    "command_id",
  );
  return runRequest(
    "POST",
    `${RUNS_ENDPOINT}/${encodeURIComponent(runId)}/approvals/${encodeURIComponent(approvalId)}`,
    runId,
    options,
    {
      approved: input.approved,
      command_id: commandId,
      response: input.response ?? {},
    },
  );
}

export function cancelRun(
  runId: string,
  input: CancelRunInput = {},
  options: RuntimeRequestOptions = {},
): Promise<RuntimeStreamResult> {
  validateRuntimeId(runId, "run_id");
  const commandId = validateRuntimeId(
    input.command_id ?? createRuntimeId(),
    "command_id",
  );
  return runRequest(
    "POST",
    `${RUNS_ENDPOINT}/${encodeURIComponent(runId)}/cancel`,
    runId,
    options,
    { command_id: commandId },
  );
}
