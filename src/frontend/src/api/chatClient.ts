import axios from "axios";
import { getSession } from "next-auth/react";

// ─── HAgent Bridge Client ────────────────────────────
// Dùng riêng một axios instance cho HAgent Bridge service (host port 8900).
// Tách biệt hoàn toàn với axiosClient chính (HAutoML backend).

const HAGENT_BASE_URL =
  process.env.NEXT_PUBLIC_HAGENT_URL || "/api/hagent";

const hagentClient = axios.create({
  baseURL: HAGENT_BASE_URL,
});

// Bỏ interceptor để gửi token trực tiếp qua hàm sendChatMessage

// ─── Types ──────────────────────────────────────────────

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string | null;
  context?: Record<string, unknown>;
  /** LLM provider override */
  provider?: string | null;
  /** Tên model cụ thể muốn sử dụng */
  model?: string | null;
}

export interface WorldModelSummary {
  user_id?: string;
  phase?: string;
  active_dataset_id?: string | null;
  active_job_id?: string | null;
  n_datasets?: number;
  n_jobs?: number;
  dataset_ids?: string[];
  job_ids?: string[];
  last_surprise?: {
    value?: number;
    level?: string;
    metric?: string;
  } | null;
}

export interface SelectedPlanSummary {
  plan_id?: string;
  title?: string;
  steps?: Array<{ action?: { type?: string }; type?: string } | string>;
  cost?: number | Record<string, unknown>;
}

export interface ChatResponse {
  message: string;
  conversation_id: string;
  sources?: string[];
  suggestions?: string[];
  provider?: string;
  model?: string;
  plan_status?: string | null;
  selected_plan?: SelectedPlanSummary | null;
  surprise?: {
    value?: number;
    level?: string;
    metric?: string;
  } | null;
  cost_metrics?: Record<string, unknown> | null;
  execution_events?: Array<Record<string, unknown>> | null;
  world_model?: WorldModelSummary | null;
  campaign_status?: string | null;
  hierarchy_status?: string | null;
  evaluation?: {
    best_job_id?: string | null;
    recommendation?: string | null;
  } | null;
}

export type ChatStreamEventName =
  | "meta"
  | "route"
  | "phase"
  | "plan"
  | "plan_event"
  | "surprise"
  | "token"
  | "tool_call"
  | "tool_result"
  | "done"
  | "error";

export interface ChatStreamEvent {
  event: ChatStreamEventName;
  id: number;
  data: Record<string, unknown>;
}

export interface ChatStreamOptions {
  token?: string;
  signal?: AbortSignal;
  onEvent?: (event: ChatStreamEvent) => void;
  onConversationId?: (conversationId: string) => void;
}

export interface ProviderInfo {
  name: string;
  provider_id: string;
  models: string[];
  available: boolean;
  description?: string;
}

export interface ProvidersResponse {
  default_provider: string;
  default_model: string;
  providers: ProviderInfo[];
}

export interface SuggestionsResponse {
  suggestions: string[];
}

export interface HealthResponse {
  hagent_url: string;
  connected: boolean;
  hautoml_connected: boolean;
  mode: string;
}

// ─── API Functions ──────────────────────────────────────

const CHAT_ENDPOINT = "/api/v1/chat";

const STREAM_EVENT_NAMES = new Set<ChatStreamEventName>([
  "meta",
  "route",
  "phase",
  "plan",
  "plan_event",
  "surprise",
  "token",
  "tool_call",
  "tool_result",
  "done",
  "error",
]);

type SessionWithAccessToken = {
  user?: {
    access_token?: string;
  };
};

type ChatStreamErrorOptions = {
  status?: number;
  framesReceived?: boolean;
  code?: string;
  cause?: unknown;
};

export class ChatStreamError extends Error {
  readonly status?: number;
  readonly framesReceived: boolean;
  readonly code?: string;
  readonly cause?: unknown;

  constructor(message: string, options: ChatStreamErrorOptions = {}) {
    super(message);
    this.name = "ChatStreamError";
    this.status = options.status;
    this.framesReceived = options.framesReceived ?? false;
    this.code = options.code;
    this.cause = options.cause;
  }
}

export function isStreamUnsupportedError(error: unknown): boolean {
  return (
    error instanceof ChatStreamError &&
    !error.framesReceived &&
    (error.status === 404 || error.status === 415)
  );
}

async function getAuthorizationHeaders(
  token?: string,
): Promise<Record<string, string>> {
  const headers: Record<string, string> = {};
  let accessToken = token;
  if (!accessToken) {
    const session = (await getSession()) as SessionWithAccessToken | null;
    accessToken = session?.user?.access_token;
  }
  if (accessToken) {
    headers.Authorization = `Bearer ${accessToken}`;
  }
  return headers;
}

function getChatUrl(path: string): string {
  return `${HAGENT_BASE_URL.replace(/\/+$/, "")}${path}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseSseFrame(rawFrame: string): ChatStreamEvent {
  let eventName: string | undefined;
  let eventId: string | undefined;
  const dataLines: string[] = [];

  for (const line of rawFrame.split("\n")) {
    if (!line || line.startsWith(":")) continue;
    const separator = line.indexOf(":");
    if (separator < 0) {
      throw new Error("Malformed SSE field");
    }
    const field = line.slice(0, separator);
    const value = line.slice(separator + 1).replace(/^ /, "");
    if (field === "event") {
      if (eventName !== undefined) throw new Error("Duplicate SSE event field");
      eventName = value;
    } else if (field === "id") {
      if (eventId !== undefined) throw new Error("Duplicate SSE id field");
      eventId = value;
    } else if (field === "data") {
      dataLines.push(value);
    }
  }

  if (
    !eventName ||
    !STREAM_EVENT_NAMES.has(eventName as ChatStreamEventName) ||
    !eventId ||
    dataLines.length === 0
  ) {
    throw new Error("Incomplete SSE frame");
  }
  if (!/^\d+$/.test(eventId)) {
    throw new Error("Invalid SSE event id");
  }
  const id = Number(eventId);
  if (!Number.isSafeInteger(id) || id <= 0) {
    throw new Error("Invalid SSE event id");
  }

  const parsed: unknown = JSON.parse(dataLines.join("\n"));
  if (!isRecord(parsed) || parsed.type !== eventName) {
    throw new Error("SSE event/data mismatch");
  }
  return {
    event: eventName as ChatStreamEventName,
    id,
    data: parsed,
  };
}

function splitFirstSseFrame(
  buffer: string,
): { frame: string; rest: string } | null {
  const boundary = /\r?\n\r?\n/.exec(buffer);
  if (!boundary) return null;
  return {
    frame: buffer.slice(0, boundary.index).replace(/\r\n/g, "\n"),
    rest: buffer.slice(boundary.index + boundary[0].length),
  };
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

export async function sendChatMessage(
  req: ChatRequest,
  token?: string,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  } else {
    // Lấy thử từ next-auth nếu không truyền
    const session: any = await getSession();
    if (session?.user?.access_token) {
      headers["Authorization"] = `Bearer ${session.user.access_token}`;
    }
  }

  const { data } = await hagentClient.post<ChatResponse>(`${CHAT_ENDPOINT}/`, req, {
    headers,
    signal,
  });
  return data;
}

export async function sendChatMessageStream(
  req: ChatRequest,
  options: ChatStreamOptions = {},
): Promise<ChatResponse> {
  const headers = await getAuthorizationHeaders(options.token);
  headers["Content-Type"] = "application/json";
  headers.Accept = "text/event-stream";

  let response: Response;
  try {
    response = await fetch(getChatUrl(`${CHAT_ENDPOINT}/stream`), {
      method: "POST",
      headers,
      body: JSON.stringify(req),
      signal: options.signal,
      cache: "no-store",
    });
  } catch (error) {
    if (isAbortError(error)) throw error;
    throw new ChatStreamError("Unable to open chat stream", { cause: error });
  }

  if (!response.ok) {
    try {
      await response.body?.cancel();
    } catch {
      // Close the rejected stream before any supported sync fallback starts.
    }

    throw new ChatStreamError(
      `Chat stream returned HTTP ${response.status}`,
      { status: response.status },
    );
  }
  if (!response.body) {
    throw new ChatStreamError("Chat stream response has no body");
  }

  const headerConversationId =
    response.headers.get("X-Conversation-Id")?.trim() || null;

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let lastEventId = 0;
  let frameCount = 0;
  let terminalHandled = false;

  try {
    if (headerConversationId) {
      options.onConversationId?.(headerConversationId);
    }

    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });

      let split = splitFirstSseFrame(buffer);
      while (split) {
        buffer = split.rest;
        if (split.frame.trim()) {
          let event: ChatStreamEvent;
          try {
            event = parseSseFrame(split.frame);
          } catch (error) {
            throw new ChatStreamError("Invalid chat stream frame", {
              framesReceived: frameCount > 0,
              cause: error,
            });
          }

          if (event.id <= lastEventId) {
            throw new ChatStreamError("Chat stream IDs are not monotonic", {
              framesReceived: frameCount > 0,
            });
          }
          lastEventId = event.id;
          frameCount += 1;

          if (event.event === "done") {
            const responsePayload = event.data.response;
            if (!isRecord(responsePayload)) {
              throw new ChatStreamError("Chat stream done payload is invalid", {
                framesReceived: true,
              });
            }
            const responseConversationId =
              typeof responsePayload.conversation_id === "string"
                ? responsePayload.conversation_id
                : headerConversationId;
            if (
              headerConversationId &&
              responseConversationId &&
              headerConversationId !== responseConversationId
            ) {
              throw new ChatStreamError(
                "Chat stream conversation ID does not match its header",
                { framesReceived: true },
              );
            }
            if (
              typeof responsePayload.message !== "string" ||
              !responseConversationId
            ) {
              throw new ChatStreamError("Chat stream response is incomplete", {
                framesReceived: true,
              });
            }

            const result = {
              ...responsePayload,
              conversation_id: responseConversationId,
            } as unknown as ChatResponse;
            options.onEvent?.(event);
            try {
              await reader.cancel();
            } catch {
              // The terminal frame is already complete.
            }
            terminalHandled = true;
            return result;
          }

          options.onEvent?.(event);
          if (event.event === "error") {
            const errorPayload = isRecord(event.data.error)
              ? event.data.error
              : {};
            throw new ChatStreamError("Chat stream returned an error", {
              framesReceived: true,
              code:
                typeof errorPayload.code === "string"
                  ? errorPayload.code
                  : undefined,
            });
          }
        }
        split = splitFirstSseFrame(buffer);
      }

      if (done) break;
    }
  } catch (error) {
    if (error instanceof ChatStreamError || isAbortError(error)) {
      throw error;
    }
    throw new ChatStreamError("Unable to read chat stream", {
      framesReceived: frameCount > 0,
      cause: error,
    });
  } finally {
    if (!terminalHandled) {
      try {
        await reader.cancel();
      } catch {
        // Cancellation is best effort after a failed or aborted stream.
      }
    }
    reader.releaseLock();
  }

  throw new ChatStreamError(
    buffer.trim()
      ? "Chat stream ended with an incomplete frame"
      : "Chat stream ended without a terminal event",
    { framesReceived: frameCount > 0 },
  );
}

/**
 * Gửi tin nhắn kèm file đính kèm.
 * Dùng để upload dataset hoặc dữ liệu inference qua chat widget.
 */
export async function sendChatWithFile(
  message: string,
  file: File,
  conversationId?: string | null,
  token?: string,
  model?: string | null,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  const formData = new FormData();
  formData.append("message", message || `Upload file ${file.name}`);
  formData.append("file", file);
  if (conversationId) {
    formData.append("conversation_id", conversationId);
  }
  if (model) {
    formData.append("model", model);
  }

  const headers: Record<string, string> = {
    "Content-Type": "multipart/form-data",
  };
  
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  } else {
    // Lấy thử từ next-auth nếu không truyền
    const session: any = await getSession();
    if (session?.user?.access_token) {
      headers["Authorization"] = `Bearer ${session.user.access_token}`;
    }
  }

  const { data } = await hagentClient.post<ChatResponse>(
    `${CHAT_ENDPOINT}/upload`,
    formData,
    {
      headers,
      signal,
      timeout: 120_000, // upload file có thể mất lâu hơn
    }
  );
  return data;
}

export async function getInitialSuggestions(): Promise<SuggestionsResponse> {
  const { data } = await hagentClient.get<SuggestionsResponse>(`${CHAT_ENDPOINT}/suggestions`);
  return data;
}

/**
 * Kiểm tra trạng thái kết nối của HAgent gateway.
 */
export async function checkHAgentHealth(): Promise<HealthResponse> {
  const { data } = await hagentClient.get<HealthResponse>(`${CHAT_ENDPOINT}/health`);
  return data;
}

export async function getChatProviders(token?: string): Promise<ProvidersResponse> {
  const headers = await getAuthorizationHeaders(token);
  const { data } = await hagentClient.get<ProvidersResponse>(
    `${CHAT_ENDPOINT}/providers`,
    { headers },
  );
  return data;
}

export async function clearConversation(conversationId: string, token?: string): Promise<void> {
  const headers: Record<string, string> = {};
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  } else {
    const session: any = await getSession();
    if (session?.user?.access_token) {
      headers["Authorization"] = `Bearer ${session.user.access_token}`;
    }
  }
  await hagentClient.delete(`${CHAT_ENDPOINT}/conversation/${conversationId}`, { headers });
}


/**
 * Lấy danh sách các cuộc hội thoại gần đây.
 */
export async function getConversations(token?: string): Promise<any> {
  const headers: Record<string, string> = {};
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  } else {
    const session: any = await getSession();
    if (session?.user?.access_token) {
      headers["Authorization"] = `Bearer ${session.user.access_token}`;
    }
  }
  const { data } = await hagentClient.get(`${CHAT_ENDPOINT}/conversations`, { headers });
  return data;
}

/**
 * Lấy toàn bộ tin nhắn của một cuộc hội thoại cụ thể.
 */
export async function getConversationHistory(conversationId: string, token?: string): Promise<any> {
  const headers: Record<string, string> = {};
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  } else {
    const session: any = await getSession();
    if (session?.user?.access_token) {
      headers["Authorization"] = `Bearer ${session.user.access_token}`;
    }
  }
  const { data } = await hagentClient.get(`${CHAT_ENDPOINT}/conversation/${conversationId}`, { headers });
  return data;
}
