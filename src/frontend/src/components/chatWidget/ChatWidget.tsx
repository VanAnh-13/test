"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  MessageCircle,
  X,
  Trash2,
  ChevronDown,
  Send,
  Zap,
  Square,
  Paperclip,
  FileText,
} from "lucide-react";
import {
  sendChatMessage,
  sendChatMessageStream,
  sendChatWithFile,
  clearConversation,
  checkHAgentHealth,
  getChatProviders,
  getConversations,
  getConversationHistory,
  isStreamUnsupportedError,
  type ChatResponse,
  type ChatStreamEvent,
  type WorldModelSummary,
  type SelectedPlanSummary,
} from "@/api/chatClient";
import { useSession } from "next-auth/react";
import styles from "./ChatWidget.module.css";

// ─── Types ──────────────────────────────────────────
interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  time: string;
  fileName?: string;
  meta?: {
    model?: string;
    plan_status?: string | null;
    surprise?: ChatResponse["surprise"];
    campaign_status?: string | null;
    hierarchy_status?: string | null;
  };
}

interface ModelOption {
  value: string;
  label: string;
}

type ModelRegistryState = "idle" | "loading" | "ready" | "error";

interface WorldPanelState {
  world_model?: WorldModelSummary | null;
  selected_plan?: SelectedPlanSummary | null;
  plan_status?: string | null;
  surprise?: ChatResponse["surprise"];
  campaign_status?: string | null;
  hierarchy_status?: string | null;
  evaluation?: ChatResponse["evaluation"];
}

// ─── Constants ──────────────────────────────────────
const QUICK_ACTIONS = [
  { emoji: "📁", label: "Xem danh sách datasets" },
  { emoji: "🚀", label: "Train model mới" },
  { emoji: "⚡", label: "Kiểm tra trạng thái hệ thống" },
  { emoji: "🧬", label: "Thuật toán nào phù hợp cho phân loại?" },
] as const;

const ACCEPTED_FILE_TYPES = ".csv,.xls,.xlsx";
const MAX_FILE_SIZE_MB = 50;

// ─── Helpers ────────────────────────────────────────
function generateId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

function getCurrentTime(): string {
  const now = new Date();
  return `${now.getHours().toString().padStart(2, "0")}:${now.getMinutes().toString().padStart(2, "0")}`;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

function streamActivityForEvent(event: ChatStreamEvent): string {
  const agent =
    typeof event.data.agent === "string" ? event.data.agent : "agent";
  const phase =
    typeof event.data.phase === "string" ? event.data.phase : "processing";
  const tool =
    typeof event.data.tool === "string" ? event.data.tool : "tool";

  switch (event.event) {
    case "route":
      return `Routing to ${agent}`;
    case "phase":
      return `Phase: ${phase}`;
    case "plan":
      return "Plan created";
    case "plan_event":
      return "Executing plan";
    case "surprise":
      return "Evaluating result";
    case "tool_call":
      return `Running tool: ${tool}`;
    case "tool_result":
      return `Tool completed: ${tool}`;
    case "token":
      return "Generating response";
    case "done":
      return "Complete";
    case "error":
      return "Response stream failed";
    default:
      return "Processing";
  }
}

function mapHistoryMessages(rawMessages: any[]): Message[] {
  return (rawMessages || []).map((m: any, index: number) => {
    const parsedTime = m?.timestamp ? new Date(m.timestamp) : null;
    const displayTime =
      parsedTime && !Number.isNaN(parsedTime.getTime())
        ? parsedTime.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
        : getCurrentTime();

    return {
      id: `${m?.role || "assistant"}-${m?.timestamp || "no-ts"}-${index}`,
      role: m?.role === "assistant" ? "assistant" : "user",
      content: typeof m?.content === "string" ? m.content : "",
      time: displayTime,
      fileName:
        typeof m?.content === "string" && m.content.startsWith("Upload file")
          ? "Tệp tin"
          : undefined,
    };
  });
}

function extractErrorMessage(error: any): string {
  const status = error?.response?.status ?? error?.status;
  const detail = error?.response?.data?.detail;

  if (typeof detail === "string" && detail.length > 0) {
    if (
      detail.includes("Thiếu header Authorization") ||
      detail.includes("Not authenticated")
    ) {
      return "Bạn cần đăng nhập để nhắn tin với trợ lý AI.";
    }
    if (detail.includes("Token đã hết hạn")) {
      return "Phiên đăng nhập đã hết hạn. Vui lòng tải lại trang và đăng nhập lại.";
    }
    if (
      detail.includes("Loại token không hợp lệ") ||
      detail.includes("Token không hợp lệ") ||
      detail.includes("Invalid token") ||
      detail.includes("User not found")
    ) {
      return "Lỗi xác thực người dùng. Vui lòng đăng nhập lại.";
    }
    return detail;
  }

  if (status === 401 || status === 403) {
    return "Bạn cần đăng nhập để sử dụng HAgent.";
  }
  if (status === 502 || status === 503 || status === 504) {
    return "Không kết nối được tới HAgent Bridge. Vui lòng kiểm tra dịch vụ backend.";
  }

  return "Không thể kết nối đến server. Vui lòng kiểm tra kết nối mạng và thử lại.";
}

// ─── Component ──────────────────────────────────────
export default function ChatWidget() {
  const [mounted, setMounted] = useState(false);
  const { data: session } = useSession();
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isClearing, setIsClearing] = useState(false);
  const [streamActivity, setStreamActivity] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [showBadge, setShowBadge] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [modelOptions, setModelOptions] = useState<ModelOption[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [modelRegistryState, setModelRegistryState] = useState<ModelRegistryState>("idle");
  const [hagentStatus, setHagentStatus] = useState<{
    bridgeReachable: boolean;
    gatewayConnected: boolean;
    hautomlConnected: boolean;
  } | null>(null);
  const [worldPanel, setWorldPanel] = useState<WorldPanelState | null>(null);
  const [wmOpen, setWmOpen] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const activeRequestRef = useRef<AbortController | null>(null);
  const activeAssistantMessageIdRef = useRef<string | null>(null);

  useEffect(() => {
    return () => {
      activeRequestRef.current?.abort();
      activeRequestRef.current = null;
    };
  }, []);

  // Show notification badge after 3 seconds
  useEffect(() => {
    setMounted(true);
    const timer = setTimeout(() => {
      if (!isOpen) setShowBadge(true);
    }, 3000);
    return () => clearTimeout(timer);
  }, [isOpen]);

  // Kiểm tra trạng thái HAgent khi mở chat và thử tải lại hội thoại cũ
  useEffect(() => {
    let cancelled = false;

    if (isOpen && hagentStatus === null && !isClearing) {
      void checkHAgentHealth()
        .then((health) => {
          if (cancelled) return;
          setHagentStatus({
            bridgeReachable: true,
            gatewayConnected: health.connected,
            hautomlConnected: health.hautoml_connected,
          });
        })
        .catch(() => {
          if (cancelled) return;
          setHagentStatus({
            bridgeReachable: false,
            gatewayConnected: false,
            hautomlConnected: false,
          });
        });
    }

    return () => {
      cancelled = true;
    };
  }, [isOpen, hagentStatus, isClearing]);

  useEffect(() => {
    if (!isOpen || modelRegistryState !== "idle") return;

    setModelRegistryState("loading");
    const token = (session?.user as any)?.access_token;
    void getChatProviders(token)
      .then((registry) => {
        const seen = new Set<string>();
        const options = registry.providers.flatMap((provider) => {
          if (!provider.available) return [];
          return provider.models.flatMap((model) => {
            if (!model || seen.has(model)) return [];
            seen.add(model);
            return [{ value: model, label: `${model} / ${provider.name}` }];
          });
        });
        if (options.length === 0) {
          throw new Error("No available chat models");
        }
        setModelOptions(options);
        setSelectedModel((current) => {
          if (current && options.some((option) => option.value === current)) {
            return current;
          }
          return (
            options.find((option) => option.value === registry.default_model)
              ?.value ?? options[0].value
          );
        });
        setModelRegistryState("ready");
      })
      .catch(() => {
        setModelOptions([]);
        setSelectedModel(null);
        setModelRegistryState("error");
      });
  }, [isOpen, modelRegistryState, session]);

  // Tải lịch sử cuộc hội thoại gần nhất khi mở chat
  useEffect(() => {
    let cancelled = false;

    if (
      isOpen &&
      !conversationId &&
      hagentStatus !== null &&
      !isClearing
    ) {
      const loadHistory = async () => {
        try {
          const token = (session?.user as any)?.access_token;
          const list = await getConversations(token);
          if (cancelled) return;
          if (list?.conversations?.length > 0) {
            const latestId = list.conversations[0].conversation_id;
            const historyData = await getConversationHistory(latestId, token);
            if (cancelled) return;
            if (!cancelled && historyData && historyData.messages?.length > 0) {
              setConversationId(historyData.conversation_id);
              setMessages(mapHistoryMessages(historyData.messages));
            }
          }
        } catch {
          console.log("Không tìm thấy hội thoại cũ hoặc không tải được");
        }
      };
      // Chỉ tải khi danh sách tin nhắn hiện tại đang trống
      if (messages.length === 0) {
        void loadHistory();
      }
    }
    return () => {
      cancelled = true;
    };

  }, [
    isOpen,
    conversationId,
    hagentStatus,
    isClearing,
    messages.length,
    session,
  ]);

  // Đồng bộ tin nhắn mới từ server theo chu kỳ để hiển thị thông báo hậu huấn luyện.
  useEffect(() => {
    if (!isOpen || !conversationId || isLoading || isClearing) return;

    let cancelled = false;

    const syncMessages = async () => {
      try {
        const token = (session?.user as any)?.access_token;
        const historyData = await getConversationHistory(conversationId, token);

        if (cancelled || !historyData?.messages) return;

        const nextMessages = mapHistoryMessages(historyData.messages);

        setMessages((prev) => {
          if (nextMessages.length === 0) return prev;

          const prevLast = prev[prev.length - 1];
          const nextLast = nextMessages[nextMessages.length - 1];

          const unchanged =
            prev.length === nextMessages.length &&
            prevLast?.role === nextLast?.role &&
            prevLast?.content === nextLast?.content &&
            prevLast?.time === nextLast?.time;

          return unchanged ? prev : nextMessages;
        });
      } catch {
        // Ignore polling errors to avoid disrupting chat UX.
      }
    };

    void syncMessages();
    const intervalId = window.setInterval(syncMessages, 8000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [isOpen, conversationId, isClearing, isLoading, session]);

  // Handle textarea auto-resize
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 150)}px`;
    }
  }, [input]);

  // Auto-scroll to bottom when new message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  // Focus input when chat opens
  useEffect(() => {
    if (isOpen) {
      const timer = setTimeout(() => inputRef.current?.focus(), 300);
      return () => clearTimeout(timer);
    }
  }, [isOpen]);

  const cancelActiveRequest = useCallback((activity: string | null = null) => {
    const controller = activeRequestRef.current;
    activeRequestRef.current = null;
    controller?.abort();

    const assistantId = activeAssistantMessageIdRef.current;
    activeAssistantMessageIdRef.current = null;
    if (assistantId) {
      setMessages((previous) =>
        previous.filter((message) => message.id !== assistantId),
      );
    }
    setIsLoading(false);
    setIsStreaming(false);
    setStreamActivity(activity);
  }, []);

  const handleStop = useCallback(() => {
    cancelActiveRequest("Response stopped");
  }, [cancelActiveRequest]);

  const toggleChat = useCallback(() => {
    if (isOpen) {
      cancelActiveRequest();
      setIsOpen(false);
      return;
    }
    setShowBadge(false);
    setIsOpen(true);
  }, [cancelActiveRequest, isOpen]);

  // File selection handler
  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
        alert(`File quá lớn. Tối đa ${MAX_FILE_SIZE_MB}MB.`);
        return;
      }

      setSelectedFile(file);
      if (!input.trim()) {
        setInput(`Upload file ${file.name} vào hệ thống`);
      }
    },
    [input]
  );

  const clearFile = useCallback(() => {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }, []);

  const updateAssistantMessage = useCallback(
    (
      id: string,
      content: string,
      options: { append?: boolean; meta?: Message["meta"] } = {},
    ) => {
      setMessages((previous) => {
        const existingIndex = previous.findIndex((message) => message.id === id);
        if (existingIndex < 0) {
          return [
            ...previous,
            {
              id,
              role: "assistant",
              content,
              time: getCurrentTime(),
              meta: options.meta,
            },
          ];
        }
        return previous.map((message, index) =>
          index === existingIndex
            ? {
                ...message,
                content: options.append
                  ? `${message.content}${content}`
                  : content,
                meta: options.meta ?? message.meta,
              }
            : message,
        );
      });
    },
    [],
  );

  const applyResponseMetadata = useCallback((response: ChatResponse) => {
    if (
      response.world_model ||
      response.selected_plan ||
      response.surprise ||
      response.plan_status ||
      response.campaign_status ||
      response.hierarchy_status ||
      response.evaluation
    ) {
      setWorldPanel({
        world_model: response.world_model,
        selected_plan: response.selected_plan,
        plan_status: response.plan_status,
        surprise: response.surprise,
        campaign_status: response.campaign_status,
        hierarchy_status: response.hierarchy_status,
        evaluation: response.evaluation,
      });
      setWmOpen(true);
    }
    if (response.suggestions?.length) {
      setSuggestions(response.suggestions);
    }
  }, []);

  const applyStreamMetadata = useCallback((event: ChatStreamEvent) => {
    if (event.event === "phase" && typeof event.data.phase === "string") {
      setWorldPanel((previous) => ({
        ...(previous ?? {}),
        world_model: {
          ...(previous?.world_model ?? {}),
          phase: event.data.phase as string,
        },
      }));
      setWmOpen(true);
      return;
    }

    if (event.event === "plan") {
      const steps = Array.isArray(event.data.steps)
        ? event.data.steps.filter(
            (step): step is string => typeof step === "string",
          )
        : [];
      setWorldPanel((previous) => ({
        ...(previous ?? {}),
        plan_status: "streaming",
        selected_plan: {
          plan_id:
            typeof event.data.plan_id === "string"
              ? event.data.plan_id
              : undefined,
          title:
            typeof event.data.title === "string"
              ? event.data.title
              : undefined,
          steps,
          cost:
            typeof event.data.cost === "number" || isRecord(event.data.cost)
              ? event.data.cost
              : undefined,
        },
      }));
      setWmOpen(true);
      return;
    }

    if (event.event === "surprise" && isRecord(event.data.surprise)) {
      setWorldPanel((previous) => ({
        ...(previous ?? {}),
        surprise: event.data.surprise as ChatResponse["surprise"],
      }));
      setWmOpen(true);
    }
  }, []);
  const dispatchMessage = useCallback(
    async (text: string, fileToSend: File | null) => {
      const trimmed = text.trim();
      if (
        (!trimmed && !fileToSend) ||
        isLoading ||
        isClearing ||
        !selectedModel ||
        activeRequestRef.current
      ) {
        return;
      }

      const controller = new AbortController();
      const assistantMessageId = fileToSend ? null : generateId();
      activeRequestRef.current = controller;
      activeAssistantMessageIdRef.current = assistantMessageId;
      const userMsg: Message = {
        id: generateId(),
        role: "user",
        content: trimmed || `📎 ${fileToSend?.name}`,
        time: getCurrentTime(),
        fileName: fileToSend?.name,
      };

      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);
      setIsStreaming(!fileToSend);
      setStreamActivity(
        fileToSend ? "Uploading file" : "Connecting to HAgent",
      );
      setSuggestions([]);

      try {
        const token = (session?.user as any)?.access_token;
        const request = {
          message: trimmed,
          conversation_id: conversationId,
          model: selectedModel,
        };
        let response: ChatResponse;

        if (fileToSend) {
          response = await sendChatWithFile(
            trimmed,
            fileToSend,
            conversationId,
            token,
            selectedModel,
            controller.signal,
          );
        } else {
          try {
            response = await sendChatMessageStream(request, {
              token,
              signal: controller.signal,
              onConversationId: (nextConversationId) => {
                if (activeRequestRef.current === controller) {
                  setConversationId(nextConversationId);
                }
              },
              onEvent: (event) => {
                if (activeRequestRef.current !== controller) return;
                setStreamActivity(streamActivityForEvent(event));
                applyStreamMetadata(event);
                if (
                  event.event === "token" &&
                  typeof event.data.content === "string" &&
                  assistantMessageId
                ) {
                  updateAssistantMessage(
                    assistantMessageId,
                    event.data.content,
                    { append: true },
                  );
                }
              },
            });
          } catch (error) {
            if (!isStreamUnsupportedError(error)) throw error;
            setStreamActivity("Streaming unavailable; using sync response");
            response = await sendChatMessage(
              request,
              token,
              controller.signal,
            );
          }
        }

        if (activeRequestRef.current !== controller) {
          return;
        }

        setConversationId(response.conversation_id);
        const finalAssistantId = assistantMessageId ?? generateId();
        updateAssistantMessage(finalAssistantId, response.message, {
          meta: {
            model: response.model || selectedModel,
            plan_status: response.plan_status,
            surprise: response.surprise,
            campaign_status: response.campaign_status,
            hierarchy_status: response.hierarchy_status,
          },
        });
        activeAssistantMessageIdRef.current = null;
        applyResponseMetadata(response);
      } catch (error: unknown) {
        if (activeRequestRef.current !== controller || isAbortError(error)) {
          return;
        }
        const errorAssistantId = assistantMessageId ?? generateId();
        updateAssistantMessage(
          errorAssistantId,
          `Error: ${extractErrorMessage(error)}`,
        );
        activeAssistantMessageIdRef.current = null;
      } finally {
        if (activeRequestRef.current === controller) {
          activeRequestRef.current = null;
          activeAssistantMessageIdRef.current = null;
          setIsLoading(false);
          setIsStreaming(false);
          setStreamActivity(null);
        }
      }
    },
    [
      applyResponseMetadata,
      applyStreamMetadata,
      conversationId,
      isLoading,
      isClearing,
      selectedModel,
      session,
      updateAssistantMessage,
    ],
  );

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (
      (!text && !selectedFile) ||
      isLoading ||
      isClearing ||
      !selectedModel
    ) {
      return;
    }

    const fileToSend = selectedFile;
    setInput("");
    setSelectedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = "";

    await dispatchMessage(text, fileToSend);
  }, [
    dispatchMessage,
    input,
    isClearing,
    isLoading,
    selectedFile,
    selectedModel,
  ]);

  const handleQuickSend = useCallback(
    (text: string) => {
      void dispatchMessage(text, null);
    },
    [dispatchMessage],
  );

  const handleClear = useCallback(async () => {
    if (isClearing) return;
    const conversationToClear = conversationId;
    cancelActiveRequest();

    setIsClearing(true);
    setMessages([]);
    setSuggestions([]);
    setConversationId(null);
    setInput("");
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    setHagentStatus(null);
    setWorldPanel(null);
    setWmOpen(false);

    try {
      if (conversationToClear) {
        const token = (session as any)?.user?.access_token;
        await clearConversation(conversationToClear, token);
      }
    } catch {
      // The local reset remains authoritative while the bridge is unavailable.
    } finally {
      setIsClearing(false);
    }
  }, [cancelActiveRequest, conversationId, isClearing, session]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  if (!mounted) return null;

  const showWelcome = messages.length === 0;

  return (
    <>
      {/* ── Floating Action Button ── */}
      <button
        className={`${styles.fab} ${isOpen ? styles.fabOpen : ""}`}
        onClick={toggleChat}
        aria-label={isOpen ? "Đóng chat" : "Mở chat assistant"}
      >
        {showBadge && !isOpen && <span className={styles.badge}>1</span>}
        {isOpen ? <X size={22} /> : <MessageCircle size={26} />}
      </button>

      {/* ── Chat Window ── */}
      <div className={`${styles.window} ${isOpen ? styles.windowVisible : ""}`}>
        {/* Header */}
        <div className={styles.header}>
          <div className={styles.headerAvatar}>
            <Zap size={20} />
          </div>
          <div className={styles.headerInfo}>
            <div className={styles.headerTitle}>HAgent</div>
            <div className={styles.headerStatus}>
              <span
                className={styles.statusDot}
                style={{
                  background:
                    hagentStatus === null
                      ? "#888"
                      : hagentStatus.bridgeReachable && hagentStatus.gatewayConnected
                        ? "#00d26a"
                        : hagentStatus.bridgeReachable
                          ? "#f59e0b"
                          : "#ef4444",
                }}
              />
              <span>
                {hagentStatus === null
                  ? "Checking..."
                  : hagentStatus.bridgeReachable && hagentStatus.gatewayConnected
                    ? "HAgent Connected"
                    : hagentStatus.bridgeReachable
                      ? "Bridge Ready · Gateway Connecting..."
                      : "Disconnected"}
              </span>
            </div>
          </div>
          <div className={styles.headerActions}>
            <button
              className={styles.headerBtn}
              onClick={handleClear}
              title="Xóa lịch sử"
              aria-label="Clear conversation"
              disabled={isClearing}
            >
              <Trash2 size={16} />
            </button>
            <button
              className={styles.headerBtn}
              onClick={toggleChat}
              title="Thu nhỏ"
              aria-label="Minimize chat"
            >
              <ChevronDown size={16} />
            </button>
          </div>
        </div>
        <div className={styles.modelBar}>
          <label htmlFor="hagent-chat-model">Model</label>
          <select
            id="hagent-chat-model"
            className={styles.modelSelect}
            value={selectedModel ?? ""}
            onChange={(event) => setSelectedModel(event.target.value)}
            disabled={
              modelRegistryState !== "ready" || isLoading || isClearing
            }
            aria-describedby="hagent-model-status"
          >
            <option value="" disabled>
              {modelRegistryState === "loading"
                ? "Loading models..."
                : "Select a model"}
            </option>
            {modelOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <span id="hagent-model-status" className={styles.modelStatus}>
            {modelRegistryState === "error" ? "Model registry unavailable" : ""}
          </span>
          {modelRegistryState === "error" && (
            <button
              type="button"
              className={styles.modelRetry}
              onClick={() => setModelRegistryState("idle")}
            >
              Retry
            </button>
          )}
        </div>

        {/* World Model panel */}
        {worldPanel && (
          <div className={styles.wmPanel}>
            <button
              type="button"
              className={styles.wmToggle}
              onClick={() => setWmOpen((v) => !v)}
            >
              <span>World Model</span>
              <span className={styles.wmChips}>
                {worldPanel.plan_status && (
                  <span className={styles.wmChip}>plan:{worldPanel.plan_status}</span>
                )}
                {worldPanel.surprise?.level && (
                  <span
                    className={`${styles.wmChip} ${
                      worldPanel.surprise.level === "high"
                        ? styles.wmChipHigh
                        : worldPanel.surprise.level === "medium"
                          ? styles.wmChipMed
                          : ""
                    }`}
                  >
                    surprise:{worldPanel.surprise.level}
                    {typeof worldPanel.surprise.value === "number"
                      ? ` ${worldPanel.surprise.value.toFixed(2)}`
                      : ""}
                  </span>
                )}
                {worldPanel.campaign_status && (
                  <span className={styles.wmChip}>
                    campaign:{worldPanel.campaign_status}
                  </span>
                )}
                {worldPanel.hierarchy_status && (
                  <span className={styles.wmChip}>
                    hierarchy:{worldPanel.hierarchy_status}
                  </span>
                )}
              </span>
              <ChevronDown
                size={14}
                className={wmOpen ? styles.wmChevronOpen : styles.wmChevron}
              />
            </button>
            {wmOpen && (
              <div className={styles.wmBody}>
                {worldPanel.world_model && (
                  <div className={styles.wmRow}>
                    <strong>State</strong>
                    <span>
                      phase={worldPanel.world_model.phase || "—"} · datasets=
                      {worldPanel.world_model.n_datasets ?? 0} · jobs=
                      {worldPanel.world_model.n_jobs ?? 0}
                      {worldPanel.world_model.active_dataset_id
                        ? ` · ds=${worldPanel.world_model.active_dataset_id}`
                        : ""}
                    </span>
                  </div>
                )}
                {worldPanel.selected_plan?.steps && (
                  <div className={styles.wmRow}>
                    <strong>Plan</strong>
                    <span>
                      {(worldPanel.selected_plan.steps || [])
                        .map((s) =>
                          typeof s === "string"
                            ? s
                            : s?.action?.type || s?.type || "?"
                        )
                        .filter(Boolean)
                        .join(" → ") || "—"}
                    </span>
                  </div>
                )}
                {worldPanel.evaluation?.best_job_id && (
                  <div className={styles.wmRow}>
                    <strong>Best job</strong>
                    <span>
                      {worldPanel.evaluation.best_job_id}
                      {worldPanel.evaluation.recommendation
                        ? ` (${worldPanel.evaluation.recommendation})`
                        : ""}
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Messages */}
        <div className={styles.messages} aria-busy={isLoading}>
          {showWelcome && (
            <div className={styles.welcome}>
              <div className={styles.welcomeIcon}>
                <Zap size={28} />
              </div>
              <h3 className={styles.welcomeTitle}>Xin chào! 👋</h3>
              <p className={styles.welcomeDesc}>
                Tôi là trợ lý AI của HAutoML — powered by{" "}
                <strong>HAgent</strong>. Tôi có thể quản lý datasets, train
                models, chạy predictions, và giám sát hệ thống cho bạn.
              </p>
              <div className={styles.welcomeActions}>
                {QUICK_ACTIONS.map((action) => (
                  <button
                    key={action.label}
                    className={styles.welcomeBtn}
                    onClick={() => handleQuickSend(action.label)}
                  >
                    <span>{action.emoji}</span> {action.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`${styles.message} ${
                msg.role === "user" ? styles.messageUser : styles.messageBot
              }`}
            >
              <div className={styles.msgAvatar}>
                {msg.role === "assistant" ? "AI" : "U"}
              </div>
              <div>
                {msg.fileName && (
                  <div className={styles.fileChip}>
                    <FileText size={14} />
                    <span>{msg.fileName}</span>
                  </div>
                )}
                <div className={`${styles.msgBubble} ${styles.markdownBody}`}>
                  {msg.content}
                </div>
                {msg.meta &&
                  (msg.meta.model ||
                    msg.meta.surprise?.level ||
                    msg.meta.plan_status ||
                    msg.meta.campaign_status) && (
                    <div className={styles.msgMeta}>
                      {msg.meta.model && (
                        <span className={styles.wmChip}>{msg.meta.model}</span>
                      )}
                      {msg.meta.plan_status && (
                        <span className={styles.wmChip}>
                          plan:{msg.meta.plan_status}
                        </span>
                      )}
                      {msg.meta.surprise?.level && (
                        <span
                          className={`${styles.wmChip} ${
                            msg.meta.surprise.level === "high"
                              ? styles.wmChipHigh
                              : ""
                          }`}
                        >
                          surprise:{msg.meta.surprise.level}
                        </span>
                      )}
                      {msg.meta.campaign_status && (
                        <span className={styles.wmChip}>
                          {msg.meta.campaign_status}
                        </span>
                      )}
                    </div>
                  )}
                <div className={styles.msgTime}>{msg.time}</div>
              </div>
            </div>
          ))}

          {isLoading && !isStreaming && (
            <div className={styles.typing}>
              <div className={styles.msgAvatar}>AI</div>
              <div className={styles.typingDots}>
                <span />
                <span />
                <span />
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
        {streamActivity && (
          <div
            className={styles.streamStatus}
            role="status"
            aria-live="polite"
            data-streaming={isLoading ? "true" : "false"}
          >
            {isLoading && <span className={styles.streamPulse} aria-hidden="true" />}
            <span className={styles.streamStatusText}>{streamActivity}</span>
            {isLoading && (
              <button
                type="button"
                className={styles.stopButton}
                onClick={handleStop}
              >
                <Square size={12} aria-hidden="true" />
                Stop
              </button>
            )}
          </div>
        )}

        {/* Suggestion chips */}
        {suggestions.length > 0 && (
          <div className={styles.suggestions}>
            {suggestions.map((s) => (
              <button
                key={s}
                className={styles.chip}
                onClick={() => handleQuickSend(s)}
              >
                {s}
              </button>
            ))}
          </div>
        )}

        {/* File preview bar */}
        {selectedFile && (
          <div className={styles.filePreview}>
            <FileText size={16} />
            <span className={styles.filePreviewName}>{selectedFile.name}</span>
            <span className={styles.filePreviewSize}>
              ({formatFileSize(selectedFile.size)})
            </span>
            <button
              className={styles.filePreviewClose}
              onClick={clearFile}
              title="Xóa file"
              aria-label="Remove selected file"
            >
              <X size={14} />
            </button>
          </div>
        )}

        {/* Input Area */}
        <div className={styles.inputArea}>
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPTED_FILE_TYPES}
            onChange={handleFileSelect}
            style={{ display: "none" }}
            id="chat-file-input"
          />
          <button
            className={styles.attachBtn}
            onClick={() => fileInputRef.current?.click()}
            title="Đính kèm file (CSV, Excel)"
            aria-label="Attach a CSV or Excel file"
            disabled={isLoading || isClearing}
          >
            <Paperclip size={18} />
          </button>
          <textarea
            ref={inputRef}
            className={styles.input}
            placeholder="Hỏi tôi bất cứ điều gì..."
            aria-label="Chat message"
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button
            className={styles.sendBtn}
            onClick={handleSend}
            disabled={
              (!input.trim() && !selectedFile) ||
              isLoading ||
              isClearing ||
              !selectedModel
            }
            aria-label="Gửi tin nhắn"
          >
            <Send size={18} />
          </button>
        </div>

        {/* Footer */}
        <div className={styles.footer}>
          🔬 Powered by{" "}
          <a
            href="https://optivisionlab.fit-haui.edu.vn/"
            target="_blank"
            rel="noopener noreferrer"
          >
            OptivisionLab
          </a>{" "}
          •{" "}
          <a
            href=""
            target="_blank"
            rel="noopener noreferrer"
          >
            HAgent
          </a>
        </div>
      </div>
    </>
  );
}
