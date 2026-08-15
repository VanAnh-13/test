"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { signIn, useSession } from "next-auth/react";

import {
  RuntimeApiError,
  cancelRun,
  createRuntimeId,
  replayRun,
  resolveApproval,
  startRun,
  type RuntimeEvent,
  type RuntimeStreamResult,
} from "@/lib/hagentRuntime";

import { ChatCanvas, type WorkspaceMessage } from "./ChatCanvas";
import { ConversationRail } from "./ConversationRail";
import { RunLedger } from "./RunLedger";

type WorkspaceStatus =
  | "idle"
  | "running"
  | "awaiting_approval"
  | "completed"
  | "failed"
  | "cancelled"
  | "disconnected";

const statusLabels: Record<WorkspaceStatus, string> = {
  idle: "Sẵn sàng",
  running: "Đang xử lý",
  awaiting_approval: "Chờ phê duyệt",
  completed: "Đã hoàn tất",
  failed: "Thất bại",
  cancelled: "Đã hủy",
  disconnected: "Mất kết nối",
};

const terminalTypes = new Set<RuntimeEvent["type"]>([
  "run_completed",
  "run_failed",
  "run_cancelled",
]);

function publicResultMessage(event: RuntimeEvent): string | null {
  if (event.type === "run_failed") return event.message.slice(0, 4000);
  if (event.type === "run_cancelled") {
    return `Run đã dừng: ${event.reason.slice(0, 4000)}`;
  }
  if (event.type !== "run_completed") return null;
  for (const field of ["summary", "message", "recommendation"] as const) {
    const value = event.result[field];
    if (typeof value === "string" && value.trim()) return value.slice(0, 4000);
  }
  return "Run đã hoàn tất. Artifact và bằng chứng đã được ghi trong ledger.";
}

function publicError(error: unknown): string {
  if (!(error instanceof RuntimeApiError)) {
    return "Không thể hoàn tất yêu cầu. Hãy nhận lại event trước khi thử thao tác khác.";
  }
  const messages: Record<string, string> = {
    AUTH_REQUIRED: "Phiên đăng nhập không còn hợp lệ. Hãy đăng nhập lại.",
    RUN_NOT_FOUND: "Run không tồn tại hoặc không thuộc tài khoản hiện tại.",
    RUNTIME_UNAVAILABLE: "Runtime đang mất kết nối. HAgent sẽ tiếp tục từ sequence cuối khi kết nối lại.",
    COMMAND_ID_CONFLICT: "Command ID đã được dùng cho một thao tác khác.",
    COMMAND_REPLAY_EXPIRED: "Cửa sổ replay của command đã hết hạn.",
  };
  return messages[error.code] ?? "HAgent không thể xử lý yêu cầu hiện tại.";
}

function isDisconnected(error: unknown) {
  return error instanceof RuntimeApiError && error.code === "RUNTIME_UNAVAILABLE";
}

export function HAgentWorkspace() {
  const { status: sessionStatus } = useSession();
  const [messages, setMessages] = useState<WorkspaceMessage[]>([]);
  const [events, setEvents] = useState<RuntimeEvent[]>([]);
  const [runId, setRunId] = useState<string | null>(null);
  const [lastSequence, setLastSequence] = useState(0);
  const [status, setStatus] = useState<WorkspaceStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [isRequestActive, setIsRequestActive] = useState(false);
  const [activeApprovalId, setActiveApprovalId] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const lastSequenceRef = useRef(0);
  const lastEventTypeRef = useRef<RuntimeEvent["type"] | null>(null);
  const approvalSequenceRef = useRef(0);
  const actionIdsRef = useRef(new Map<string, string>());

  const appendEvent = useCallback((event: RuntimeEvent) => {
    if (event.sequence <= lastSequenceRef.current) return;
    lastSequenceRef.current = event.sequence;
    lastEventTypeRef.current = event.type;
    setLastSequence(event.sequence);
    setEvents((current) => [...current, event]);

    if (event.type === "approval_required") {
      approvalSequenceRef.current = event.sequence;
      setActiveApprovalId(event.approval_id);
      setStatus("awaiting_approval");
    } else {
      if (event.sequence > approvalSequenceRef.current) setActiveApprovalId(null);
      if (event.type === "run_completed") setStatus("completed");
      else if (event.type === "run_failed") setStatus("failed");
      else if (event.type === "run_cancelled") setStatus("cancelled");
      else setStatus("running");
    }

    const assistantMessage = publicResultMessage(event);
    if (assistantMessage) {
      setMessages((current) => [
        ...current,
        { id: `event-${event.sequence}`, role: "assistant", content: assistantMessage },
      ]);
    }
  }, []);

  const settleStream = useCallback((result: RuntimeStreamResult) => {
    const lastType = result.events.at(-1)?.type ?? lastEventTypeRef.current;
    if (lastType !== "approval_required" && (!lastType || !terminalTypes.has(lastType))) {
      setStatus("disconnected");
      setError("Luồng event kết thúc trước terminal event. Hãy nhận lại event từ ledger.");
    }
  }, []);

  const prepareController = useCallback(() => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    return controller;
  }, []);

  const requestFailed = useCallback((requestError: unknown, fallback: WorkspaceStatus) => {
    if (requestError instanceof Error && requestError.name === "AbortError") return;
    setError(publicError(requestError));
    setStatus(isDisconnected(requestError) ? "disconnected" : fallback);
  }, []);

  const handleSubmit = useCallback(
    async (message: string) => {
      const controller = prepareController();
      const nextRunId = createRuntimeId();
      const commandId = createRuntimeId();
      const history = messages.slice(-20).map(({ role, content }) => ({ role, content }));
      lastSequenceRef.current = 0;
      lastEventTypeRef.current = null;
      approvalSequenceRef.current = 0;
      actionIdsRef.current.clear();
      setRunId(nextRunId);
      setEvents([]);
      setLastSequence(0);
      setActiveApprovalId(null);
      setError(null);
      setStatus("running");
      setIsRequestActive(true);
      setMessages((current) => [
        ...current,
        { id: `user-${commandId}`, role: "user", content: message },
      ]);
      try {
        const result = await startRun(
          { message, run_id: nextRunId, command_id: commandId, history },
          { signal: controller.signal, onEvent: appendEvent },
        );
        settleStream(result);
      } catch (requestError) {
        requestFailed(requestError, "failed");
      } finally {
        if (controllerRef.current === controller) setIsRequestActive(false);
      }
    },
    [appendEvent, messages, prepareController, requestFailed, settleStream],
  );

  const handleReplay = useCallback(async () => {
    if (!runId || isRequestActive) return;
    const controller = prepareController();
    setError(null);
    setIsRequestActive(true);
    try {
      const result = await replayRun(runId, lastSequenceRef.current, {
        signal: controller.signal,
        onEvent: appendEvent,
      });
      settleStream(result);
    } catch (requestError) {
      requestFailed(requestError, status);
    } finally {
      if (controllerRef.current === controller) setIsRequestActive(false);
    }
  }, [appendEvent, isRequestActive, prepareController, requestFailed, runId, settleStream, status]);

  const commandIdFor = useCallback((key: string) => {
    const existing = actionIdsRef.current.get(key);
    if (existing) return existing;
    const commandId = createRuntimeId();
    actionIdsRef.current.set(key, commandId);
    return commandId;
  }, []);

  const handleResolveApproval = useCallback(
    async (approvalId: string, approved: boolean) => {
      if (!runId || isRequestActive) return;
      const key = `approval:${approvalId}:${approved}`;
      const controller = prepareController();
      setError(null);
      setIsRequestActive(true);
      try {
        const result = await resolveApproval(
          runId,
          approvalId,
          { approved, command_id: commandIdFor(key) },
          { signal: controller.signal, afterSequence: lastSequenceRef.current, onEvent: appendEvent },
        );
        actionIdsRef.current.delete(key);
        settleStream(result);
      } catch (requestError) {
        requestFailed(requestError, "awaiting_approval");
      } finally {
        if (controllerRef.current === controller) setIsRequestActive(false);
      }
    },
    [appendEvent, commandIdFor, isRequestActive, prepareController, requestFailed, runId, settleStream],
  );

  const handleCancel = useCallback(async () => {
    if (!runId || isRequestActive) return;
    const key = `cancel:${runId}`;
    const controller = prepareController();
    setError(null);
    setIsRequestActive(true);
    try {
      const result = await cancelRun(
        runId,
        { command_id: commandIdFor(key) },
        { signal: controller.signal, afterSequence: lastSequenceRef.current, onEvent: appendEvent },
      );
      actionIdsRef.current.delete(key);
      settleStream(result);
    } catch (requestError) {
      requestFailed(requestError, status);
    } finally {
      if (controllerRef.current === controller) setIsRequestActive(false);
    }
  }, [appendEvent, commandIdFor, isRequestActive, prepareController, requestFailed, runId, settleStream, status]);

  const handleNewRun = useCallback(() => {
    controllerRef.current?.abort();
    lastSequenceRef.current = 0;
    lastEventTypeRef.current = null;
    approvalSequenceRef.current = 0;
    actionIdsRef.current.clear();
    setMessages([]);
    setEvents([]);
    setRunId(null);
    setLastSequence(0);
    setStatus("idle");
    setError(null);
    setActiveApprovalId(null);
    setIsRequestActive(false);
  }, []);

  useEffect(() => () => controllerRef.current?.abort(), []);
  useEffect(() => {
    const resume = () => {
      if (status === "disconnected") void handleReplay();
    };
    window.addEventListener("online", resume);
    return () => window.removeEventListener("online", resume);
  }, [handleReplay, status]);

  const statusLabel = statusLabels[status];
  const canReplay = Boolean(runId) && !terminalTypes.has(lastEventTypeRef.current ?? "run_started");
  const canCancel = Boolean(runId) && ["running", "awaiting_approval", "disconnected"].includes(status);
  const isAuthenticated = sessionStatus === "authenticated";
  const busy = isRequestActive || sessionStatus === "loading";

  return (
    <section className="grid min-h-[calc(100vh-7.5rem)] grid-cols-1 border-y border-border bg-background pb-24 lg:grid-cols-[16rem_minmax(0,1fr)_21rem] lg:pb-0">
      <ConversationRail
        runId={runId}
        statusLabel={statusLabel}
        lastSequence={lastSequence}
        onNewRun={handleNewRun}
      />
      <ChatCanvas
        messages={messages}
        isAuthenticated={isAuthenticated}
        isBusy={busy}
        statusLabel={statusLabel}
        error={error}
        onSubmit={handleSubmit}
        onSignIn={() => void signIn()}
      />
      <RunLedger
        events={events}
        statusLabel={statusLabel}
        activeApprovalId={activeApprovalId}
        isBusy={busy}
        canReplay={canReplay}
        canCancel={canCancel}
        onReplay={handleReplay}
        onResolveApproval={handleResolveApproval}
        onCancel={handleCancel}
      />
    </section>
  );
}
