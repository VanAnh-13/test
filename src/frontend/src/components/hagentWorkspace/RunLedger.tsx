"use client";

import {
  Ban,
  CheckCircle2,
  ClipboardCheck,
  FileOutput,
  History,
  ListChecks,
  PauseCircle,
  PlayCircle,
  RefreshCw,
  ShieldCheck,
  XCircle,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import type { JsonObject, RuntimeEvent } from "@/lib/hagentRuntime";

interface RunLedgerProps {
  events: readonly RuntimeEvent[];
  statusLabel: string;
  activeApprovalId: string | null;
  isBusy: boolean;
  canReplay: boolean;
  canCancel: boolean;
  onReplay: () => Promise<void>;
  onResolveApproval: (approvalId: string, approved: boolean) => Promise<void>;
  onCancel: () => Promise<void>;
}

const eventLabels: Record<RuntimeEvent["type"], string> = {
  run_started: "Bắt đầu run",
  plan_proposed: "Đề xuất kế hoạch",
  artifact_produced: "Tạo artifact",
  check_completed: "Checker hoàn tất",
  approval_required: "Chờ phê duyệt",
  action_completed: "Action hoàn tất",
  evidence_added: "Thêm bằng chứng",
  run_completed: "Run hoàn tất",
  run_failed: "Run thất bại",
  run_cancelled: "Run đã hủy",
};

const fieldLabels: Record<string, string> = {
  target: "Target",
  problem_type: "Bài toán",
  metric: "Metric",
  metric_direction: "Chiều metric",
  split_strategy: "Chiến lược split",
  hpo_budget: "Ngân sách HPO",
  dataset_id: "Dataset",
  version: "Phiên bản",
  supersedes: "Thay thế",
  status: "Trạng thái",
  readiness_verdict: "Kết luận",
  job_id: "Job",
  model_id: "Model",
  source: "Nguồn",
  uri: "URI",
  hash: "Hash",
  value: "Giá trị",
  description: "Mô tả",
  reason: "Lý do",
  cost: "Chi phí",
  baseline_delta: "Chênh lệch baseline",
};

const allowedFields = [
  "target",
  "problem_type",
  "metric",
  "metric_direction",
  "split_strategy",
  "hpo_budget",
  "dataset_id",
  "version",
  "supersedes",
  "status",
  "readiness_verdict",
  "job_id",
  "model_id",
  "source",
  "uri",
  "hash",
  "value",
  "description",
  "reason",
  "cost",
  "baseline_delta",
] as const;

function safeValue(value: unknown): string | null {
  if (typeof value === "string") return value.slice(0, 240);
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  if (typeof value === "boolean") return value ? "Có" : "Không";
  if (Array.isArray(value) && value.every((item) => typeof item === "string")) {
    return value.slice(0, 6).join(", ").slice(0, 240);
  }
  return null;
}

function rows(payload: JsonObject) {
  return allowedFields.flatMap((field) => {
    const value = safeValue(payload[field]);
    return value ? [{ label: fieldLabels[field], value }] : [];
  });
}

function eventPayload(event: RuntimeEvent): JsonObject | null {
  switch (event.type) {
    case "plan_proposed":
      return event.plan;
    case "artifact_produced":
      return event.artifact;
    case "check_completed":
      return event.details;
    case "approval_required":
      return event.proposal;
    case "action_completed":
      return event.details;
    case "evidence_added":
      return event.evidence;
    case "run_completed":
      return event.result;
    default:
      return null;
  }
}

function eventMeta(event: RuntimeEvent): string | null {
  switch (event.type) {
    case "artifact_produced":
      return event.artifact_type.slice(0, 240);
    case "check_completed":
      return `${event.checker} · ${event.verdict}`.slice(0, 240);
    case "action_completed":
      return `${event.action} · ${event.outcome}`.slice(0, 240);
    case "evidence_added":
      return event.evidence_type.slice(0, 240);
    case "run_failed":
      return `${event.error_code} · ${event.message}`.slice(0, 240);
    case "run_cancelled":
      return event.reason.slice(0, 240);
    default:
      return null;
  }
}

function EventIcon({ type }: { type: RuntimeEvent["type"] }) {
  const icons = {
    run_started: PlayCircle,
    plan_proposed: ListChecks,
    artifact_produced: FileOutput,
    check_completed: ShieldCheck,
    approval_required: PauseCircle,
    action_completed: ClipboardCheck,
    evidence_added: History,
    run_completed: CheckCircle2,
    run_failed: XCircle,
    run_cancelled: Ban,
  } as const;
  const Icon = icons[type];
  return <Icon className="size-4" aria-hidden="true" />;
}

export function RunLedger({
  events,
  statusLabel,
  activeApprovalId,
  isBusy,
  canReplay,
  canCancel,
  onReplay,
  onResolveApproval,
  onCancel,
}: RunLedgerProps) {
  return (
    <aside aria-label="Run ledger" className="border-t border-border bg-muted/20 lg:border-l lg:border-t-0">
      <header className="flex items-center justify-between gap-3 border-b border-border px-4 py-4">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
            Run ledger
          </p>
          <h2 className="mt-1 text-sm font-semibold">{statusLabel}</h2>
        </div>
        <div className="flex gap-1">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label="Nhận event mới"
            disabled={!canReplay || isBusy}
            onClick={() => void onReplay()}
          >
            <RefreshCw className={isBusy ? "animate-spin motion-reduce:animate-none" : ""} aria-hidden="true" />
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="text-destructive hover:text-destructive"
            aria-label="Hủy run"
            disabled={!canCancel || isBusy}
            onClick={() => void onCancel()}
          >
            <Ban aria-hidden="true" />
          </Button>
        </div>
      </header>

      <div className="max-h-[32rem] overflow-y-auto px-4 py-5 lg:max-h-[calc(100vh-13rem)]" aria-live="polite">
        {events.length === 0 ? (
          <div className="border border-dashed border-border p-4 text-sm leading-6 text-muted-foreground">
            Ledger đang trống. Event có sequence, artifact và bằng chứng thật sẽ xuất hiện tại đây.
          </div>
        ) : (
          <ol className="relative ml-2 border-l border-amber-500/70 pl-5">
            {events.map((event) => {
              const payloadRows = eventPayload(event) ? rows(eventPayload(event)!) : [];
              const meta = eventMeta(event);
              const canResolve = event.type === "approval_required" && event.approval_id === activeApprovalId;
              return (
                <li key={event.sequence} className="relative pb-6 last:pb-0">
                  <span className="absolute -left-[1.8rem] flex size-5 items-center justify-center rounded-full border border-amber-500 bg-background text-amber-700 dark:text-amber-400">
                    <EventIcon type={event.type} />
                  </span>
                  <div className="flex items-start justify-between gap-2">
                    <h3 className="text-xs font-semibold">{eventLabels[event.type]}</h3>
                    <span className="font-mono text-[10px] text-muted-foreground">#{event.sequence}</span>
                  </div>
                  {meta && <p className="mt-1 break-words text-xs text-muted-foreground">{meta}</p>}
                  {payloadRows.length > 0 && (
                    <dl className="mt-3 space-y-2 border-l border-border pl-3">
                      {payloadRows.map(({ label, value }) => (
                        <div key={`${event.sequence}-${label}`}>
                          <dt className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</dt>
                          <dd className="mt-0.5 break-words font-mono text-[11px]">{value}</dd>
                        </div>
                      ))}
                    </dl>
                  )}
                  {canResolve && (
                    <div className="mt-3 grid grid-cols-2 gap-2">
                      <Button
                        type="button"
                        size="sm"
                        disabled={isBusy}
                        onClick={() => void onResolveApproval(event.approval_id, true)}
                      >
                        Phê duyệt
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        disabled={isBusy}
                        onClick={() => void onResolveApproval(event.approval_id, false)}
                      >
                        Từ chối
                      </Button>
                    </div>
                  )}
                </li>
              );
            })}
          </ol>
        )}
      </div>
    </aside>
  );
}
