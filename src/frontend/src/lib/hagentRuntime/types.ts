export type JsonObject = Record<string, unknown>;

export type RuntimeEventType =
  | "run_started"
  | "plan_proposed"
  | "artifact_produced"
  | "check_completed"
  | "approval_required"
  | "action_completed"
  | "evidence_added"
  | "run_completed"
  | "run_failed"
  | "run_cancelled";

interface RuntimeEventBase<TType extends RuntimeEventType> {
  readonly type: TType;
  readonly run_id: string;
  readonly command_id: string;
  readonly sequence: number;
  readonly created_at: string;
}

export interface RunStartedEvent extends RuntimeEventBase<"run_started"> {
  readonly metadata: JsonObject;
}

export interface PlanProposedEvent extends RuntimeEventBase<"plan_proposed"> {
  readonly plan: JsonObject;
}

export interface ArtifactProducedEvent
  extends RuntimeEventBase<"artifact_produced"> {
  readonly artifact_type: string;
  readonly artifact: JsonObject;
}

export interface CheckCompletedEvent
  extends RuntimeEventBase<"check_completed"> {
  readonly checker: string;
  readonly verdict: string;
  readonly details: JsonObject;
}

export interface ApprovalRequiredEvent
  extends RuntimeEventBase<"approval_required"> {
  readonly approval_id: string;
  readonly proposal: JsonObject;
}

export interface ActionCompletedEvent
  extends RuntimeEventBase<"action_completed"> {
  readonly action: string;
  readonly outcome: string;
  readonly details: JsonObject;
}

export interface EvidenceAddedEvent
  extends RuntimeEventBase<"evidence_added"> {
  readonly evidence_type: string;
  readonly evidence: JsonObject;
}

export interface RunCompletedEvent extends RuntimeEventBase<"run_completed"> {
  readonly result: JsonObject;
}

export interface RunFailedEvent extends RuntimeEventBase<"run_failed"> {
  readonly error_code: string;
  readonly message: string;
}

export interface RunCancelledEvent extends RuntimeEventBase<"run_cancelled"> {
  readonly reason: string;
}

export type RuntimeEvent =
  | RunStartedEvent
  | PlanProposedEvent
  | ArtifactProducedEvent
  | CheckCompletedEvent
  | ApprovalRequiredEvent
  | ActionCompletedEvent
  | EvidenceAddedEvent
  | RunCompletedEvent
  | RunFailedEvent
  | RunCancelledEvent;

export interface RunHistoryMessage {
  readonly role: "user" | "assistant";
  readonly content: string;
}

export interface StartRunInput {
  readonly message: string;
  readonly run_id?: string;
  readonly command_id?: string;
  readonly history?: readonly RunHistoryMessage[];
  readonly model?: string | null;
}

export interface ResolveApprovalInput {
  readonly approved: boolean;
  readonly command_id?: string;
  readonly response?: JsonObject;
}

export interface CancelRunInput {
  readonly command_id?: string;
}

export interface RuntimeRequestOptions {
  readonly token?: string;
  readonly signal?: AbortSignal;
  readonly afterSequence?: number;
  readonly onEvent?: (event: RuntimeEvent) => void;
}

export interface RuntimeStreamResult {
  readonly runId: string;
  readonly lastSequence: number;
  readonly events: readonly RuntimeEvent[];
}
