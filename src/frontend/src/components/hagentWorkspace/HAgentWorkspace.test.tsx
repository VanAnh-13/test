import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { RuntimeEvent } from "@/lib/hagentRuntime";

import { HAgentWorkspace } from "./HAgentWorkspace";

const runtimeMocks = vi.hoisted(() => ({
  cancelRun: vi.fn(),
  createRuntimeId: vi.fn(),
  replayRun: vi.fn(),
  resolveApproval: vi.fn(),
  startRun: vi.fn(),
}));

vi.mock("next-auth/react", () => ({
  signIn: vi.fn(),
  useSession: () => ({ status: "authenticated" }),
}));

vi.mock("@/lib/hagentRuntime", () => ({
  RuntimeApiError: class RuntimeApiError extends Error {
    readonly code: string;

    constructor(code: string, message: string) {
      super(message);
      this.code = code;
    }
  },
  ...runtimeMocks,
}));

vi.mock("./ConversationRail", () => ({
  ConversationRail: ({
    runId,
    statusLabel,
  }: {
    runId: string | null;
    statusLabel: string;
  }) => (
    <div data-testid="conversation-rail">
      {statusLabel}:{runId ?? "none"}
    </div>
  ),
}));

vi.mock("./ChatCanvas", () => ({
  ChatCanvas: ({
    messages,
    onSubmit,
  }: {
    messages: readonly { readonly content: string }[];
    onSubmit: (message: string) => Promise<void>;
  }) => (
    <div>
      <button type="button" onClick={() => void onSubmit("Audit dữ liệu churn")}>Bắt đầu</button>
      {messages.map((message) => (
        <p key={message.content}>{message.content}</p>
      ))}
    </div>
  ),
}));

vi.mock("./RunLedger", () => ({
  RunLedger: ({ statusLabel }: { statusLabel: string }) => (
    <div data-testid="run-ledger">{statusLabel}</div>
  ),
}));

const startedEvent: RuntimeEvent = {
  type: "run_started",
  run_id: "run-1",
  command_id: "command-1",
  sequence: 1,
  created_at: "2026-08-11T00:00:00Z",
  metadata: {},
};

const completedEvent: RuntimeEvent = {
  type: "run_completed",
  run_id: "run-1",
  command_id: "command-1",
  sequence: 2,
  created_at: "2026-08-11T00:00:01Z",
  result: { summary: "Audit hoàn tất và đã lưu bằng chứng." },
};

afterEach(cleanup);

describe("HAgentWorkspace", () => {
  beforeEach(() => {
    runtimeMocks.createRuntimeId
      .mockReturnValueOnce("run-1")
      .mockReturnValueOnce("command-1");
    runtimeMocks.startRun.mockImplementation(async (_input, options) => {
      options.onEvent?.(startedEvent);
      options.onEvent?.(completedEvent);
      return {
        runId: "run-1",
        lastSequence: 2,
        events: [startedEvent, completedEvent],
      };
    });
  });

  it("phát run qua runtime client và hiển thị terminal event", async () => {
    const user = userEvent.setup();
    render(<HAgentWorkspace />);

    await user.click(screen.getByRole("button", { name: "Bắt đầu" }));

    await waitFor(() => {
      expect(screen.getByTestId("run-ledger").textContent).toBe("Đã hoàn tất");
    });
    expect(runtimeMocks.startRun).toHaveBeenCalledOnce();
    expect(runtimeMocks.startRun.mock.calls[0]?.[0]).toEqual({
      message: "Audit dữ liệu churn",
      run_id: "run-1",
      command_id: "command-1",
      history: [],
    });
    expect(screen.getByText("Audit dữ liệu churn")).toBeTruthy();
    expect(screen.getByText("Audit hoàn tất và đã lưu bằng chứng.")).toBeTruthy();
    expect(screen.getByTestId("conversation-rail").textContent).toBe(
      "Đã hoàn tất:run-1",
    );
  });
});
