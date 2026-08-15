import assert from "node:assert/strict";
import test from "node:test";

import {
  RuntimeSseParser,
  RuntimeStreamProtocolError,
} from "./sse.ts";

const baseEvent = (overrides = {}) => ({
  type: "run_started",
  run_id: "run001",
  command_id: "command001",
  sequence: 1,
  created_at: "2026-08-09T00:00:00Z",
  metadata: {},
  ...overrides,
});

const frame = (event, newline = "\n") =>
  [
    `id: ${event.sequence}`,
    `event: ${event.type}`,
    `data: ${JSON.stringify(event)}`,
    "",
    "",
  ].join(newline);

test("ghép chunk và giữ sequence tăng đơn điệu", () => {
  const parser = new RuntimeSseParser({ expectedRunId: "run001" });
  const first = frame(baseEvent(), "\r\n");
  const secondEvent = baseEvent({
    type: "check_completed",
    sequence: 2,
    checker: "contract",
    verdict: "pass",
    details: {},
    metadata: undefined,
  });
  const second = frame(secondEvent);

  assert.deepEqual(parser.feed(first.slice(0, 23)), []);
  const events = parser.feed(first.slice(23) + second);
  parser.finish();

  assert.equal(events.length, 2);
  assert.equal(events[0].type, "run_started");
  assert.equal(events[1].type, "check_completed");
  assert.equal(parser.sequence, 2);
});

test("từ chối event name không khớp payload", () => {
  const parser = new RuntimeSseParser({ expectedRunId: "run001" });
  const invalid = frame(baseEvent()).replace(
    "event: run_started",
    "event: run_completed",
  );

  assert.throws(
    () => parser.feed(invalid),
    (error) =>
      error instanceof RuntimeStreamProtocolError &&
      error.code === "EVENT_MISMATCH",
  );
});

test("từ chối sequence phát trùng sau reconnect", () => {
  const parser = new RuntimeSseParser({
    afterSequence: 1,
    expectedRunId: "run001",
  });

  assert.throws(
    () => parser.feed(frame(baseEvent())),
    (error) =>
      error instanceof RuntimeStreamProtocolError &&
      error.code === "NON_MONOTONIC_SEQUENCE",
  );
});

test("từ chối event thuộc run khác", () => {
  const parser = new RuntimeSseParser({ expectedRunId: "run001" });

  assert.throws(
    () => parser.feed(frame(baseEvent({ run_id: "run002" }))),
    (error) =>
      error instanceof RuntimeStreamProtocolError &&
      error.code === "RUN_ID_MISMATCH",
  );
});

test("từ chối stream kết thúc giữa frame", () => {
  const parser = new RuntimeSseParser();
  parser.feed("id: 1\nevent: run_started\n");

  assert.throws(
    () => parser.finish(),
    (error) =>
      error instanceof RuntimeStreamProtocolError &&
      error.code === "INCOMPLETE_FRAME",
  );
});

test("parse đủ mười runtime event variants", () => {
  const variants = [
    { type: "run_started", metadata: {} },
    { type: "plan_proposed", plan: {} },
    { type: "artifact_produced", artifact_type: "DatasetAudit", artifact: {} },
    { type: "check_completed", checker: "policy", verdict: "pass", details: {} },
    { type: "approval_required", approval_id: "approval001", proposal: {} },
    { type: "action_completed", action: "train", outcome: "submitted", details: {} },
    { type: "evidence_added", evidence_type: "dataset", evidence: {} },
    { type: "run_completed", result: {} },
    { type: "run_failed", error_code: "UPSTREAM_FAILED", message: "failed" },
    { type: "run_cancelled", reason: "user_requested" },
  ];
  const parser = new RuntimeSseParser({ expectedRunId: "run001" });
  const events = variants.flatMap((variant, index) =>
    parser.feed(
      frame({
        ...baseEvent(),
        ...variant,
        sequence: index + 1,
      }),
    ),
  );
  parser.finish();

  assert.deepEqual(
    events.map((event) => event.type),
    variants.map((event) => event.type),
  );
});

test("chặn frame vượt giới hạn buffer", () => {
  const parser = new RuntimeSseParser();

  assert.throws(
    () => parser.feed("x".repeat(1_048_577)),
    (error) =>
      error instanceof RuntimeStreamProtocolError &&
      error.code === "FRAME_TOO_LARGE",
  );
});
