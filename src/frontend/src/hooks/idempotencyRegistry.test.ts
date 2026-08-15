import assert from "node:assert/strict";
import test from "node:test";

const registryModule = await import(
  new URL("./idempotencyRegistry.ts", import.meta.url).href
);
const { IdempotencyRegistry, shouldReleaseIdempotencyKey } = registryModule;

test("retry lỗi giữ key và action khác không ghi đè", () => {
  const generatedKeys = ["action-a", "action-b"];
  const registry = new IdempotencyRegistry(() => generatedKeys.shift()!);

  const actionA = registry.getOrCreate("fingerprint-a");
  const actionB = registry.getOrCreate("fingerprint-b");

  assert.equal(registry.getOrCreate("fingerprint-a"), actionA);
  assert.notEqual(actionA, actionB);
});

test("success xoay key cho action chủ ý tiếp theo", () => {
  const generatedKeys = ["first-action", "next-action"];
  const registry = new IdempotencyRegistry(() => generatedKeys.shift()!);
  const firstKey = registry.getOrCreate("same-request");

  registry.release("same-request", firstKey);

  assert.equal(registry.getOrCreate("same-request"), "next-action");
});

test("needs_reconciliation không release key", () => {
  assert.equal(
    shouldReleaseIdempotencyKey({ status: "needs_reconciliation" }),
    false,
  );
  assert.equal(shouldReleaseIdempotencyKey({ status: "success" }), true);
  assert.equal(shouldReleaseIdempotencyKey({}), false);
});

test("key mặc định không chứa fingerprint hoặc payload", () => {
  const registry = new IdempotencyRegistry();
  const sensitiveFingerprint = "training:/dataset/private-token";

  const key = registry.getOrCreate(sensitiveFingerprint);

  assert.match(
    key,
    /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/,
  );
  assert.equal(key.includes(sensitiveFingerprint), false);
  assert.equal(key.includes("private-token"), false);
});
