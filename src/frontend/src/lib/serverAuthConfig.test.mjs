import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  resolveAuthApiUrl,
  resolvePasswordResetCookieSecure,
} from "./serverAuthConfig.ts";

test("tạo URL nội bộ tuyệt đối và encode query bằng URLSearchParams", () => {
  const url = resolveAuthApiUrl(
    "/forgot-password",
    { email: "a+b&c@example.com" },
    { AUTH_API_BASE_URL: "http://toolkit:8585" },
  );

  assert.equal(
    url.toString(),
    "http://toolkit:8585/forgot-password?email=a%2Bb%26c%40example.com",
  );
});

test("reject cấu hình URL thiếu, relative hoặc có authority/path phụ", () => {
  const invalidValues = [
    undefined,
    "",
    "/api/backend",
    "ftp://toolkit:8585",
    "http://user:password@toolkit:8585",
    "http://toolkit:8585/base",
    "http://toolkit:8585/?debug=true",
    "http://toolkit:8585/#fragment",
  ];

  for (const AUTH_API_BASE_URL of invalidValues) {
    assert.throws(
      () => resolveAuthApiUrl("/reset-password", undefined, { AUTH_API_BASE_URL }),
      /AUTH_API_BASE_URL/,
    );
  }
});

test("reject pathname không cố định ở boundary server", () => {
  const environment = { AUTH_API_BASE_URL: "http://toolkit:8585" };
  assert.throws(
    () => resolveAuthApiUrl("reset-password", undefined, environment),
    /pathname/,
  );
  assert.throws(
    () => resolveAuthApiUrl("//attacker.invalid/path", undefined, environment),
    /pathname/,
  );
});

test("production bắt buộc khai báo rõ chính sách secure cookie", () => {
  assert.equal(
    resolvePasswordResetCookieSecure({
      NODE_ENV: "production",
      SESSION_HTTPS_ONLY: "true",
    }),
    true,
  );
  assert.equal(
    resolvePasswordResetCookieSecure({
      NODE_ENV: "production",
      SESSION_HTTPS_ONLY: "false",
    }),
    false,
  );
  assert.throws(
    () => resolvePasswordResetCookieSecure({ NODE_ENV: "production" }),
    /SESSION_HTTPS_ONLY/,
  );
  assert.throws(
    () =>
      resolvePasswordResetCookieSecure({
        NODE_ENV: "production",
        SESSION_HTTPS_ONLY: "yes",
      }),
    /SESSION_HTTPS_ONLY/,
  );
});

test("development thiếu SESSION_HTTPS_ONLY dùng cookie HTTP cục bộ", () => {
  assert.equal(resolvePasswordResetCookieSecure({ NODE_ENV: "development" }), false);
  assert.equal(resolvePasswordResetCookieSecure({ NODE_ENV: "test" }), false);
});

test("server action không dùng public API URL hoặc Bearer undefined", async () => {
  const source = await readFile(
    new URL("../app/serverActions/auth.ts", import.meta.url),
    "utf8",
  );

  assert.doesNotMatch(source, /NEXT_PUBLIC_BASE_API/);
  assert.doesNotMatch(source, /getSession/);
  assert.doesNotMatch(source, /Bearer/);
  assert.match(source, /httpOnly:\s*true/);
  assert.match(source, /sameSite:\s*"strict"/);
});
