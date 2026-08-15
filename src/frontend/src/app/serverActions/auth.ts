"use server";

import { cookies } from "next/headers";

import {
  resolveAuthApiUrl,
  resolvePasswordResetCookieSecure,
} from "@/lib/serverAuthConfig";

const PASSWORD_RESET_COOKIE = "hagent_password_reset";
const PASSWORD_RESET_COOKIE_PATH = "/change-pw";
const MAX_PASSWORD_RESET_SECONDS = 10 * 60;
const MAX_RESET_TOKEN_LENGTH = 4096;
const MAX_PUBLIC_MESSAGE_LENGTH = 240;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function publicMessage(payload: unknown, field: string): string | null {
  if (!isRecord(payload)) return null;
  const value = payload[field];
  if (typeof value !== "string" || !value.trim()) return null;
  return value.trim().slice(0, MAX_PUBLIC_MESSAGE_LENGTH);
}

function configurationUnavailable() {
  return {
    ok: false as const,
    error: "Dịch vụ xác thực chưa được cấu hình. Vui lòng thử lại sau.",
  };
}

function passwordResetCookieOptions(secure: boolean, maxAge: number) {
  return {
    httpOnly: true,
    secure,
    sameSite: "strict" as const,
    path: PASSWORD_RESET_COOKIE_PATH,
    maxAge,
  };
}

export async function forgotPassword(email: string) {
  const normalizedEmail = email.trim();
  if (!normalizedEmail || normalizedEmail.length > 320) {
    return { ok: false as const, error: "Email không hợp lệ" };
  }

  let url: URL;
  try {
    url = resolveAuthApiUrl("/forgot-password", { email: normalizedEmail });
  } catch {
    return configurationUnavailable();
  }

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    if (!response.ok) {
      return {
        ok: false as const,
        error: "Email chưa được đăng ký hoặc không tồn tại",
      };
    }
    return { ok: true as const };
  } catch {
    return {
      ok: false as const,
      error: "Không thể kết nối dịch vụ quên mật khẩu. Vui lòng thử lại.",
    };
  }
}

async function setPasswordResetCookie(
  resetToken: string,
  expiresIn: number,
  secure: boolean,
) {
  if (
    !resetToken ||
    resetToken.length > MAX_RESET_TOKEN_LENGTH ||
    !Number.isFinite(expiresIn) ||
    expiresIn <= 0
  ) {
    return { ok: false as const, error: "Phiên đặt lại mật khẩu không hợp lệ" };
  }

  const cookieStore = await cookies();
  cookieStore.set(
    PASSWORD_RESET_COOKIE,
    resetToken,
    passwordResetCookieOptions(
      secure,
      Math.min(Math.floor(expiresIn), MAX_PASSWORD_RESET_SECONDS),
    ),
  );
  return { ok: true as const };
}

export async function verifyPasswordResetOtp(email: string, otp: string) {
  const normalizedEmail = email.trim();
  if (!normalizedEmail || normalizedEmail.length > 320 || !/^\d{6}$/.test(otp)) {
    return { ok: false as const, error: "Email hoặc mã OTP không hợp lệ" };
  }

  let url: URL;
  let secure: boolean;
  try {
    url = resolveAuthApiUrl("/auth/verify-otp");
    secure = resolvePasswordResetCookieSecure();
  } catch {
    return configurationUnavailable();
  }

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: normalizedEmail, otp }),
      cache: "no-store",
    });
    if (!response.ok) {
      return { ok: false as const, error: "Mã OTP không đúng hoặc đã hết hạn" };
    }

    const payload = await readJson(response);
    if (!isRecord(payload)) {
      return { ok: false as const, error: "Phản hồi đặt lại mật khẩu không hợp lệ" };
    }
    const resetToken = payload.reset_token;
    const expiresIn = payload.expires_in;
    if (typeof resetToken !== "string" || typeof expiresIn !== "number") {
      return { ok: false as const, error: "Phản hồi đặt lại mật khẩu không hợp lệ" };
    }
    return setPasswordResetCookie(resetToken, expiresIn, secure);
  } catch {
    return {
      ok: false as const,
      error: "Không thể kết nối dịch vụ xác thực OTP. Vui lòng thử lại.",
    };
  }
}

export async function changePassword(
  new1_password: string,
  new2_password: string,
) {
  if (
    new1_password.length < 8 ||
    new1_password.length > 128 ||
    new1_password !== new2_password
  ) {
    return { ok: false as const, error: "Mật khẩu mới không hợp lệ" };
  }

  const cookieStore = await cookies();
  const resetToken = cookieStore.get(PASSWORD_RESET_COOKIE)?.value;
  if (!resetToken || resetToken.length > MAX_RESET_TOKEN_LENGTH) {
    return {
      ok: false as const,
      error: "Phiên đặt lại mật khẩu đã hết hạn. Vui lòng yêu cầu mã OTP mới.",
    };
  }

  let url: URL;
  let secure: boolean;
  try {
    url = resolveAuthApiUrl("/reset-password");
    secure = resolvePasswordResetCookieSecure();
  } catch {
    return configurationUnavailable();
  }

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        reset_token: resetToken,
        new_password: new1_password,
        confirm_password: new2_password,
      }),
      cache: "no-store",
    });
    const payload = await readJson(response);
    if (!response.ok) {
      return {
        ok: false as const,
        error: publicMessage(payload, "detail") ?? "Không thể đặt lại mật khẩu",
      };
    }

    cookieStore.set(
      PASSWORD_RESET_COOKIE,
      "",
      passwordResetCookieOptions(secure, 0),
    );
    return {
      ok: true as const,
      noError:
        publicMessage(payload, "message") ?? "Thay đổi mật khẩu thành công",
    };
  } catch {
    return {
      ok: false as const,
      error: "Không thể kết nối dịch vụ đặt lại mật khẩu. Vui lòng thử lại.",
    };
  }
}
