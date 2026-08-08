"use server";

import { getSession } from "next-auth/react";
import { cookies } from "next/headers";

const PASSWORD_RESET_COOKIE = "hagent_password_reset";
const MAX_PASSWORD_RESET_SECONDS = 10 * 60;

// Quên mật khẩu -> gọi yêu cầu gửi
export async function forgotPassword(email: string) {
  const session = await getSession();

  try {
    const res = await fetch(
      `${process.env.NEXT_PUBLIC_BASE_API}/forgot-password?email=${email}`,
      {
        method: "POST",
        headers: {
          accept: "application/json",
          Authorization: `Bearer ${session?.user?.access_token}`,
        },
      },
    );

    if (!res.ok) {
      return { ok: false, error: "Email chưa được đăng kí hoặc không tồn tại" };
    }

    const data = await res.json();
    return { ok: true, data };
  } catch (error: any) {
    return { ok: false, error: error.message || "Có lỗi xảy ra" };
  }
}

async function setPasswordResetCookie(resetToken: string, expiresIn: number) {
  if (!resetToken || !Number.isFinite(expiresIn) || expiresIn <= 0) {
    return { ok: false, error: "Phiên đặt lại mật khẩu không hợp lệ" };
  }

  const cookieStore = await cookies();
  cookieStore.set(PASSWORD_RESET_COOKIE, resetToken, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "strict",
    path: "/change-pw",
    maxAge: Math.min(Math.floor(expiresIn), MAX_PASSWORD_RESET_SECONDS),
  });
  return { ok: true };
}

// Token reset chỉ đi qua server action và không được trả về JavaScript phía client.
export async function verifyPasswordResetOtp(email: string, otp: string) {
  if (!email || !/^\d{6}$/.test(otp)) {
    return { ok: false, error: "Email hoặc mã OTP không hợp lệ" };
  }

  try {
    const res = await fetch(
      `${process.env.NEXT_PUBLIC_BASE_API}/auth/verify-otp`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, otp }),
        cache: "no-store",
      },
    );
    if (!res.ok) {
      return { ok: false, error: "Mã OTP không đúng hoặc đã hết hạn" };
    }

    const data: { reset_token?: unknown; expires_in?: unknown } = await res.json();
    if (
      typeof data.reset_token !== "string" ||
      typeof data.expires_in !== "number"
    ) {
      return { ok: false, error: "Phản hồi đặt lại mật khẩu không hợp lệ" };
    }
    return setPasswordResetCookie(data.reset_token, data.expires_in);
  } catch {
    return {
      ok: false,
      error: "Không thể kết nối dịch vụ xác thực OTP. Vui lòng thử lại.",
    };
  }
}

// Token reset chỉ được đọc ở server và bị xóa ngay sau khi dùng thành công.
export async function changePassword(
  new1_password: string,
  new2_password: string,
) {
  const cookieStore = await cookies();
  const resetToken = cookieStore.get(PASSWORD_RESET_COOKIE)?.value;
  if (!resetToken) {
    return {
      ok: false,
      error: "Phiên đặt lại mật khẩu đã hết hạn. Vui lòng yêu cầu mã OTP mới.",
    };
  }

  try {
    const res = await fetch(
      `${process.env.NEXT_PUBLIC_BASE_API}/reset-password`,
      {
        method: "POST",
        headers: {
          accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          reset_token: resetToken,
          new_password: new1_password,
          confirm_password: new2_password,
        }),
      },
    );
    const data = await res.json();

    if (!res.ok) {
      return { ok: false, error: data.detail || "Có lỗi xảy ra" };
    }

    cookieStore.set(PASSWORD_RESET_COOKIE, "", {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "strict",
      path: "/change-pw",
      maxAge: 0,
    });
    return { ok: true, noError: data.message };
  } catch {
    return {
      ok: false,
      error: "Không thể kết nối dịch vụ đặt lại mật khẩu. Vui lòng thử lại.",
    };
  }
}
