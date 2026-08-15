"use client";

import { signIn } from "next-auth/react";
import { Suspense, useEffect } from "react";
import { useSearchParams, useRouter } from "next/navigation";

function GoogleCallbackContent() {
  const searchParams = useSearchParams();
  const router = useRouter();

  const authorizationCode = searchParams?.get("code");

  useEffect(() => {
    if (!authorizationCode) {
      router.replace("/login");
      return;
    }

    window.history.replaceState(null, "", window.location.pathname);
    void signIn("credentials", {
      authorization_code: authorizationCode,
      redirect: false,
    }).then((result) => {
      router.replace(result?.ok ? "/" : "/login");
    });
  }, [authorizationCode, router]);

  return <p>Đang đăng nhập bằng Google...</p>;
}

export default function GoogleCallbackPage() {
  return (
    <Suspense fallback={<p role="status">Đang xử lý đăng nhập Google...</p>}>
      <GoogleCallbackContent />
    </Suspense>
  );
}
