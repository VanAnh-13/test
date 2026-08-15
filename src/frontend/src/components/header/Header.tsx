"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { signIn, signOut, useSession } from "next-auth/react";
import { LogOut, Menu, User2, X } from "lucide-react";
import { FaGithub } from "react-icons/fa";

import ModeToggle from "@/components/mode-toggle";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useApi } from "@/hooks/useApi";

const navigationItems = [
  { label: "TRANG CHỦ", href: "#home" },
  { label: "GIỚI THIỆU", href: "#introduction" },
  { label: "VỀ CHÚNG TÔI", href: "#about-us" },
  { label: "LIÊN HỆ", href: "#contact" },
] as const;

export default function Header() {
  const { get } = useApi();
  const { data: session } = useSession();
  const [avatarUrl, setAvatarUrl] = useState<string | null>(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    let active = true;
    let objectUrl: string | null = null;

    const fetchAvatar = async () => {
      if (!session?.user?.username) return;
      try {
        const blob = await get(
          `${process.env.NEXT_PUBLIC_BASE_API}/get_avatar/${session.user.username}`,
          { responseType: "blob" },
        );
        if (!active) return;
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        objectUrl = blob?.size ? URL.createObjectURL(blob) : null;
        setAvatarUrl(objectUrl ?? "");
      } catch (error) {
        if (active) setAvatarUrl("");
        console.warn("Không thể tải avatar", {
          errorType: error instanceof Error ? error.name : "UnknownError",
        });
      }
    };

    void fetchAvatar();
    const handleAvatarUpdate = () => void fetchAvatar();
    window.addEventListener("avatar-updated", handleAvatarUpdate);
    return () => {
      active = false;
      window.removeEventListener("avatar-updated", handleAvatarUpdate);
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [get, session?.user?.username]);

  return (
    <header className="relative z-50 flex h-16 w-full items-center justify-between border-b bg-background px-4 md:px-6">
      <Link href="/" className="flex shrink-0 items-center gap-2" prefetch={false}>
        <Image src="/image.png" priority width={120} height={120} alt="HAutoML" />
      </Link>

      <nav className="hidden items-center gap-6 lg:flex xl:gap-10" aria-label="Điều hướng chính">
        {!session &&
          navigationItems.map((item) => (
            <Link
              key={item.label}
              href={{ pathname: "/", hash: item.href.slice(1) }}
              className="py-3 text-center text-sm font-medium text-foreground transition-colors hover:text-blue-600 motion-reduce:transition-none"
            >
              {item.label}
            </Link>
          ))}
      </nav>

      <div className="flex shrink-0 items-center gap-2 sm:gap-3">
        <Link
          href="https://github.com/optivisionlab/AutoML"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Mở mã nguồn HAutoML trên GitHub"
          className="text-gray-700 transition-colors hover:text-black focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring dark:text-gray-300 dark:hover:text-white motion-reduce:transition-none"
        >
          <FaGithub className="size-7 sm:size-8" aria-hidden="true" />
        </Link>
        <ModeToggle />

        {session ? (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button type="button" aria-label="Mở menu tài khoản">
                <Avatar className="cursor-pointer">
                  <AvatarImage src={avatarUrl ?? undefined} alt="Ảnh đại diện" />
                  <AvatarFallback className="bg-gray-100">
                    <User2 className="size-6" aria-hidden="true" />
                  </AvatarFallback>
                </Avatar>
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>{session.user?.username}</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => void signOut()}>
                Đăng xuất <LogOut className="ml-2 size-4 text-red-600" aria-hidden="true" />
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        ) : (
          <div className="hidden items-center gap-3 lg:flex">
            <Button type="button" onClick={() => void signIn()}>
              ĐĂNG NHẬP
            </Button>
            <Button asChild>
              <Link href="/register">ĐĂNG KÝ</Link>
            </Button>
          </div>
        )}

        {!session && (
          <button
            type="button"
            className="ml-1 rounded-md p-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring lg:hidden"
            aria-label={mobileMenuOpen ? "Đóng menu" : "Mở menu"}
            aria-expanded={mobileMenuOpen}
            aria-controls="mobile-navigation"
            onClick={() => setMobileMenuOpen((current) => !current)}
          >
            {mobileMenuOpen ? <X aria-hidden="true" /> : <Menu aria-hidden="true" />}
          </button>
        )}
      </div>

      {mobileMenuOpen && !session && (
        <nav
          id="mobile-navigation"
          aria-label="Điều hướng mobile"
          className="absolute left-0 top-16 z-40 flex w-full flex-col gap-2 border-t bg-background px-4 py-3 shadow-md lg:hidden"
        >
          {navigationItems.map((item) => (
            <Link
              key={item.label}
              href={item.href}
              className="w-full border-b border-border py-3 text-center text-sm font-medium last:border-0 hover:bg-muted hover:text-blue-600"
              onClick={() => setMobileMenuOpen(false)}
            >
              {item.label}
            </Link>
          ))}
          <Button type="button" className="w-full" onClick={() => void signIn()}>
            Đăng nhập
          </Button>
          <Button asChild className="w-full">
            <Link href="/register" onClick={() => setMobileMenuOpen(false)}>
              Đăng ký
            </Link>
          </Button>
        </nav>
      )}
    </header>
  );
}
