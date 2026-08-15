"use client";

import { Fragment, useEffect, useState } from "react";
import { NavItems } from "@/config";
import { cn } from "@/lib/utils";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useSession } from "next-auth/react";
import { SideNavItem } from "./SideNavItem";

export default function SideNav() {
  const [isSidebarExpanded, setIsSidebarExpanded] = useState(true);
  const { data: session, status } = useSession();

  useEffect(() => {
    if (typeof window !== "undefined") {
      const saved = window.localStorage.getItem("sidebarExpanded");
      if (saved !== null) {
        setIsSidebarExpanded(JSON.parse(saved));
      }
    }
  }, []);

  // Chờ xác định phiên trước khi dựng menu.
  if (status === "loading") return null;

  // Không hiển thị menu khi người dùng chưa đăng nhập.
  if (!session) return null;

  const toggleSidebar = () => {
    setIsSidebarExpanded((expanded) => !expanded);
  };

  const navItems = NavItems(session?.user.role || "");

  return (
    <div suppressHydrationWarning>
      <div
        className={cn(
          isSidebarExpanded ? "w-[250px]" : "w-[68px]",
          "border-r transition-all duration-300 ease-in-out transform hidden sm:flex h-full "
        )}
        suppressHydrationWarning
      >
        <aside className="flex h-full flex-col w-full break-words px-4 overflow-x-hidden columns-1">
          {/* Nhóm điều hướng phía trên */}
          <div className="mt-4 relative pb-2">
            <div className="flex flex-col space-y-1">
              {navItems.map((item, idx) => {
                if (item.position === "top") {
                  return (
                    <Fragment key={idx}>
                      <div className="space-y-1">
                        <SideNavItem
                          label={item.name}
                          icon={item.icon}
                          path={item.href}
                          active={item.active}
                          isSidebarExpanded={isSidebarExpanded}
                        />
                      </div>
                    </Fragment>
                  );
                }
              })}
            </div>
          </div>
          {/* Nhóm điều hướng phía dưới */}
          <div className="sticky bottom-0 mt-auto whitespace-nowrap mb-4 transition duration-200 block">
            {navItems.map((item, idx) => {
              if (item.position === "bottom") {
                return (
                  <Fragment key={idx}>
                    <div className="space-y-1">
                      <SideNavItem
                        label={item.name}
                        icon={item.icon}
                        path={item.href}
                        active={item.active}
                        isSidebarExpanded={isSidebarExpanded}
                      />
                    </div>
                  </Fragment>
                );
              }
            })}
          </div>
        </aside>
        <div className="mt-[calc(calc(90vh)-40px)] relative">
          <button
            type="button"
            className="absolute bottom-32 right-[-12px] flex h-6 w-6 items-center justify-center border border-muted-foreground/20 rounded-full bg-accent shadow-md hover:shadow-lg transition-shadow duration-300 ease-in-out"
            onClick={toggleSidebar}
          >
            {isSidebarExpanded ? (
              <ChevronLeft size={16} className="stroke-foreground" />
            ) : (
              <ChevronRight size={16} className="stroke-foreground" />
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
