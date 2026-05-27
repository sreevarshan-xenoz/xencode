import React, { useState, useEffect } from "react";
import Link from "next/link";
import Image from "next/image";
import { useTheme } from "next-themes";

import { HugeiconsIcon } from "@hugeicons/react";
import {
  DocumentCodeIcon,
  TaskDaily01Icon,
  Home01Icon,
  ComputerIcon,
  UserIcon,
  Logout01Icon,
} from "@hugeicons/core-free-icons";
import { usePathname } from "next/navigation";
import { useAuth } from "@/hooks/useAuth";
import { Button } from "@/components/ui/button";

export function Header() {
  const { resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const pathname = usePathname();
  const { user, isAuthenticated, logout } = useAuth();
  const [showUserMenu, setShowUserMenu] = useState(false);

  // After mounting, we can safely show the theme-dependent content
  useEffect(() => {
    setMounted(true);
  }, []);

  // Function to determine if a link is active
  const isActive = (path: string) => {
    if (path === "/") {
      return pathname === "/";
    }
    return pathname?.startsWith(path);
  };

  // Get classes for navigation links based on active state
  const getLinkClasses = (path: string) => {
    const baseClasses =
      "flex items-center gap-1.5 transition-colors px-3 py-1.5 rounded-lg";
    const activeClasses =
      "bg-bytebot-bronze-light-a3 text-bytebot-bronze-light-12";
    const inactiveClasses =
      "text-bytebot-bronze-dark-9 hover:bg-bytebot-bronze-light-a1 hover:text-bytebot-bronze-light-12";

    return `${baseClasses} ${isActive(path) ? activeClasses : inactiveClasses}`;
  };

  return (
    <header className="bg-background flex items-center justify-between border-b p-4">
      <div className="flex items-center gap-6">
        {/* Logo without link */}
        <div>
          {mounted ? (
            <Image
              src={
                resolvedTheme === "dark"
                  ? "/bytebot_transparent_logo_white.svg"
                  : "/bytebot_transparent_logo_dark.svg"
              }
              alt="Bytebot Logo"
              width={100}
              height={30}
              className="h-8 w-auto"
            />
          ) : (
            <div className="h-8 w-[110px]" />
          )}
        </div>
        <div className="border-bytebot-bronze-dark-11 h-5 border border-l-[0.5px]"></div>
        <div className="flex items-center gap-2">
          <Link href="/" className={getLinkClasses("/")}>
            <HugeiconsIcon icon={Home01Icon} className="h-4 w-4" />
            <span className="text-sm">Home</span>
          </Link>
          <Link href="/tasks" className={getLinkClasses("/tasks")}>
            <HugeiconsIcon icon={TaskDaily01Icon} className="h-4 w-4" />
            <span className="text-sm">Tasks</span>
          </Link>
          <Link href="/desktop" className={getLinkClasses("/desktop")}>
            <HugeiconsIcon icon={ComputerIcon} className="h-4 w-4" />
            <span className="text-sm">Desktop</span>
          </Link>
          <Link
            href="https://docs.bytebot.ai/quickstart"
            target="_blank"
            rel="noopener noreferrer"
            className={getLinkClasses("https://docs.bytebot.ai")}
          >
            <HugeiconsIcon icon={DocumentCodeIcon} className="h-4 w-4" />
            <span className="text-sm">Docs</span>
          </Link>
        </div>
      </div>
      <div className="flex items-center gap-3">
        {isAuthenticated && user ? (
          <div className="relative">
            <button
              onClick={() => setShowUserMenu(!showUserMenu)}
              className="flex items-center gap-2 rounded-lg px-3 py-1.5 transition-colors hover:bg-bytebot-bronze-light-a1"
            >
              {user.picture ? (
                <Image
                  src={user.picture}
                  alt={user.name}
                  width={28}
                  height={28}
                  className="rounded-full"
                />
              ) : (
                <div className="flex h-7 w-7 items-center justify-center rounded-full bg-bytebot-bronze-light-a3">
                  <HugeiconsIcon
                    icon={UserIcon}
                    className="h-4 w-4 text-bytebot-bronze-light-12"
                  />
                </div>
              )}
              <span className="text-sm font-medium">{user.name}</span>
            </button>
            {showUserMenu && (
              <div className="absolute right-0 top-full mt-1 w-48 rounded-lg border bg-background p-2 shadow-lg">
                <div className="border-b px-3 py-2">
                  <p className="text-sm font-medium">{user.name}</p>
                  <p className="text-xs text-gray-500">{user.email}</p>
                </div>
                <Button
                  variant="ghost"
                  onClick={logout}
                  className="w-full justify-start text-red-500 hover:text-red-600"
                >
                  <HugeiconsIcon icon={Logout01Icon} className="mr-2 h-4 w-4" />
                  Sign out
                </Button>
              </div>
            )}
          </div>
        ) : (
          <Link href="/onboarding">
            <Button variant="outline" size="sm">
              Sign in
            </Button>
          </Link>
        )}
      </div>
    </header>
  );
}
