"use client";

import { AuthProvider as ReactAuthProvider } from "@/hooks/useAuth";
import { ReactNode } from "react";

export function AuthProvider({ children }: { children: ReactNode }) {
  return <ReactAuthProvider>{children}</ReactAuthProvider>;
}