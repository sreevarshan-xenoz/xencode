import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

export function middleware(request: NextRequest) {
  const sessionCookie = request.cookies.get("session");
  const { pathname } = request.nextUrl;

  const publicPaths = [
    "/",
    "/onboarding",
    "/api/auth",
    "/api/auth/google",
    "/api/auth/github",
    "/api/auth/callback",
    "/api/auth/session",
  ];

  const isPublicPath = publicPaths.some((path) => pathname.startsWith(path));

  if (!sessionCookie?.value && !isPublicPath) {
    const loginUrl = new URL("/onboarding", request.url);
    return NextResponse.redirect(loginUrl);
  }

  if (sessionCookie?.value && pathname === "/onboarding") {
    return NextResponse.redirect(new URL("/", request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    "/((?!_next/static|_next/image|favicon.ico|.*\\..*).*)",
  ],
};