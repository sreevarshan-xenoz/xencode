import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  const searchParams = req.nextUrl.searchParams;
  const code = searchParams.get("code");
  const state = searchParams.get("state");
  const error = searchParams.get("error");

  const storedState = req.cookies.get("oauth_state")?.value;

  if (error) {
    return NextResponse.redirect(new URL("/onboarding?error=oauth_error", req.url));
  }

  if (!state || !storedState || state !== storedState) {
    return NextResponse.redirect(new URL("/onboarding?error=invalid_state", req.url));
  }

  if (!code) {
    return NextResponse.redirect(new URL("/onboarding?error=no_code", req.url));
  }

  const GITHUB_CLIENT_ID = process.env.GITHUB_CLIENT_ID;
  const GITHUB_CLIENT_SECRET = process.env.GITHUB_CLIENT_SECRET;
  const REDIRECT_URI = process.env.GITHUB_REDIRECT_URI || "http://localhost:3000/api/auth/callback/github";

  if (!GITHUB_CLIENT_ID || !GITHUB_CLIENT_SECRET) {
    return NextResponse.redirect(new URL("/onboarding?error=oauth_not_configured", req.url));
  }

  try {
    const tokenResponse = await fetch("https://github.com/login/oauth/access_token", {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        client_id: GITHUB_CLIENT_ID,
        client_secret: GITHUB_CLIENT_SECRET,
        code,
      }),
    });

    const tokenData = await tokenResponse.json();

    if (!tokenData.access_token) {
      return NextResponse.redirect(new URL("/onboarding?error=token_exchange_failed", req.url));
    }

    const userResponse = await fetch("https://api.github.com/user", {
      headers: {
        Authorization: `Bearer ${tokenData.access_token}`,
        Accept: "application/vnd.github.v3+json",
      },
    });

    const userData = await userResponse.json();

    const emailsResponse = await fetch("https://api.github.com/user/emails", {
      headers: {
        Authorization: `Bearer ${tokenData.access_token}`,
        Accept: "application/vnd.github.v3+json",
      },
    });

    const emails = await emailsResponse.json();
    const primaryEmail = emails.find((e: { primary: boolean }) => e.primary)?.email || emails[0]?.email;

    const sessionData = {
      id: userData.id.toString(),
      email: primaryEmail || userData.email,
      name: userData.name || userData.login,
      picture: userData.avatar_url,
      username: userData.login,
      provider: "github",
      accessToken: tokenData.access_token,
    };

    const response = NextResponse.redirect(new URL("/onboarding?success=true", req.url));

    response.cookies.set("session", JSON.stringify(sessionData), {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      maxAge: 60 * 60 * 24 * 7,
    });

    response.cookies.delete("oauth_state");

    return response;
  } catch (err) {
    console.error("GitHub OAuth callback error:", err);
    return NextResponse.redirect(new URL("/onboarding?error=callback_failed", req.url));
  }
}