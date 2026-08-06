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

  const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID;
  const GOOGLE_CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET;
  const REDIRECT_URI = process.env.GOOGLE_REDIRECT_URI || "http://localhost:3000/api/auth/callback/google";

  if (!GOOGLE_CLIENT_ID || !GOOGLE_CLIENT_SECRET) {
    return NextResponse.redirect(new URL("/onboarding?error=oauth_not_configured", req.url));
  }

  try {
    const tokenResponse = await fetch("https://oauth2.googleapis.com/token", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: new URLSearchParams({
        client_id: GOOGLE_CLIENT_ID,
        client_secret: GOOGLE_CLIENT_SECRET,
        code,
        grant_type: "authorization_code",
        redirect_uri: REDIRECT_URI,
      }),
    });

    const tokenData = await tokenResponse.json();

    if (!tokenData.access_token) {
      return NextResponse.redirect(new URL("/onboarding?error=token_exchange_failed", req.url));
    }

    const userResponse = await fetch("https://www.googleapis.com/oauth2/v2/userinfo", {
      headers: {
        Authorization: `Bearer ${tokenData.access_token}`,
      },
    });

    const userData = await userResponse.json();

    const sessionData = {
      id: userData.id,
      email: userData.email,
      name: userData.name,
      picture: userData.picture,
      provider: "google",
      accessToken: tokenData.access_token,
      refreshToken: tokenData.refresh_token,
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
    console.error("Google OAuth callback error:", err);
    return NextResponse.redirect(new URL("/onboarding?error=callback_failed", req.url));
  }
}