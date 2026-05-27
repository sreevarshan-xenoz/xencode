import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  const sessionCookie = req.cookies.get("session");

  if (!sessionCookie?.value) {
    return NextResponse.json({ authenticated: false, user: null });
  }

  try {
    const sessionData = JSON.parse(sessionCookie.value);
    
    return NextResponse.json({
      authenticated: true,
      user: {
        id: sessionData.id,
        email: sessionData.email,
        name: sessionData.name,
        picture: sessionData.picture,
        provider: sessionData.provider,
        username: sessionData.username,
      },
    });
  } catch {
    return NextResponse.json({ authenticated: false, user: null });
  }
}

export async function DELETE(req: NextRequest) {
  const response = NextResponse.json({ success: true });
  
  response.cookies.delete("session");
  response.cookies.delete("oauth_state");
  
  return response;
}