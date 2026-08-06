import { NextRequest, NextResponse } from "next/server";

interface CopilotModel {
  id: string;
  name: string;
  description: string;
  contextWindow: string;
  provider: string;
}

const COPILOT_MODELS: CopilotModel[] = [
  {
    id: "gpt-4o",
    name: "GPT-4o (Copilot)",
    description: "OpenAI's flagship model with advanced coding capabilities",
    contextWindow: "128K",
    provider: "OpenAI",
  },
  {
    id: "claude-3-5-sonnet",
    name: "Claude 3.5 Sonnet (Copilot)",
    description: "Anthropic's model optimized for complex code reasoning",
    contextWindow: "200K",
    provider: "Anthropic",
  },
  {
    id: "cursor-fast",
    name: "Cursor Fast",
    description: "Fast mode for quick completions and suggestions",
    contextWindow: "100K",
    provider: "Cursor",
  },
  {
    id: "cursor-pro",
    name: "Cursor Pro",
    description: "Enhanced mode with larger context and better reasoning",
    contextWindow: "500K",
    provider: "Cursor",
  },
  {
    id: "github-copilot-claude",
    name: "GitHub Copilot (Claude Sonnet)",
    description: "Latest Claude model integrated with GitHub Copilot",
    contextWindow: "200K",
    provider: "GitHub/Anthropic",
  },
];

export async function GET(req: NextRequest) {
  const sessionCookie = req.cookies.get("session");

  if (!sessionCookie?.value) {
    return NextResponse.json(
      { error: "Unauthorized - Please sign in first" },
      { status: 401 }
    );
  }

  try {
    const sessionData = JSON.parse(sessionCookie.value);
    const { provider, accessToken } = sessionData;

    if (provider === "github" && accessToken) {
      try {
        const copilotResponse = await fetch("https://api.github.com/copilot/models", {
          headers: {
            Authorization: `Bearer ${accessToken}`,
            Accept: "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
          },
        });

        if (copilotResponse.ok) {
          const copilotData = await copilotResponse.json();
          
          const models = copilotData.models?.map((model: { id: string; name: string; description: string }) => ({
            ...model,
            provider: "GitHub Copilot",
            contextWindow: "200K",
          })) || [];

          return NextResponse.json({
            models: models.length > 0 ? models : COPILOT_MODELS,
            source: "github_api",
          });
        }
      } catch (copilotError) {
        console.warn("Failed to fetch GitHub Copilot models, using defaults:", copilotError);
      }
    }

    return NextResponse.json({
      models: COPILOT_MODELS,
      source: "default",
    });
  } catch {
    return NextResponse.json(
      { error: "Invalid session" },
      { status: 401 }
    );
  }
}

export async function POST(req: NextRequest) {
  const sessionCookie = req.cookies.get("session");

  if (!sessionCookie?.value) {
    return NextResponse.json(
      { error: "Unauthorized" },
      { status: 401 }
    );
  }

  try {
    const body = await req.json();
    const { modelId } = body;

    const response = NextResponse.json({
      success: true,
      selectedModel: modelId,
    });

    response.cookies.set("selected_copilot_model", modelId, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      maxAge: 60 * 60 * 24 * 30,
    });

    return response;
  } catch {
    return NextResponse.json(
      { error: "Invalid request" },
      { status: 400 }
    );
  }
}