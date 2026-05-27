"use client";

import React, { useState } from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  GoogleIcon,
  GithubIcon,
  ArrowRight01Icon,
  SparkIcon,
  CheckmarkCircle01Icon,
} from "@hugeicons/core-free-icons";
import { Card, CardContent } from "@/components/ui/card";

interface CopilotModel {
  id: string;
  name: string;
  description: string;
  isActive: boolean;
}

export default function OnboardingPage() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState<string | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [copilotModels, setCopilotModels] = useState<CopilotModel[]>([
    {
      id: "gpt-4o",
      name: "GitHub Copilot (Claude Sonnet 4.5)",
      description: "Most capable model for coding tasks",
      isActive: true,
    },
    {
      id: "claude-3-5-sonnet",
      name: "GitHub Copilot (Claude 3.5 Sonnet)",
      description: "Balanced speed and quality",
      isActive: false,
    },
    {
      id: "cursor-fast",
      name: "GitHub Copilot (Fast Mode)",
      description: "Quick completions and suggestions",
      isActive: false,
    },
  ]);

  const handleOAuthLogin = async (provider: "google" | "github") => {
    setIsLoading(provider);
    setSelectedProvider(provider);

    try {
      const response = await fetch(`/api/auth/${provider}`, {
        method: "POST",
      });

      const data = await response.json();

      if (data.url) {
        window.location.href = data.url;
      } else {
        console.error("No OAuth URL received");
        setIsLoading(null);
        setSelectedProvider(null);
      }
    } catch (error) {
      console.error("OAuth login failed:", error);
      setIsLoading(null);
      setSelectedProvider(null);
    }
  };

  const handleModelSelect = async (modelId: string) => {
    const updatedModels = copilotModels.map((m) => ({
      ...m,
      isActive: m.id === modelId,
    }));
    setCopilotModels(updatedModels);
  };

  const handleContinue = () => {
    const selectedModel = copilotModels.find((m) => m.isActive);
    localStorage.setItem("selectedCopilotModel", selectedModel?.id || "");
    router.push("/");
  };

  const handleSkip = () => {
    router.push("/");
  };

  if (isAuthenticated) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
        <div className="flex min-h-screen items-center justify-center p-4">
          <Card className="w-full max-w-lg">
            <CardContent className="pt-8">
              <div className="text-center">
                <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-green-100">
                  <HugeiconsIcon
                    icon={CheckmarkCircle01Icon}
                    className="h-8 w-8 text-green-600"
                  />
                </div>
                <h2 className="mb-2 text-2xl font-bold">
                  Choose Your Copilot Model
                </h2>
                <p className="mb-6 text-gray-600 dark:text-gray-400">
                  Select the AI model you want to use for coding assistance
                </p>
              </div>

              <div className="space-y-3">
                {copilotModels.map((model) => (
                  <button
                    key={model.id}
                    onClick={() => handleModelSelect(model.id)}
                    className={`w-full rounded-lg border-2 p-4 text-left transition-all ${
                      model.isActive
                        ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                        : "border-gray-200 hover:border-gray-300 dark:border-gray-700"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="font-semibold">{model.name}</h3>
                        <p className="text-sm text-gray-500">{model.description}</p>
                      </div>
                      {model.isActive && (
                        <HugeiconsIcon
                          icon={CheckmarkCircle01Icon}
                          className="h-5 w-5 text-blue-500"
                        />
                      )}
                    </div>
                  </button>
                ))}
              </div>

              <div className="mt-6 flex gap-3">
                <Button variant="outline" onClick={handleSkip} className="flex-1">
                  Skip for now
                </Button>
                <Button onClick={handleContinue} className="flex-1">
                  Continue
                  <HugeiconsIcon icon={ArrowRight01Icon} className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <div className="flex min-h-screen items-center justify-center p-4">
        <div className="w-full max-w-md">
          <div className="mb-8 text-center">
            <div className="mb-4 flex justify-center">
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-600 to-purple-600">
                <HugeiconsIcon icon={SparkIcon} className="h-8 w-8 text-white" />
              </div>
            </div>
            <h1 className="mb-2 text-3xl font-bold">Welcome to Xencode</h1>
            <p className="text-gray-600 dark:text-gray-400">
              Sign in to access GitHub Copilot models and supercharge your
              development workflow
            </p>
          </div>

          <Card>
            <CardContent className="pt-6">
              <div className="space-y-4">
                <Button
                  onClick={() => handleOAuthLogin("google")}
                  disabled={isLoading !== null}
                  className="flex w-full items-center justify-center gap-3 bg-white py-6 text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:text-gray-200 dark:hover:bg-gray-700"
                  variant="outline"
                >
                  {isLoading === "google" ? (
                    <div className="h-5 w-5 animate-spin rounded-full border-2 border-gray-300 border-t-gray-600" />
                  ) : (
                    <svg
                      className="h-5 w-5"
                      viewBox="0 0 24 24"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path
                        fill="#4285F4"
                        d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                      />
                      <path
                        fill="#34A853"
                        d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                      />
                      <path
                        fill="#FBBC05"
                        d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                      />
                      <path
                        fill="#EA4335"
                        d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                      />
                    </svg>
                  )}
                  Continue with Google
                </Button>

                <Button
                  onClick={() => handleOAuthLogin("github")}
                  disabled={isLoading !== null}
                  className="flex w-full items-center justify-center gap-3 bg-gray-900 py-6 text-white hover:bg-gray-800 dark:bg-gray-800 dark:hover:bg-gray-700"
                >
                  {isLoading === "github" ? (
                    <div className="h-5 w-5 animate-spin rounded-full border-2 border-gray-300 border-t-white" />
                  ) : (
                    <svg
                      className="h-5 w-5"
                      fill="currentColor"
                      viewBox="0 0 24 24"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                    </svg>
                  )}
                  Continue with GitHub
                </Button>
              </div>

              <div className="mt-6">
                <div className="relative">
                  <div className="absolute inset-0 flex items-center">
                    <div className="w-full border-t border-gray-300 dark:border-gray-600" />
                  </div>
                  <div className="relative flex justify-center text-sm">
                    <span className="bg-white px-2 text-gray-500 dark:bg-gray-800 dark:text-gray-400">
                      Why connect?
                    </span>
                  </div>
                </div>

                <div className="mt-4 grid grid-cols-2 gap-3 text-sm text-gray-600 dark:text-gray-400">
                  <div className="flex items-center gap-2">
                    <HugeiconsIcon
                      icon={SparkIcon}
                      className="h-4 w-4 text-blue-500"
                    />
                    <span>Access Copilot models</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <HugeiconsIcon
                      icon={SparkIcon}
                      className="h-4 w-4 text-purple-500"
                    />
                    <span>Smart code completion</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <HugeiconsIcon
                      icon={SparkIcon}
                      className="h-4 w-4 text-green-500"
                    />
                    <span>Natural language coding</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <HugeiconsIcon
                      icon={SparkIcon}
                      className="h-4 w-4 text-orange-500"
                    />
                    <span>Context-aware suggestions</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <p className="mt-6 text-center text-sm text-gray-500 dark:text-gray-400">
            By continuing, you agree to our{" "}
            <a href="#" className="text-blue-500 hover:underline">
              Terms of Service
            </a>{" "}
            and{" "}
            <a href="#" className="text-blue-500 hover:underline">
              Privacy Policy
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}