# Browser-Based LLM Access Plan for Xencode

**Document Type:** Technical Architecture Plan  
**Status:** Research-Complete, Ready for Implementation  
**Date:** April 3, 2026  
**Version:** 1.0

---

## Executive Summary

This plan evaluates the feasibility of connecting Xencode to LLM providers (ChatGPT, Claude, Gemini, Qwen, Z.AI) through browser-based login and interaction — **bypassing the need for API keys entirely**. Users would authenticate via OAuth/device flow in their browser, and Xencode would communicate with the LLM through the authenticated session.

**Final Verdict: Partially feasible with significant caveats.** Three approaches exist with different risk profiles:

| Approach | Feasibility | ToS Risk | Implementation Cost |
|----------|------------|----------|-------------------|
| **A. OAuth API (Recommended)** | ✅ High | None | Low |
| **B. Session Token Relay** | ⚠️ Medium | Medium | Medium |
| **C. Browser Automation** | ❌ Low | High | High |

---

## 1. Provider OAuth/Device Flow Support Matrix

| Provider | OAuth Support | Device Flow | Free Tier | Notes |
|----------|:---:|:---:|:---:|-------|
| **Google Gemini** | ✅ | ✅ | ✅ 60 RPM free | Standard Google OAuth 2.0, well-documented |
| **OpenRouter** | ✅ PKCE | ✅ | ✅ Free tier | User-controlled key generation, ideal for xencode |
| **Qwen (Alibaba)** | ✅ | ✅ | ✅ ~2000 req/day | Qwen OAuth for Qwen Code free tier |
| **Z.AI (Zhipu)** | ✅ | ✅ | ✅ Free tier | OAuth 2.1 added March 2025 |
| **HuggingFace** | ✅ | ✅ | ✅ Free inference | OAuth/OIDC, tokens work for Inference API |
| **Anthropic Claude** | ⚠️ Restricted | ⚠️ Restricted | ❌ | OAuth tokens `sk-ant-oat01-*` **blocked for 3rd party** since Feb 2026 |
| **OpenAI ChatGPT** | ❌ | ❌ | ❌ | OAuth only for ChatGPT Apps SDK (plugins inside ChatGPT) |
| **xAI Grok** | ❌ | ❌ | ❌ | API key only ($25 free credit on signup) |

### Key Finding: No Provider Supports Full Browser Automation

**No major LLM provider supports a "login in browser → use web UI programmatically" flow.** The closest alternatives are:

1. **OAuth API access** — proper, ToS-compliant, but still uses API calls (not browser UI)
2. **Session token extraction** — reverse-engineered, ToS-violating, fragile
3. **Browser automation (Playwright)** — most fragile, highest ToS risk

---

## 2. Three Implementation Approaches

### Approach A: OAuth API Integration (RECOMMENDED)

**How it works:**
1. User launches `xencode`
2. Chooses "Connect Provider" from menu
3. Xencode opens browser to provider's OAuth consent page
4. User clicks "Allow" → OAuth callback returns access token
5. Xencode uses token for **official API calls** (not browser UI)
6. Token auto-refreshes via refresh_token

**User Flow:**
```
$ xencode providers
🔌 Available LLM Providers
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Provider      ┃ Status   ┃ Auth Method    ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ google_gemini │ ☁️ Cloud │ [Browser Login] │  ← OAuth
│ openrouter    │ ☁️ Cloud │ [Browser Login] │  ← OAuth PKCE
│ qwen          │ ☁️ Cloud │ [Browser Login] │  ← OAuth
│ z.ai          │ ☁️ Cloud │ [Browser Login] │  ← OAuth 2.1
│ huggingface   │ ☁️ Cloud │ [Browser Login] │  ← OAuth/OIDC
│ openai        │ ☁️ Cloud │ [API Key]       │  ← No OAuth
│ anthropic     │ ☁️ Cloud │ [API Key]       │  ← No OAuth for 3rd party
│ ollama        │ ✅ Local │ [Auto-Detect]   │  ← Local
└───────────────┴──────────┴────────────────┘

Select a provider to connect: google_gemini
🌐 Opening browser for Google OAuth login...
✅ Connected to Google Gemini (free tier, 60 RPM)
```

**Supported Providers:**
- Google Gemini ✅ (Google standard OAuth 2.0)
- OpenRouter ✅ (PKCE flow, creates API key automatically)
- Qwen ✅ (Qwen OAuth free tier)
- Z.AI ✅ (OAuth 2.1)
- HuggingFace ✅ (OAuth/OIDC)

**NOT Supported (require API keys):**
- OpenAI ChatGPT ❌ (no OAuth for end users)
- Anthropic Claude ❌ (OAuth blocked for 3rd party since Feb 2026)
- xAI Grok ❌ (API key only)

**Implementation:**
```python
# Uses existing OAuth libraries
import aiohttp
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

class OAuthFlow:
    def __init__(self, provider_config):
        self.provider = provider_config
        self.local_port = 8765
    
    async def authenticate(self) -> str:
        """Open browser, wait for callback, return access_token."""
        # 1. Start local callback server
        # 2. Open browser to OAuth consent URL
        # 3. Wait for callback with code
        # 4. Exchange code for access_token + refresh_token
        # 5. Store token securely
        return access_token
    
    async def refresh_token(self) -> str:
        """Auto-refresh expired access tokens."""
        ...
```

**Pros:**
- ✅ 100% ToS-compliant
- ✅ No account ban risk
- ✅ Stable (official APIs don't change UI)
- ✅ Auto-refresh tokens
- ✅ Free tiers available
- ✅ Works with existing `UnifiedModelClient`

**Cons:**
- ❌ OpenAI/Anthropic/Grok still need API keys
- ❌ Some OAuth flows have rate limits lower than API keys

---

### Approach B: Session Token Relay (MEDIUM RISK)

**How it works:**
1. User logs into ChatGPT/Claude in their browser normally
2. Xencode reads the session token from the browser's cookie storage
3. Xencode uses the token with **unofficial API wrappers** (reverse-engineered)
4. Token expires → user re-authenticates in browser

**User Flow:**
```
$ xencode providers
Select a provider to connect: chatgpt
🌐 Please log into chatgpt.com in your browser first, then press Enter.
⏳ Reading session token from browser...
✅ Connected to ChatGPT as user@example.com (session expires in 24h)
```

**How Token Extraction Works:**
```python
# For Chromium-based browsers (Chrome, Edge, Brave)
import sqlite3
from pathlib import Path
from cryptography.fernet import Fernet

def get_chatgpt_session_token():
    """Extract ChatGPT session token from Chrome cookie store."""
    cookie_path = Path.home() / ".config/google-chrome/Default/Cookies"
    # Decrypt cookies using OS keychain
    # Find __Secure-next-auth.session-token for chatgpt.com
    return session_token
```

**Existing Libraries:**
| Library | Provider | Status |
|---------|----------|--------|
| `reverse-engineered-chatgpt` (Zai-Kun) | ChatGPT | Active |
| `PawanOsman/ChatGPT` | ChatGPT proxy | Active |
| `chatgpt-automation-mcp` (cbusillo) | ChatGPT | ⚠️ Archived |

**Pros:**
- ✅ No API key needed for ChatGPT (uses existing web session)
- ✅ Free (uses user's ChatGPT Plus/Free subscription)
- ✅ Works with any ChatGPT model available on web UI

**Cons:**
- ❌ **Violates OpenAI Terms of Use** — "automation abuse" is a ban trigger
- ❌ **Account suspension risk** — documented bans for automation
- ❌ Session tokens expire (hours to days) — requires re-authentication
- ❌ Fragile — UI changes break reverse-engineered APIs
- ❌ Cloudflare Turnstile anti-bot may block requests
- ❌ No Claude support (Anthropic cracked down Feb 2026)
- ❌ Gives xencode full account access (security concern)

**Risk Assessment:**

| Risk | Likelihood | Impact |
|------|-----------|--------|
| Account ban | Medium-High | User loses ChatGPT access |
| Token revocation | High | Temporary disruption |
| ToS violation | Certain | Legal liability for xencode |
| Data exposure | Low-Medium | Session token = full account access |

---

### Approach C: Browser Automation via Playwright (HIGH RISK)

**How it works:**
1. User logs into LLM provider in a managed browser instance
2. Xencode uses Playwright to programmatically interact with the web UI
3. Sends messages by filling textareas, clicking send buttons
4. Reads responses by scraping DOM elements
5. Handles file uploads via Playwright file chooser

**Architecture:**
```
┌─────────────────────────────────────────────────┐
│                    Xencode                       │
│  ┌───────────────┐    ┌──────────────────┐      │
│  │  User Prompt  │───>│  LLMWebAdapter   │      │
│  └───────────────┘    │  ┌────────────┐  │      │
│                       │  │ Playwright │  │      │
│  ┌───────────────┐    │  │ Browser    │  │      │
│  │  LLM Response │<───│  │ Instance   │  │      │
│  └───────────────┘    │  └─────┬──────┘  │      │
│                       │        │         │      │
│                       │  ┌─────▼──────┐  │      │
│                       │  │ DOM Parser │  │      │
│                       │  │ & Scraper  │  │      │
│                       │  └────────────┘  │      │
│                       └──────────────────┘      │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Browser Window    │
│  ┌─────────────────┐│
│  │  chatgpt.com    ││  ← Playwright controls this
│  │  [textarea]     ││
│  │  [Send button]  ││
│  │  [response]     ││
│  └─────────────────┘│
└─────────────────────┘
```

**Implementation with `browser-use`:**
```python
from browser_use import Agent, Controller
from langchain_openai import ChatOpenAI  # or any LLM

async def chat_via_browser(provider: str, prompt: str) -> str:
    """Use browser-use to interact with LLM web UI."""
    url_map = {
        "chatgpt": "https://chatgpt.com",
        "claude": "https://claude.ai",
        "gemini": "https://gemini.google.com",
    }
    
    agent = Agent(
        task=f"Go to {url_map[provider]}, type '{prompt}' in the chat, "
             f"click Send, wait for response, and copy the response text.",
        llm=ChatOpenAI(model="gpt-4o-mini"),  # needs an LLM to drive the browser agent!
    )
    result = await agent.run()
    return result
```

**Existing Tools:**
| Tool | Stars | Status | Notes |
|------|-------|--------|-------|
| `browser-use/browser-use` | 86k | ✅ Active | Full AI browser agent, Playwright-based |
| `chatgpt-automation-mcp` | ~hundreds | ⚠️ Archived | Playwright MCP for ChatGPT |
| `playwright-mcp` | ~hundreds | ✅ Active | General Playwright MCP server |

**Critical Irony:** Browser automation needs **its own LLM** to interpret the UI and decide what to click. You'd need an API key or local model to automate a web UI. This creates a bootstrapping problem.

**Pros:**
- ✅ Theoretically works with ANY LLM web UI
- ✅ Can handle file uploads, image analysis, code interpreter
- ✅ Uses the user's existing subscription (Free/Plus/Pro)
- ✅ `browser-use` is a mature, well-maintained library

**Cons:**
- ❌ **Violates all major providers' ToS** (OpenAI, Anthropic, Google)
- ❌ **Highest account ban risk** — detectable automation patterns
- ❌ **Needs an LLM to drive the browser agent** (chicken-and-egg problem)
- ❌ **Extremely fragile** — UI changes break selectors constantly
- ❌ **Slow** — each message requires browser rendering + DOM scraping (~5-15s overhead)
- ❌ **Anti-bot measures** — Cloudflare Turnstile, reCAPTCHA v3, behavioral analysis
- ❌ **No concurrent sessions** — single browser = single conversation at a time
- ❌ **Resource heavy** — full Chromium instance per provider
- ❌ **ChatGPT automation repo is archived** — no maintenance

**Risk Assessment:**

| Risk | Likelihood | Impact |
|------|-----------|--------|
| Account ban | **Very High** | User loses access permanently |
| Cloudflare block | **High** | Requests silently fail |
| UI breaking changes | **Certain** | Constant maintenance burden |
| ToS violation | **Certain** | Legal liability for xencode |
| Resource exhaustion | **Medium** | Full browser = 500MB+ RAM each |

---

## 3. Recommended Implementation Plan

### Phase 1: OAuth Browser Login (Weeks 1-2) ✅ Recommended

Implement proper OAuth flows for providers that support it. This is the **only ToS-compliant way** to offer "login in browser" without API keys.

**Target Providers:**
1. Google Gemini (standard Google OAuth 2.0)
2. OpenRouter (PKCE flow — creates API key automatically)
3. Qwen (Qwen OAuth free tier)
4. HuggingFace (OAuth/OIDC)
5. Z.AI (OAuth 2.1)

**Implementation Steps:**

#### Step 1: OAuth Flow Engine
```
xencode/
└── oauth/
    ├── __init__.py          # OAuthFlow class, provider registry
    ├── providers/
    │   ├── google.py        # Google OAuth 2.0 (Gemini)
    │   ├── openrouter.py    # OpenRouter PKCE
    │   ├── qwen.py          # Qwen OAuth
    │   ├── zai.py           # Z.AI OAuth 2.1
    │   └── huggingface.py   # HF OAuth/OIDC
    ├── token_store.py       # Secure token storage (keyring/encrypted file)
    ├── callback_server.py   # Local HTTP server for OAuth callbacks
    └── refresh_manager.py   # Auto-refresh expired tokens
```

#### Step 2: CLI Integration
```python
# New CLI commands
@cli.command()
def connect():
    """Connect to an LLM provider via browser login"""
    # Lists providers with OAuth support
    # Opens browser for selected provider
    # Stores token securely
    
@cli.command()
def disconnect():
    """Disconnect from an LLM provider"""
    
@cli.command()
def connections():
    """Show active provider connections"""
```

#### Step 3: Unified Client Integration
```python
# Wire OAuth tokens into existing UnifiedModelClient
from xencode.unified_model_client import UnifiedModelClient
from xencode.oauth import OAuthFlow

client = UnifiedModelClient()
oauth = OAuthFlow()
token = await oauth.authenticate("google_gemini")
client.set_provider_token("google_gemini", token)
```

#### Step 4: TUI Integration
```
# In TUI settings panel:
┌──────────────────────────────────────────┐
│  Connected Providers                     │
├──────────────────────────────────────────┤
│  ✅ Google Gemini  (OAuth, free tier)    │
│     Connected as user@gmail.com          │
│     [Disconnect] [Re-authenticate]       │
│                                          │
│  ✅ OpenRouter     (OAuth PKCE)          │
│     Connected via PKCE flow              │
│     [Disconnect] [Re-authenticate]       │
│                                          │
│  ❌ OpenAI         (API Key required)    │
│     [Enter API Key]                      │
│                                          │
│  ❌ Anthropic      (API Key required)    │
│     [Enter API Key]                      │
│                                          │
│  [+ Connect New Provider]                │
└──────────────────────────────────────────┘
```

**Effort:** ~80-120 hours (2-3 weeks for 1 developer)

---

### Phase 2: Session Token Relay (Weeks 3-4) ⚠️ Optional

Add support for reading browser session tokens for providers that lack OAuth.

**Target Providers:**
- ChatGPT (via cookie extraction from Chromium browsers)

**Implementation:**
```
xencode/
└── session_relay/
    ├── __init__.py
    ├── token_extractor.py   # Read cookies from Chrome/Edge/Firefox
    ├── unofficial_api.py    # Wrapper around reverse-engineered APIs
    └── refresh_monitor.py   # Detect token expiration, prompt re-login
```

**Requirements to Document:**
- ⚠️ Users must accept ToS violation risk
- ⚠️ Account bans are possible
- ⚠️ This feature is opt-in with explicit warning

**Effort:** ~40-60 hours

---

### Phase 3: Browser Automation (NOT RECOMMENDED) ❌

Based on research, we **strongly recommend against** implementing browser automation for LLM interaction because:

1. **It needs an LLM to drive the automation** — chicken-and-egg problem
2. **Highest ToS violation risk** — guaranteed ban triggers
3. **Most fragile approach** — UI changes break everything
4. **Slowest** — 5-15s overhead per message
5. **The best tool (`chatgpt-automation-mcp`) is archived**
6. **Anthropic explicitly banned this in Feb 2026**

If a user insists on this capability, recommend `browser-use` as a separate tool that Xencode can integrate with as a plugin, not as a core feature.

---

## 4. Provider-by-Provider Feasibility

| Provider | Browser Login | API Key Alternative | Recommended Path |
|----------|:---:|:---:|---|
| **Google Gemini** | ✅ OAuth | ✅ Free tier via OAuth | **Phase 1** — Full OAuth |
| **OpenRouter** | ✅ PKCE | ✅ Auto-creates key | **Phase 1** — PKCE flow |
| **Qwen** | ✅ OAuth | ✅ Free tier | **Phase 1** — OAuth |
| **Z.AI** | ✅ OAuth 2.1 | ✅ Free tier | **Phase 1** — OAuth |
| **HuggingFace** | ✅ OIDC | ✅ Free inference | **Phase 1** — OAuth |
| **ChatGPT** | ❌ No OAuth | ⚠️ Session tokens | **Phase 2** — Token relay (optional) |
| **Claude** | ❌ OAuth blocked | ❌ None | **API key only** |
| **Grok/xAI** | ❌ No OAuth | ❌ None | **API key only** ($25 free) |
| **Ollama** | ✅ N/A | ✅ Local | **Already works** |

---

## 5. Technical Architecture (Recommended)

```
┌─────────────────────────────────────────────────────────────────┐
│                         Xencode                                  │
│                                                                   │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐    │
│  │     CLI      │   │     TUI      │   │   API Server     │    │
│  └──────┬───────┘   └──────┬───────┘   └────────┬─────────┘    │
│         └──────────────────┼────────────────────┘               │
│                            │                                     │
│  ┌─────────────────────────▼───────────────────────────┐        │
│  │              UnifiedModelClient                      │        │
│  │  • Provider routing (7 providers)                   │        │
│  │  • Automatic fallback                               │        │
│  │  • Token management                                 │        │
│  └─────────────────────┬───────────────────────────────┘        │
│                        │                                         │
│  ┌─────────────────────▼───────────────────────────────┐        │
│  │              Auth Layer                              │        │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │        │
│  │  │ OAuth Engine│  │Session Relay │  │ API Key Mgr│  │        │
│  │  │ (Phase 1)   │  │ (Phase 2)    │  │            │  │        │
│  │  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘  │        │
│  │         │                │                 │          │        │
│  │  ┌──────▼──────┐  ┌──────▼──────┐  ┌───────▼──────┐  │        │
│  │  │Browser Login│  │Cookie Read  │  │Env/Config    │  │        │
│  │  │(OAuth flow) │  │(Token ext)  │  │Variables     │  │        │
│  │  └─────────────┘  └─────────────┘  └──────────────┘  │        │
│  └──────────────────────────────────────────────────────┘        │
│                                                                   │
│  ┌───────────────────────────────────────────────────────┐        │
│  │              Token Store (Secure)                      │        │
│  │  • Encrypted file (~/.xencode/tokens.enc)             │        │
│  │  • OS keyring integration (optional)                   │        │
│  │  • Auto-refresh via refresh_tokens                     │        │
│  └───────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
         │                │                │
         ▼                ▼                ▼
    ┌─────────┐    ┌──────────┐    ┌──────────────┐
    │ Google  │    │ OpenAI   │    │ Ollama       │
    │ OAuth   │    │ API Key  │    │ Local        │
    └─────────┘    └──────────┘    └──────────────┘
```

---

## 6. Security Considerations

### Token Storage
```python
# Priority order for token storage
1. OS Keyring (most secure) — use `keyring` Python package
2. Encrypted file — AES-256 with user passphrase
3. Environment variables — for CI/CD and server deployments

class SecureTokenStore:
    def store(self, provider: str, token: dict):
        """Store token with encryption."""
        # 1. Try OS keyring
        # 2. Fall back to encrypted file
        pass
    
    def get(self, provider: str) -> dict:
        """Retrieve and decrypt token."""
        pass
    
    def is_expired(self, provider: str) -> bool:
        """Check if token needs refresh."""
        pass
```

### OAuth Security
- Use PKCE for all public client flows (no client secrets)
- Store refresh_tokens encrypted
- Implement token rotation (new refresh token on each refresh)
- Set appropriate token scopes (minimal permissions)

### Session Token Relay Risks
- Clearly document ToS violation risk to users
- Require explicit opt-in with warning dialog
- Never store session tokens longer than necessary
- Clear tokens on disconnect

---

## 7. Implementation Timeline

| Phase | Duration | Deliverables |
|-------|----------|-------------|
| **Phase 1: OAuth** | 2-3 weeks | Google, OpenRouter, Qwen, Z.AI, HF OAuth flows |
| **Phase 2: Session Relay** | 1-2 weeks | ChatGPT cookie extraction, unofficial API wrapper |
| **Phase 3: Testing** | 1 week | Integration tests, user acceptance testing |
| **Phase 4: Documentation** | 1 week | User guides, security documentation, FAQ |

**Total:** 5-7 weeks

---

## 8. What NOT to Build

| Feature | Why Not | Alternative |
|---------|---------|-------------|
| **Playwright browser automation** | Needs LLM to drive it; ToS violations; archived tools | Use official OAuth/API |
| **Auto-account creation** | CAPTCHA, phone verification, ban risk | User creates own account |
| **Shared token pools** | ToS violation; data exposure; legal risk | Per-user tokens only |
| **Claude OAuth bypass** | Explicitly blocked by Anthropic since Feb 2026 | API key only |
| **ChatGPT UI scraping** | Cloudflare Turnstile; ban risk; ToS violation | Official API or session relay with warnings |

---

## 9. Cost Comparison

| Provider | OAuth (Free Tier) | API Key (Pay-per-use) | Session Relay (Subscription) |
|----------|------------------|---------------------|----------------------------|
| **Google Gemini** | ✅ 60 RPM free | $1.25-7.50/1M tokens | $20/mo (Google One AI Premium) |
| **OpenRouter** | ✅ Free models available | $0.001-10/1M tokens | N/A (aggregator) |
| **Qwen** | ✅ 2000 req/day free | ¥0.01-0.1/1K tokens | N/A |
| **Z.AI** | ✅ Free tier | ¥0.01-0.05/1K tokens | N/A |
| **ChatGPT** | N/A | $0.002-10/1M tokens | $20/mo (Plus) |
| **Claude** | N/A | $3-15/1M tokens | $20/mo (Pro) |

---

## 10. Final Recommendation

### Do This:
1. ✅ **Implement OAuth browser login** for Google Gemini, OpenRouter, Qwen, Z.AI, and HuggingFace (Phase 1)
2. ✅ **Add session token relay** as opt-in for ChatGPT with explicit ToS warnings (Phase 2)
3. ✅ **Keep API key support** for OpenAI, Anthropic, and Grok
4. ✅ **Keep Ollama** as the default local provider

### Don't Do This:
1. ❌ **Don't build browser automation** (Playwright/browser-use) — it's fragile, slow, illegal under ToS, and needs its own LLM
2. ❌ **Don't attempt Claude OAuth bypass** — Anthropic explicitly blocked it
3. ❌ **Don't share tokens** between users — security nightmare

### The Bottom Line

**"Login in browser → no API key needed" is achievable for ~5 out of 8 major providers through proper OAuth flows.** For the remaining 3 (OpenAI, Anthropic, xAI), API keys remain the only supported method. Browser automation as a universal fallback is technically possible but practically inadvisable due to ToS violations, account ban risks, and the ironic requirement of needing an LLM to drive the browser automation.

---

*This document is based on research conducted in April 2026. Provider policies may change. Always check current provider documentation before implementation.*
