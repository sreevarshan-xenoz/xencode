You are compressing a working coding conversation into its durable layers. Keep immutable facts, decisions, the current task, completed work, and unresolved issues. Never invent facts. Keep the last 6 messages verbatim, source lines and all: content that arrived under a `[data]` line keeps that line with it whenever you carry the content forward.

# Current state
{state}

# Transcript tail (this is the working window being folded)
{transcript_tail}

Return ONLY this exact markdown shape:

## working-on
<one parallel sentence>

## completed
- <item>

## decisions
- <decision [d]>

## unresolved
- <item>

## recent (last 6 messages, verbatim)
{recent}