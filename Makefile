.PHONY: stats

stats:
	uv run python cli.py stats --model "gpt-5,deepseek-chat,claude-sonnet-4-5-20250929,gemini-2.5-pro"
