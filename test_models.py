#!/usr/bin/env python3

import subprocess
import sys

SIMPLE_PROMPT = "Count to 10"
COMPLEX_PROMPT = (
    "Briefly summarize key papers and insights in LLM prompt optimization, with links"
)

# Test cases: (model_abbrev, description, extra_args, prompt)
# Using cheapest/fastest models from each provider
TEST_MODELS = [
    (
        "o4m",
        "OpenAI o4-mini (minimal reasoning/verbosity)",
        ["-re", "minimal", "-v", "low"],
        SIMPLE_PROMPT,
    ),
    ("cs", "Anthropic Claude Sonnet 5", [], SIMPLE_PROMPT),
    ("g", "Google Gemini 3.5 Flash", [], SIMPLE_PROMPT),
    (
        "5.6",
        "GPT-5.6 Sol (medium reasoning)",
        ["-re", "medium"],
        COMPLEX_PROMPT,
    ),
    (
        "d",
        "o3-deep-research",
        [],
        COMPLEX_PROMPT,
    ),
]


def test_model(model_abbrev, description, extra_args, prompt):
    """Test a single model by running gpt_cli.py"""
    try:
        cmd = [
            "python3",
            "gpt_cli.py",
            "-m",
            model_abbrev,
            "-p",
            prompt,
            "--disable-markdown",
        ] + extra_args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"✓ {description}")
            output = result.stdout.strip().replace("\n", " ")[:150]
            print(f"  {output}")
            return True
        else:
            print(f"✗ {description} (exit code: {result.returncode})")
            print(f"  {result.stderr[:100]}")
            return False

    except Exception as e:
        print(f"✗ {description} ({e})")
        return False


def main():
    results = {}
    for model_abbrev, description, extra_args, prompt in TEST_MODELS:
        results[description] = test_model(model_abbrev, description, extra_args, prompt)

    # Exit with error if any test failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
