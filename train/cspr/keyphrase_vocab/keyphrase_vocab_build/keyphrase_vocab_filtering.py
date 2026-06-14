import json, os, asyncio
from argparse import ArgumentParser
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are a research vocabulary curator. "
    "Given a keyphrase, decide whether it represents a research topic or academic discipline — "
    "something that could be a subject of scientific study, a field of knowledge, or a technical concept. "
    "Respond with a JSON object with two keys: "
    "\"reasoning\": a single sentence explaining your decision, and "
    "\"answer\": either \"A\" (yes, it is a research topic/discipline) or \"B\" (no, it is not)."
)


def _should_include_variants(canonical: str) -> bool:
    return canonical.isupper() or len(canonical) <= 15


def _make_user_message(canonical: str, members: list[str]) -> str:
    variants = [m for m in members if m != canonical]
    aka = f'\n(also known as: {", ".join(variants)})' if variants and _should_include_variants(canonical) else ""
    return (
        f'Is the following keyphrase aligned to a research topic or academic discipline?\n\n'
        f'"{canonical}"{aka}\n\n'
        f'Provide your reasoning (1 sentence) then answer:\n'
        f'A. Yes\n'
        f'B. No'
    )


async def _classify(
    client: AsyncOpenAI,
    model: str,
    canonical: str,
    members: list[str],
) -> bool:
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _make_user_message(canonical, members)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = json.loads(response.choices[0].message.content)
        return str(raw.get("answer", "")).strip().upper().startswith("A")
    except Exception as e:
        print(f"[WARN] Error classifying '{canonical}': {e} — keeping by default")
        return True


async def _run(args):
    with open(args.input) as f:
        vocab: list[tuple[str, list[str]]] = json.load(f)

    # Load checkpoint if present
    checkpoint: dict[str, bool] = {}
    if args.checkpoint and os.path.exists(args.checkpoint):
        with open(args.checkpoint) as f:
            checkpoint = json.load(f)
        print(f"Resuming from checkpoint: {len(checkpoint)} phrases already classified")

    canonicals = [c for c, _ in vocab]
    to_classify = [(c, members) for c, members in vocab if c not in checkpoint]
    print(f"Total phrases: {len(canonicals)} | Already done: {len(checkpoint)} | Remaining: {len(to_classify)}")

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)

    async def bounded(canonical: str, members: list[str]) -> tuple[str, bool]:
        async with sem:
            result = await _classify(client, args.model, canonical, members)
            return canonical, result

    tasks = [bounded(c, members) for c, members in to_classify]
    results = await async_tqdm.gather(*tasks, desc="Filtering keyphrases")

    for phrase, keep in results:
        checkpoint[phrase] = keep

    if args.checkpoint:
        with open(args.checkpoint, "w") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)

    kept = [(c, members) for c, members in vocab if checkpoint.get(c, True)]
    removed = len(canonicals) - len(kept)
    print(f"Kept: {len(kept)} | Removed: {removed} ({removed / len(canonicals):.1%})")

    with open(args.output, "w") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

    print(f"Saved filtered vocab to {args.output}")


def main():
    parser = ArgumentParser(description="LLM-based filtering of non-academic keyphrases.")
    parser.add_argument("--input", required=True, help="Input vocab JSON: {canonical: [member, ...]}")
    parser.add_argument("--output", required=True, help="Output vocab JSON (same format, filtered)")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a checkpoint JSON for resume support. "
             "Stores {phrase: bool} classifications so interrupted runs can resume.",
    )
    args = parser.parse_args()

    assert os.path.exists(args.input), f"Input file not found: {args.input}"
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
