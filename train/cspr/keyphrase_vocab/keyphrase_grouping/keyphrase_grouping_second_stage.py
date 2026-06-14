import json, os, asyncio, random
from argparse import ArgumentParser
from openai import AsyncOpenAI
from rapidfuzz import fuzz
from tqdm.asyncio import tqdm as async_tqdm
from dotenv import load_dotenv
from train.cspr.keyphrase_vocab.utils import _load_keyphrase_index

load_dotenv()

SYSTEM_PROMPT = (
    "You are a keyphrase deduplication assistant. "
    "Given a numbered list of keyphrases grouped by embedding similarity, "
    "partition them into one or more subgroups where every member within a subgroup "
    "can be used interchangeably with each other (same meaning in context). "
    "Output a JSON object with two keys: "
    "\"reasoning\": a single sentence explaining your grouping decision, and "
    "\"groups\": a list of lists of integer indices (0-based) corresponding to the input numbering. "
    "Every index must appear in exactly one subgroup — do not drop any."
)


def _make_user_message(members: list[str]) -> str:
    numbered = "\n".join(f"{i}: {kp}" for i, kp in enumerate(members))
    return f"Keyphrases:\n{numbered}"


def _is_fuzzy_trivial(members: list[str], threshold: int) -> bool:
    if any(any(c.isdigit() for c in m) for m in members):
        return False
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            if fuzz.ratio(members[i], members[j]) < threshold:
                return False
    return True


def _pick_root(members: list[str], keyphrase_index: dict) -> str:
    return max(members, key=lambda kp: len(keyphrase_index.get(kp, [])))


async def _split_group(
    client: AsyncOpenAI,
    model: str,
    members: list[str],
) -> list[list[str]]:
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _make_user_message(members)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        n = len(members)
        raw_groups = json.loads(response.choices[0].message.content).get("groups", [])
        result = []
        seen = set()
        for grp in raw_groups:
            valid_indices = [i for i in grp if isinstance(i, int) and 0 <= i < n and i not in seen]
            if not valid_indices:
                return [members]  # fallback
            seen.update(valid_indices)
            result.append([members[i] for i in valid_indices])

        if seen != set(range(n)):
            return [members]  # fallback: not all indices accounted for

        return result
    except Exception:
        print(members)
        return [members]  # fallback



async def _run(args):
    with open(args.input) as f:
        groups: dict = json.load(f)

    keyphrase_index = _load_keyphrase_index(args.keyphrase_index)

    multi = {root: members for root, members in groups.items() if len(members) >= 2}
    single = {root: members for root, members in groups.items() if len(members) < 2}

    fuzzy_accepted = {root: members for root, members in multi.items() if _is_fuzzy_trivial(members, args.fuzzy_threshold)}
    to_llm = {root: members for root, members in multi.items() if root not in fuzzy_accepted}

    print(f"Rapidfuzz accepted: {len(fuzzy_accepted)} groups | LLM processing: {len(to_llm)} groups")
    print("Starting LLM processing")

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)

    async def bounded(members):
        async with sem:
            return await _split_group(client, args.model, members)

    tasks = [bounded(members) for members in to_llm.values()]
    results = await async_tqdm.gather(*tasks, desc="LLM splitting")

    output = dict(single)
    for root, members in fuzzy_accepted.items():
        output[root] = members
    new_groups = 0
    for subgroups in results:
        for grp in subgroups:
            root = _pick_root(grp, keyphrase_index)
            output[root] = grp
        if len(subgroups) > 1:
            new_groups += len(subgroups) - 1

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Done. {len(output)} groups total (+{new_groups} new groups from LLM splits).")


def main():
    parser = ArgumentParser(description="LLM-based splitting of grouped keyphrases.")
    parser.add_argument("--input", required=True, help="JSON from stage 1: {canonical: [keyphrase, ...]}")
    parser.add_argument("--output", required=True, help="Output JSON")
    parser.add_argument("--keyphrase-index", required=True, help="Original keyphrase index (file or directory) for frequency-based root selection")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--fuzzy-threshold", type=int, default=98, help="fuzz.ratio threshold (0-100) to skip LLM")
    args = parser.parse_args()

    assert os.path.exists(args.input)
    assert os.path.exists(args.keyphrase_index)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
