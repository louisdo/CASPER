import json, os, bz2
from argparse import ArgumentParser
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def read_single_wiki_dump_file(path: str, do_lower: bool = True) -> dict[str, str]:
    assert path.endswith(".json.bz2")

    base_name = os.path.basename(path)
    data = {}
    with bz2.open(path, "rt") as f:
        for i, line in enumerate(f):
            jline = json.loads(line)

            title = jline.get("title")
            opening_text = jline.get("opening_text")

            if not isinstance(opening_text, str) \
                or not isinstance(title, str): continue

            if do_lower:
                title = title.lower()

            data[title] = opening_text

    return data


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type = str, required = True, help = "Wikidump folder")
    parser.add_argument("--output", type = str, required = True, help = "Output file path")
    parser.add_argument("--workers", type = int, default = 8, help = "Number of processes")
    parser.add_argument("--no_lower", action = "store_true")

    args = parser.parse_args()

    dump_files = [
        os.path.join(args.input, f)
        for f in os.listdir(args.input)
        if f.endswith(".json.bz2")
    ]
    print(f"Found {len(dump_files)} dump files")

    aggregated: dict[str, str] = {}

    with ProcessPoolExecutor(max_workers = args.workers) as executor:
        futures = {
            executor.submit(read_single_wiki_dump_file, path, not args.no_lower): path
            for path in dump_files
        }
        with tqdm(total = len(dump_files), unit = "file") as pbar:
            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    aggregated.update(result)
                    pbar.set_postfix(entries = len(aggregated), file = os.path.basename(path))
                except Exception as e:
                    pbar.set_postfix(error = str(e), file = os.path.basename(path))
                pbar.update(1)

    print(f"Total entries: {len(aggregated)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok = True)
    with open(args.output, "w") as f:
        json.dump(aggregated, f)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
