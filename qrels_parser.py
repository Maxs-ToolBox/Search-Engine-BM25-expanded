from typing import Dict


def parse_qrels(file_path: str) -> Dict[str, Dict[str, int]]:
    """
    Parse a TREC qrels file.

    Each line usually looks like:
        topic_id  0  docno  relevance

    Returns:
        {
            "301": {"LA123190-0001": 1, "FBIS3-1": 2},
            "302": {"DOC123": 0, "DOC456": 1}
        }
    """
    qrels = {}

    with open(file_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            if len(parts) < 4:
                continue

            topic_id, _, docno, rel = parts[0], parts[1], parts[2], parts[3]

            try:
                rel = int(rel)
            except ValueError:
                continue

            if topic_id not in qrels:
                qrels[topic_id] = {}

            qrels[topic_id][docno] = rel

    return qrels


if __name__ == "__main__":
    path = "qrels.txt"
    qrels = parse_qrels(path)

    print(f"Loaded qrels for {len(qrels)} topics\n")

    # Show first 3 topics and first few docs for each
    shown = 0
    for topic_id, docs in qrels.items():
        print(f"Topic {topic_id}: {len(docs)} judged docs")
        sample_docs = list(docs.items())[:5]
        print(sample_docs)
        print()

        shown += 1
        if shown == 3:
            break