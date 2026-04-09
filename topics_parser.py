import re
from typing import List, Dict


def parse_topics(file_path: str) -> List[Dict[str, str]]:
    """
    Parse TREC topics file (Robust04 format).

    Returns:
        List of dicts: [{"topic_id": "301", "query": "text"}, ...]
    """

    with open(file_path, "r", encoding="latin-1") as f:
        content = f.read()

    # Split into <top> blocks
    topics = re.findall(r"<top>(.*?)</top>", content, re.DOTALL)

    results = []

    for topic in topics:
        # Extract topic number
        num_match = re.search(r"<num>\s*Number:\s*(\d+)", topic)
        topic_id = num_match.group(1).strip() if num_match else None

        # Extract title (this is the query we use)
        title_match = re.search(r"<title>\s*(.*)", topic)
        query = title_match.group(1).strip() if title_match else ""

        if topic_id and query:
            results.append({
                "topic_id": topic_id,
                "query": query
            })

    return results


# 🔥 OPTIONAL: quick test runner
if __name__ == "__main__":
    path = "topics.txt"  # change this later to your actual file
    topics = parse_topics(path)

    print(f"Loaded {len(topics)} topics\n")

    for t in topics[:5]:
        print(t)