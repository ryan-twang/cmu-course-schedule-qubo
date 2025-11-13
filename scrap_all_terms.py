"""
scrap_all_terms.py

Pipeline:
1. Download PDF for each term
2. Parse PDF into raw structured objects
3. Post-process into clean, QUBO-ready JSON
4. Save to CSV + JSON

Folders produced:
  data/{term}/{term}_raw.json
  data/{term}/{term}_clean.json
  data/{term}/{term}_clean.csv
"""

import time
from pathlib import Path
import pandas as pd
import json

from scrapers.pdf_downloader import download_pdf
from scrapers.pdf_scraper import parse_pdf_schedule
from scrapers.post_process import enrich


# TERM URLS (PDF-only)
PDF_LINKS = {
    "fall":        "https://enr-apps.as.cmu.edu/assets/SOC/sched_layout_fall.pdf",
    "spring":      "https://enr-apps.as.cmu.edu/assets/SOC/sched_layout_spring.pdf",
    "summer_one":  "https://enr-apps.as.cmu.edu/assets/SOC/sched_layout_summer_1.pdf",
    "summer_two":  "https://enr-apps.as.cmu.edu/assets/SOC/sched_layout_summer_2.pdf",
}


# UTIL
def banner(term):
    print("\n" + "=" * 30)
    print(f"  SCRAPING TERM: {term}")
    print("=" * 30)


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


# MAIN SCRAPE PIPELINE
def scrape_term(term):
    banner(term)

    if term not in PDF_LINKS:
        print(f"[ERROR] Unknown term: {term}")
        return

    out_dir = Path(f"data/{term}")
    mkdir(out_dir)

    # STEP 1 — Download PDF
    pdf_url = PDF_LINKS[term]
    pdf_path = out_dir / f"{term}.pdf"

    print(f"[PDF] Downloading: {pdf_url}")
    download_pdf(pdf_url, pdf_path)
    print(f"[OK] Saved PDF → {pdf_path}")

    # STEP 2 — Parse PDF → raw structured list
    print(f"[PARSE] Parsing PDF for term: {term}")
    raw = parse_pdf_schedule(pdf_path)

    raw_json_path = out_dir / f"{term}_raw.json"
    with open(raw_json_path, "w") as f:
        json.dump(raw, f, indent=2)

    print(f"[OK] Saved raw JSON → {raw_json_path}")

    # STEP 3 — Post-process → normalized, QUBO-ready
    print(f"[CLEAN] Normalizing and enhancing data...")
    clean = enrich(raw)

    clean_json_path = out_dir / f"{term}_clean.json"
    with open(clean_json_path, "w") as f:
        json.dump(clean, f, indent=2)

    print(f"[OK] Saved clean JSON → {clean_json_path}")

    # STEP 4 — Convert to flat CSV
    print(f"[CSV] Generating flattened schedule...")

    flat_rows = []
    for c in clean:
        # Primary lecture
        if c["primary"]:
            flat_rows.append({
                "course": c["course"],
                "title": c["title"],
                "units": c["units"],
                "section_id": c["primary"]["section_id"],
                "section": c["primary"]["section"],
                "days": "".join(c["primary"]["days"]) if c["primary"]["days"] else None,
                "begin": c["primary"]["begin"],
                "end": c["primary"]["end"],
                "duration": c["primary"]["duration_minutes"],
                "meeting_type": c["primary"]["meeting_type"],
                "requires_time_slot": c["primary"]["requires_time_slot"],
                "minis": "|".join(c["minis"])
            })

        # Recitations / labs / others
        for s in c["sections"]:
            flat_rows.append({
                "course": c["course"],
                "title": c["title"],
                "units": s.get("units", c["units"]),
                "section_id": s["section_id"],
                "section": s["section"],
                "days": "".join(s["days"]) if s["days"] else None,
                "begin": s["begin"],
                "end": s["end"],
                "duration": s["duration_minutes"],
                "meeting_type": s["meeting_type"],
                "requires_time_slot": s["requires_time_slot"],
                "minis": "|".join(c["minis"])
            })

    df = pd.DataFrame(flat_rows)
    csv_path = out_dir / f"{term}_clean.csv"
    df.to_csv(csv_path, index=False)

    print(f"[OK] Saved CSV → {csv_path}")

    print(f"[DONE] Completed term: {term}")


# ENTRY POINT
if __name__ == "__main__":
    terms = ["fall", "spring", "summer_one", "summer_two"]

    start = time.time()
    print("=== CMU PDF COURSE SCRAPER ===\n")

    for t in terms:
        scrape_term(t)

    print("\n==============================")
    print("  ALL TERMS COMPLETE")
    print("==============================")
    print(f"Runtime: {time.time() - start:.2f} sec")
