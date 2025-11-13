from pathlib import Path

def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def banner(term):
    print("\n" + "=" * 30)
    print(f"  SCRAPING TERM: {term}")
    print("=" * 30)
