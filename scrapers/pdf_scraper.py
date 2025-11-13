# scrapers/pdf_scraper.py

import pdfplumber
import re

COURSE_HEADER_RE = re.compile(r"^(\d{5})\s+(.+)\s+(\d+\.\d+|VAR|TBA)$")
SECTION_RE = re.compile(
    r"^(Lec|A\d*|B\d*|C\d*|D\d*|E\d*|F\d*|G\d*|H\d*|I\d*|J\d*|K\d*|L\d*)\s+([MTWRF]+)\s+(\d{1,2}:\d{2}[AP]M)\s+(\d{1,2}:\d{2}[AP]M)"
)

def parse_pdf_schedule(pdf_path):
    """
    Returns list of raw dicts:
       { course, title, units, primary=None or dict, sections=[dict...] }
    """
    results = []
    current_course = None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text: 
                continue

            lines = text.split("\n")

            for line in lines:
                line = line.strip()

                # Detect Course Header
                header_match = COURSE_HEADER_RE.match(line)
                if header_match:
                    if current_course:
                        results.append(current_course)

                    course, title, units = header_match.groups()
                    current_course = {
                        "course": course,
                        "title": title.strip(),
                        "units": units if units != "VAR" else None,
                        "primary": None,
                        "sections": []
                    }
                    continue

                # Detect Section (Lec, A, B, etc.)
                sec_match = SECTION_RE.match(line)
                if sec_match and current_course:
                    sec, days, begin, end = sec_match.groups()

                    entry = {
                        "section": sec,
                        "days": days,
                        "begin": begin,
                        "end": end
                    }

                    if sec == "Lec":
                        current_course["primary"] = entry
                    else:
                        current_course["sections"].append(entry)

                    continue

                # Append multiline title
                if current_course and not COURSE_HEADER_RE.match(line):
                    # If no section match and after header, likely a line of title
                    if not SECTION_RE.match(line) and not line.startswith("Units"):
                        current_course["title"] += " " + line.strip()

    # Add last course
    if current_course:
        results.append(current_course)

    return results
