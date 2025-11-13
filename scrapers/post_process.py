# scrapers/postprocess.py

import json
from datetime import datetime
import re

def to_24h(t):
    if t is None:
        return None
    return datetime.strptime(t, "%I:%M%p").strftime("%H:%M")

def duration_minutes(begin, end):
    if begin is None or end is None:
        return None
    b = datetime.strptime(begin, "%H:%M")
    e = datetime.strptime(end, "%H:%M")
    return int((e - b).total_seconds() // 60)

def parse_days(days):
    if days is None:
        return None
    return list(days)

def infer_meeting_type(title, section):
    if section == "Lec":
        return "lecture"
    if "Lab" in title:
        return "lab"
    if "Forum" in title:
        return "forum"
    if "Studio" in title:
        return "studio"
    if "Seminar" in title:
        return "seminar"
    return "recitation"

def infer_mini(full_entry):
    title = full_entry["title"].lower()
    if "mini 1" in title:
        return ["Mini 1"]
    if "mini 2" in title:
        return ["Mini 2"]
    return ["Full"]

def requires_time_slot(primary, sections):
    # Internship, independent study, no meeting times
    if (primary is None or primary.get("begin") is None) and all(
        s.get("begin") is None for s in sections
    ):
        return False
    return True

def enrich(raw_courses):
    final = []

    for course in raw_courses:
        entry = {
            "course": course["course"],
            "title": course["title"],
            "units": float(course["units"]) if course["units"] else None,
            "primary": None,
            "sections": [],
            "minis": infer_mini(course),
            "is_primary_required": course["primary"] is not None,
            "requires_time_slot": None
        }

        # Process primary
        if course["primary"]:
            p = course["primary"]
            begin_24 = to_24h(p["begin"])
            end_24 = to_24h(p["end"])

            entry["primary"] = {
                "section_id": f"{course['course']}-Lec",
                "section": "Lec",
                "days": parse_days(p["days"]),
                "begin": begin_24,
                "end": end_24,
                "duration_minutes": duration_minutes(begin_24, end_24),
                "meeting_type": "lecture",
                "requires_time_slot": True
            }

        # Process sections
        for s in course["sections"]:
            begin_24 = to_24h(s["begin"])
            end_24 = to_24h(s["end"])

            enriched_section = {
                "section_id": f"{course['course']}-{s['section']}",
                "section": s["section"],
                "days": parse_days(s["days"]),
                "begin": begin_24,
                "end": end_24,
                "duration_minutes": duration_minutes(begin_24, end_24),
                "meeting_type": infer_meeting_type(course["title"], s["section"]),
                "requires_time_slot": begin_24 is not None
            }
            entry["sections"].append(enriched_section)

        entry["requires_time_slot"] = requires_time_slot(
            course["primary"], course["sections"]
        )

        final.append(entry)

    return final
