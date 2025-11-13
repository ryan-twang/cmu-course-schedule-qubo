# scrapers/time_slots.py

from datetime import datetime, timedelta

def generate_mwf_slots():
    slots = []
    start = datetime.strptime("08:00", "%H:%M")
    for i in range(12):  # up to ~6pm
        s = start + timedelta(minutes=60 * i)
        e = s + timedelta(minutes=50)
        slots.append((s.strftime("%H:%M"), e.strftime("%H:%M")))
    return slots

def generate_tr_slots():
    slots = []
    start = datetime.strptime("08:00", "%H:%M")
    for i in range(8):  # typical TR range
        s = start + timedelta(minutes=90 * i)
        e = s + timedelta(minutes=80)
        slots.append((s.strftime("%H:%M"), e.strftime("%H:%M")))
    return slots

MWF_SLOTS = generate_mwf_slots()
TR_SLOTS = generate_tr_slots()

def map_section_to_slots(section):
    """ Returns a list of integer slot IDs the section occupies. """

    if section["begin"] is None:
        return []

    begin = section["begin"]
    end = section["end"]
    days = section["days"]

    # Determine slot set by days
    if not days:
        return []

    result = []

    for d in days:
        if d in ["M", "W", "F"]:
            for idx, (s, e) in enumerate(MWF_SLOTS):
                if s <= begin < e:
                    result.append(f"MWF_{idx}")
        elif d in ["T", "R"]:
            for idx, (s, e) in enumerate(TR_SLOTS):
                if s <= begin < e:
                    result.append(f"TR_{idx}")

    return result
