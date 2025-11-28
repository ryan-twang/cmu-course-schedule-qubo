"""
End-to-end example for building a course-scheduling QUBO and solving it with QAOA.

The script demonstrates three flows:
1) Build a QUBO from course/section data with clear variable naming and constraints.
2) Run QAOA locally with PennyLane's default simulator to estimate a feasible schedule.
3) Show how to submit the same QAOA circuit to Amazon Braket (or run with Qiskit) by
   swapping devices.

This is meant to be copy/paste runnable once you install the optional dependencies:
    pip install pennylane pennylane-braket qiskit qiskit-algorithms
and configure AWS credentials for Braket if you want hardware runs.
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pennylane as qml

# ---------------------------------------------------------------------------
# Example data
# ---------------------------------------------------------------------------
# The data structure mirrors the user-provided course dictionary list. Each course has a
# course code and a list of section meeting times. Binary variables encode whether a
# section is chosen (1) or not (0).
EXAMPLE_COURSES: List[Dict] = [
    {
        "course": "48313",
        "title": "New Pedogogies: Unreasonable Architecture New Pedogogies Storycraft New Pedogogies Intermundium: Labyrinth of Invisible Narratives 48315 Enviroment I: Climate & Energy in Architecture 9.0 Lec TR 03:30PM 04:50PM Pittsburgh, Pennsylvania",
        "units": 9.0,
        "primary": None,
        "sections": [
            {
                "section_id": "48313-A",
                "section": "A",
                "days": ["T", "R"],
                "begin": "11:00",
                "end": "12:20",
                "duration_minutes": 80,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48313-D",
                "section": "D",
                "days": ["M", "W"],
                "begin": "11:00",
                "end": "12:20",
                "duration_minutes": 80,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48313-E",
                "section": "E",
                "days": ["W", "F"],
                "begin": "09:30",
                "end": "10:50",
                "duration_minutes": 80,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48313-A",
                "section": "A",
                "days": ["T"],
                "begin": "11:00",
                "end": "12:20",
                "duration_minutes": 80,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48313-B",
                "section": "B",
                "days": ["T"],
                "begin": "11:00",
                "end": "12:20",
                "duration_minutes": 80,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48313-C",
                "section": "C",
                "days": ["W"],
                "begin": "11:00",
                "end": "12:20",
                "duration_minutes": 80,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48313-D",
                "section": "D",
                "days": ["R"],
                "begin": "12:30",
                "end": "13:50",
                "duration_minutes": 80,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
        ],
        "minis": ["Full"],
        "is_primary_required": False,
        "requires_time_slot": True,
    },
    {
        "course": "48321",
        "title": "American City: Architectural and Urban Design History: Architectural and Urban Design History 48353 Monopolis for the Masses 9.0 A MW 11:00AM 12:20PM Pittsburgh, Pennsylvania 48356 Color Drawing 9.0 A W 09:00AM 11:50AM Pittsburgh, Pennsylvania 48374 History of Architecture in the Islamic World- A Primer 9.0 A MF 11:00AM 12:20PM Pittsburgh, Pennsylvania 48386 Portfolio & Resume Preparation (UG) 3.0 A2 F 11:00AM 12:20PM Pittsburgh, Pennsylvania 48400 Architecture Design Studio: Praxis Studio 3 18.0 Lec TR 02:00PM 04:50PM Pittsburgh, Pennsylvania 48407 Carnival Pavilion VAR A W 07:00PM 08:20PM Pittsburgh, Pennsylvania 48432 Environment II: Design Integration of Active Building Systems 9.0 A TR 09:30AM 10:50AM Pittsburgh, Pennsylvania 48433 Afrofuturism and Othered Ways of Seeing & Being in the World 9.0 A MW 11:00AM 12:20PM Pittsburgh, Pennsylvania 48434 Aztec to Zacatecas: Mesoamerican & Spanish Colonial Arch of Mexico & Guatemala 9.0 A TR 11:00AM 12:20PM Pittsburgh, Pennsylvania 48445 Design Consciousness: Working with Diverse Populations 9.0 A MW 09:00AM 10:20AM Pittsburgh, Pennsylvania 48490 Undergraduate Internship 3.0 A TBA Pittsburgh, Pennsylvania 48500 Advanced Synthesis Options Studio 0,18 Lec TR 02:00PM 04:50PM Pittsburgh, Pennsylvania 48506 Shape Grammars and Computational Design 9.0 A WF 02:00PM 03:20PM Pittsburgh, Pennsylvania 48507 Carnival Pavilion 24-36 A W 07:00PM 08:20PM Pittsburgh, Pennsylvania 48508 Thesis Pre-Production VAR A TBA Pittsburgh, Pennsylvania B TBA Pittsburgh, Pennsylvania C TBA Pittsburgh, Pennsylvania D TBA Pittsburgh, Pennsylvania E TBA Pittsburgh, Pennsylvania 48525 Thesis Seminar 9.0 A MF 12:30PM 01:50PM Pittsburgh, Pennsylvania 48531 Fabricating Customization: Prototype 9.0 A MF 09:30AM 10:50AM Pittsburgh, Pennsylvania 48543 Color Constructs 9.0 A MW 11:00AM 12:20PM Pittsburgh, Pennsylvania 48551 Lab for Cybernetics: Engaging Wicked Challenges 9.0 A WF 02:00PM 03:20PM Pittsburgh, Pennsylvania 48555 Introduction to Architectural Robotics 9.0 A MW 02:00PM 03:50PM Pittsburgh, Pennsylvania 48561 Professional Development 3.0 A1 R 11:00AM 12:20PM Pittsburgh, Pennsylvania 48562 Architectural Intelligence: Theories and Frameworks of Generative AI in Design 3.0 A1 MW 09:30AM 10:50AM Pittsburgh, Pennsylvania 48568 Advanced CAD, BIM, and 3D Visualization VAR A TBA Pittsburgh, Pennsylvania 48599 Independent Study 3-9 A TBA Pittsburgh, Pennsylvania B TBA Pittsburgh, Pennsylvania C TBA Pittsburgh, Pennsylvania D TBA Pittsburgh, Pennsylvania E TBA Pittsburgh, Pennsylvania F TBA Pittsburgh, Pennsylvania G TBA Pittsburgh, Pennsylvania H TBA Pittsburgh, Pennsylvania I TBA Pittsburgh, Pennsylvania J TBA Pittsburgh, Pennsylvania K TBA Pittsburgh, Pennsylvania L TBA Pittsburgh, Pennsylvania M TBA Pittsburgh, Pennsylvania N TBA Pittsburgh, Pennsylvania O TBA Pittsburgh, Pennsylvania P TBA Pittsburgh, Pennsylvania Q TBA Pittsburgh, Pennsylvania R TBA Pittsburgh, Pennsylvania S TBA Pittsburgh, Pennsylvania T TBA Pittsburgh, Pennsylvania 48606 Shape Grammars and Computational Design 9.0 A WF 02:00PM 03:20PM Pittsburgh, Pennsylvania 48607 Architectural Agency: Dig Where You Stand! VAR A W 02:00PM 03:50PM Pittsburgh, Pennsylvania 48608 Thesis Pre-Production VAR A TBA Pittsburgh, Pennsylvania B TBA Pittsburgh, Pennsylvania C TBA Pittsburgh, Pennsylvania D TBA Pittsburgh, Pennsylvania E TBA Pittsburgh, Pennsylvania 48610 Between Mountains and Seas: Detailing Architecture and Folklore 9.0 A MW 11:00AM 12:20PM Pittsburgh, Pennsylvania",
        "units": 9.0,
        "primary": None,
        "sections": [
            {
                "section_id": "48321-A",
                "section": "A",
                "days": ["T", "R"],
                "begin": "15:30",
                "end": "16:50",
                "duration_minutes": 80,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48321-A",
                "section": "A",
                "days": ["T", "R"],
                "begin": "13:00",
                "end": "16:50",
                "duration_minutes": 230,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48321-B",
                "section": "B",
                "days": ["T", "R"],
                "begin": "13:00",
                "end": "16:50",
                "duration_minutes": 230,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48321-C",
                "section": "C",
                "days": ["T", "R"],
                "begin": "13:00",
                "end": "16:50",
                "duration_minutes": 230,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48321-D",
                "section": "D",
                "days": ["T", "R"],
                "begin": "13:00",
                "end": "16:50",
                "duration_minutes": 230,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48321-E",
                "section": "E",
                "days": ["T", "R"],
                "begin": "13:00",
                "end": "16:50",
                "duration_minutes": 230,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48321-A",
                "section": "A",
                "days": ["T", "R"],
                "begin": "13:00",
                "end": "16:50",
                "duration_minutes": 230,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48321-B",
                "section": "B",
                "days": ["T", "R"],
                "begin": "13:00",
                "end": "16:50",
                "duration_minutes": 230,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48321-C",
                "section": "C",
                "days": ["T", "R"],
                "begin": "13:00",
                "end": "16:50",
                "duration_minutes": 230,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48321-D",
                "section": "D",
                "days": ["T", "R"],
                "begin": "13:00",
                "end": "16:50",
                "duration_minutes": 230,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48321-E",
                "section": "E",
                "days": ["T", "R"],
                "begin": "13:00",
                "end": "16:50",
                "duration_minutes": 230,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48321-F",
                "section": "F",
                "days": ["T", "R"],
                "begin": "13:00",
                "end": "16:50",
                "duration_minutes": 230,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
        ],
        "minis": ["Full"],
        "is_primary_required": False,
        "requires_time_slot": True,
    },
    {
        "course": "48613",
        "title": "New Pedagogies: Unsettling Ground: Architecture is Dead New Pedagogies Storycraft New Pedagogies Intermundium: Labyrinth of Invisible Narratives 48620 Graduate Seminar: Situating Research 3.0 A F 09:30AM 10:50AM Pittsburgh, Pennsylvania",
        "units": 9.0,
        "primary": None,
        "sections": [
            {
                "section_id": "48613-A",
                "section": "A",
                "days": ["T", "R"],
                "begin": "11:00",
                "end": "12:20",
                "duration_minutes": 80,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48613-D",
                "section": "D",
                "days": ["M", "W"],
                "begin": "11:00",
                "end": "12:20",
                "duration_minutes": 80,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48613-E",
                "section": "E",
                "days": ["W", "F"],
                "begin": "09:30",
                "end": "10:50",
                "duration_minutes": 80,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
        ],
        "minis": ["Full"],
        "is_primary_required": False,
        "requires_time_slot": True,
    },
    {
        "course": "48621",
        "title": "American City: Architectural and Urban Design History: Architectural and Urban Design History 48625 Thesis Seminar 9.0 A MF 12:30PM 01:50PM Pittsburgh, Pennsylvania 48626 Bending Active System_ Bamboo Research Pavilion Using Robotic Arm and Steam Bend VAR A TR 10:00AM 11:50AM Pittsburgh, Pennsylvania 48629 Environment Energy & Climate 9.0 Lec TR 02:00PM 03:20PM Pittsburgh, Pennsylvania A TBA Pittsburgh, Pennsylvania 48630 M.Arch Studio: Praxis 1 18.0 A MWF 02:00PM 04:50PM Pittsburgh, Pennsylvania 48633 Afrofuturism and Othered Ways of Seeing & Being in the World 9.0 A MW 11:00AM 12:20PM Pittsburgh, Pennsylvania 48634 Architectural Theory & Contemporary Issues VAR A MW 09:30AM 10:50AM Pittsburgh, Pennsylvania 48635 Environmental Systems: Climate & Energy in Buildings 9.0 Lec TR 03:30PM 04:50PM Pittsburgh, Pennsylvania 48643 Color Constructs 9.0 A MW 11:00AM 12:20PM Pittsburgh, Pennsylvania 48645 Design Consciousness: Working with Diverse Populations 9.0 A MW 09:00AM 10:20AM Pittsburgh, Pennsylvania 48650 ASO Masters Studio 18.0 A TR 01:00PM 04:50PM Pittsburgh, Pennsylvania 48653 Monopolis for the Masses 9.0 A MW 11:00AM 12:20PM Pittsburgh, Pennsylvania 48654 Aztec to Zacatecas: Mesoamerican & Spanish Colonial Arch of Mexico & Guatemala 9.0 A TR 11:00AM 12:20PM Pittsburgh, Pennsylvania 48655 Environment II: Design Integration of Active Building Systems 9.0 A TR 09:30AM 10:50AM Pittsburgh, Pennsylvania 48657 Infrastructural Landscapes I: Design Histories 3.0 A1 MF 02:00PM 03:20PM Pittsburgh, Pennsylvania 48659 Infrastructural Landscapes II: Design Futures VAR A2 MF 02:00PM 03:20PM Pittsburgh, Pennsylvania 48661 Professional Development 3.0 A1 R 11:00AM 12:20PM Pittsburgh, Pennsylvania 48662 Architectural Intelligence: Workshop and Experiment of Generative AI in Design 3.0 A2 MW 09:30AM 10:50AM Pittsburgh, Pennsylvania 48668 Sankofa Bamboo Greenhouse VAR A1 W 07:00PM 08:20PM Pittsburgh, Pennsylvania 48674 History of Architecture in the Islamic World- A Primer 9.0 A MF 11:00AM 12:20PM Pittsburgh, Pennsylvania 48675 Designing for the Internet of Things 6.0 A2 TR 10:00AM 11:50AM Pittsburgh, Pennsylvania 48689 Digital Skills Workshop 3.0 A1 TBA Pittsburgh, Pennsylvania 48699 Environmental Justice and Decolonial Ecologies 9.0 A F 02:00PM 04:50PM Pittsburgh, Pennsylvania 48700 Practicum 0-36 A TBA Pittsburgh, Pennsylvania 48702 Master's Project 18-36 A TBA Pittsburgh, Pennsylvania 48705 Urban Design Studio I 18.0 A TR 01:00PM 04:50PM Pittsburgh, Pennsylvania 48707 MUD Graduate Seminar 3.0 A W 09:30AM 10:50AM Pittsburgh, Pennsylvania 48716 MSCD Pre-thesis II 6.0 A R 10:00AM 11:50AM Pittsburgh, Pennsylvania 48718 Urban Design Studio III 18.0 A TR 01:00PM 04:50PM Pittsburgh, Pennsylvania 48724 Scripting and Parametric Design 9.0 A TR 06:00PM 07:50PM Pittsburgh, Pennsylvania 48725 Graduate Real Estate Development VAR A TR 08:00AM 09:20AM Pittsburgh, Pennsylvania 48727 Inquiry into Computational Design 9.0 A W 09:00AM 11:50AM Pittsburgh, Pennsylvania 48729 Sustainability, Health and Productivity to Accelerate a Quality Built Environmen 9-12 A TR 09:30AM 10:50AM Pittsburgh, Pennsylvania 48731 Sustainable Design Synthesis Prep 1-18 A F 09:30AM 11:50AM Pittsburgh, Pennsylvania 48732 Sustainable Design Synthesis 0,12,24 A F 09:30AM 11:50AM Pittsburgh, Pennsylvania 48733 Environmental Performance Simulation 0-3 A T 09:30AM 12:20PM Pittsburgh, Pennsylvania 48736 Master's Independent Study 0-99 C TBA Pittsburgh, Pennsylvania E TBA Pittsburgh, Pennsylvania F TBA Pittsburgh, Pennsylvania I TBA Pittsburgh, Pennsylvania J TBA Pittsburgh, Pennsylvania M TBA Pittsburgh, Pennsylvania O TBA Pittsburgh, Pennsylvania R TBA Pittsburgh, Pennsylvania V TBA Pittsburgh, Pennsylvania X TBA Pittsburgh, Pennsylvania 48740 Urban Design Methods and Theory 9.0 A F 09:00AM 11:50AM Pittsburgh, Pennsylvania 48742 Planning and Public Policy for the Future of Urbanism VAR A W 07:00PM 08:20PM Pittsburgh, Pennsylvania 48743 Introduction to Ecological Design Thinking 9.0 A W 10:00AM 12:20PM Pittsburgh, Pennsylvania 48751 Lab for Cybernetics: Engaging Wicked Challenges 12.0 A WF 02:00PM 03:20PM Pittsburgh, Pennsylvania 48753 Intro to Urban Design Media 6.0 A M 10:00AM 11:50AM Pittsburgh, Pennsylvania 48755 Introduction to Architectural Robotics 9.0 A MW 02:00PM 03:50PM Pittsburgh, Pennsylvania 48763 Sustainable Design for Uncertain Futures VAR A MW 11:00AM 12:20PM Pittsburgh, Pennsylvania 48764 BIM and AI For Architects: Empowering Innovative Architectural Design with Revit VAR A2 TR 02:00PM 03:20PM Pittsburgh, Pennsylvania 48765 AECM Synthesis 12.0 A MWF 02:00PM 03:20PM Pittsburgh, Pennsylvania 48767 Transdisciplinary Thinking 12.0 A MWF 03:30PM 04:50PM Pittsburgh, Pennsylvania 48768 Indoor Environmental Quality (IEQ); Energy, Health and Productivity VAR A MW 09:30AM 10:50AM Pittsburgh, Pennsylvania 48769 Thesis/Project VAR A T 01:00PM 03:50PM Pittsburgh, Pennsylvania B TBA Pittsburgh, Pennsylvania C TBA Pittsburgh, Pennsylvania D TBA Pittsburgh, Pennsylvania E TBA Pittsburgh, Pennsylvania F TBA Pittsburgh, Pennsylvania G TBA Pittsburgh, Pennsylvania H TBA Pittsburgh, Pennsylvania I TBA Pittsburgh, Pennsylvania J TBA Pittsburgh, Pennsylvania K TBA Pittsburgh, Pennsylvania L TBA Pittsburgh, Pennsylvania M TBA Pittsburgh, Pennsylvania N TBA Pittsburgh, Pennsylvania O TBA Pittsburgh, Pennsylvania P TBA Pittsburgh, Pennsylvania Q TBA Pittsburgh, Pennsylvania R TBA Pittsburgh, Pennsylvania S TBA Pittsburgh, Pennsylvania T TBA Pittsburgh, Pennsylvania U TBA Pittsburgh, Pennsylvania 48771 Fabricating Customization: Prototype 9-12 A MF 09:30AM 10:50AM Pittsburgh, Pennsylvania 48772 MAAD Advanced Synthesis Options Studio I 18.0 B TR 01:00PM 04:50PM Pittsburgh, Pennsylvania",
        "units": 9.0,
        "primary": None,
        "sections": [
            {
                "section_id": "48621-A",
                "section": "A",
                "days": ["T", "R"],
                "begin": "15:30",
                "end": "16:50",
                "duration_minutes": 80,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48621-B",
                "section": "B",
                "days": ["W"],
                "begin": "18:30",
                "end": "19:50",
                "duration_minutes": 80,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48621-B",
                "section": "B",
                "days": ["T", "R"],
                "begin": "13:00",
                "end": "16:50",
                "duration_minutes": 230,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48621-C",
                "section": "C",
                "days": ["T", "R"],
                "begin": "13:00",
                "end": "16:50",
                "duration_minutes": 230,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48621-D",
                "section": "D",
                "days": ["T", "R"],
                "begin": "13:00",
                "end": "16:50",
                "duration_minutes": 230,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48621-E",
                "section": "E",
                "days": ["T", "R"],
                "begin": "13:00",
                "end": "16:50",
                "duration_minutes": 230,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48621-F",
                "section": "F",
                "days": ["T", "R"],
                "begin": "13:00",
                "end": "16:50",
                "duration_minutes": 230,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48621-G",
                "section": "G",
                "days": ["T", "R"],
                "begin": "13:00",
                "end": "16:50",
                "duration_minutes": 230,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48621-H",
                "section": "H",
                "days": ["T", "R"],
                "begin": "13:00",
                "end": "16:50",
                "duration_minutes": 230,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48621-B2",
                "section": "B2",
                "days": ["W"],
                "begin": "19:00",
                "end": "20:20",
                "duration_minutes": 80,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48621-C",
                "section": "C",
                "days": ["T", "R"],
                "begin": "13:00",
                "end": "16:50",
                "duration_minutes": 230,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48621-D",
                "section": "D",
                "days": ["T", "R"],
                "begin": "13:00",
                "end": "16:50",
                "duration_minutes": 230,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
            {
                "section_id": "48621-E",
                "section": "E",
                "days": ["T", "R"],
                "begin": "13:00",
                "end": "16:50",
                "duration_minutes": 230,
                "meeting_type": "lab",
                "requires_time_slot": True,
            },
        ],
        "minis": ["Full"],
        "is_primary_required": False,
        "requires_time_slot": True,
    },
]

# ---------------------------------------------------------------------------
# Helpers for working with meeting times and overlaps
# ---------------------------------------------------------------------------


def time_to_minutes(time_str: str) -> int:
    """Convert a HH:MM string to minutes after midnight."""

    hours, minutes = map(int, time_str.split(":"))
    return hours * 60 + minutes


def overlaps(section_a: Dict, section_b: Dict) -> bool:
    """Return True when any day/time interval overlaps."""

    days_a = set(section_a.get("days", []))
    days_b = set(section_b.get("days", []))
    common_days = days_a.intersection(days_b)
    if not common_days:
        return False

    start_a, end_a = time_to_minutes(section_a["begin"]), time_to_minutes(section_a["end"])
    start_b, end_b = time_to_minutes(section_b["begin"]), time_to_minutes(section_b["end"])
    return any(not (end_a <= start_b or end_b <= start_a) for _ in common_days)


# ---------------------------------------------------------------------------
# QUBO builder
# ---------------------------------------------------------------------------

def build_qubo(
    courses: Sequence[Dict],
    course_penalty: float = 5.0,
    conflict_penalty: float = 5.0,
) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float], float, Dict[int, str]]:
    """Create the QUBO dictionaries and offset.

    Variables: one binary variable per course-section option. Names are
    ``{course_code}|{section_id}|{idx}`` so duplicates are unique.

    Constraints encoded:
    * Exactly one section per course: (sum_i x_i - 1)^2 with weight course_penalty.
    * No time conflicts: conflict_penalty * x_i * x_j for every overlapping pair.
    """

    linear: Dict[str, float] = {}
    quadratic: Dict[Tuple[str, str], float] = {}
    offset = 0.0
    index_to_var: Dict[int, str] = {}

    # Variable creation
    for course_idx, course in enumerate(courses):
        for section_idx, section in enumerate(course.get("sections", [])):
            var_name = f"{course['course']}|{section['section_id']}|{section_idx}"
            index_to_var[len(index_to_var)] = var_name
            linear.setdefault(var_name, 0.0)

    # Course selection constraints
    for course in courses:
        vars_for_course = [
            f"{course['course']}|{section['section_id']}|{idx}"
            for idx, section in enumerate(course.get("sections", []))
        ]
        for var in vars_for_course:
            linear[var] += -1.0 * course_penalty
        offset += course_penalty
        for var_a, var_b in itertools.combinations(vars_for_course, 2):
            key = tuple(sorted((var_a, var_b)))
            quadratic[key] = quadratic.get(key, 0.0) + 2.0 * course_penalty

    # Conflict constraints across all variables
    all_sections: List[Tuple[str, Dict]] = []
    for course in courses:
        for idx, section in enumerate(course.get("sections", [])):
            var = f"{course['course']}|{section['section_id']}|{idx}"
            all_sections.append((var, section))

    for (var_a, sec_a), (var_b, sec_b) in itertools.combinations(all_sections, 2):
        if overlaps(sec_a, sec_b):
            key = tuple(sorted((var_a, var_b)))
            quadratic[key] = quadratic.get(key, 0.0) + conflict_penalty

    return linear, quadratic, offset, index_to_var


# ---------------------------------------------------------------------------
# Convert QUBO -> Ising -> PennyLane Hamiltonian
# ---------------------------------------------------------------------------

def qubo_to_ising(
    linear: Dict[str, float], quadratic: Dict[Tuple[str, str], float], offset: float
) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float], float]:
    """Map QUBO coefficients to the equivalent Ising form."""

    h = {var: 0.0 for var in linear}
    J: Dict[Tuple[str, str], float] = {}
    constant = offset

    for var, coeff in linear.items():
        constant += coeff / 2.0
        h[var] += -coeff / 2.0

    for (var_a, var_b), coeff in quadratic.items():
        constant += coeff / 4.0
        h[var_a] += -coeff / 4.0
        h[var_b] += -coeff / 4.0
        key = tuple(sorted((var_a, var_b)))
        J[key] = J.get(key, 0.0) + coeff / 4.0

    return h, J, constant


def ising_to_pennylane_hamiltonian(
    h: Dict[str, float],
    J: Dict[Tuple[str, str], float],
    constant: float,
    index_to_var: Dict[int, str],
) -> qml.Hamiltonian:
    """Create a PennyLane Hamiltonian using PauliZ and Identity operators."""

    # Map variable names to wire indices (0..n-1)
    var_to_index = {var: idx for idx, var in index_to_var.items()}

    coeffs: List[float] = []
    ops: List[qml.operation.Operator] = []

    for var, coeff in h.items():
        if abs(coeff) > 1e-9:
            coeffs.append(coeff)
            ops.append(qml.PauliZ(var_to_index[var]))

    for (var_a, var_b), coeff in J.items():
        if abs(coeff) > 1e-9:
            coeffs.append(coeff)
            ops.append(qml.PauliZ(var_to_index[var_a]) @ qml.PauliZ(var_to_index[var_b]))

    if abs(constant) > 1e-9:
        coeffs.append(constant)
        ops.append(qml.Identity(0))

    return qml.Hamiltonian(coeffs, ops)


# ---------------------------------------------------------------------------
# QAOA routines
# ---------------------------------------------------------------------------

def run_local_qaoa(
    cost_h: qml.Hamiltonian,
    num_layers: int = 1,
    steps: int = 75,
    stepsize: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Optimize QAOA angles against the cost Hamiltonian on a local simulator."""

    num_wires = len(cost_h.wires)
    wires = range(num_wires)
    dev = qml.device("default.qubit", wires=num_wires)

    def circuit(gammas, betas):
        for wire in wires:
            qml.Hadamard(wires=wire)
        for layer in range(num_layers):
            qml.qaoa.cost_layer(gammas[layer], cost_h)
            qml.qaoa.mixer_layer(betas[layer], wires=wires)
        return qml.expval(cost_h)

    qnode = qml.QNode(circuit, dev, interface="autograd")

    opt = qml.GradientDescentOptimizer(stepsize)
    gammas = np.random.uniform(0, np.pi, num_layers, requires_grad=True)
    betas = np.random.uniform(0, np.pi, num_layers, requires_grad=True)

    for _ in range(steps):
        (gammas, betas), cost = opt.step_and_cost(qnode, gammas, betas)

    best_cost = qnode(gammas, betas)
    return np.array(gammas), np.array(betas), float(best_cost)


def sample_bitstrings(
    cost_h: qml.Hamiltonian,
    gammas: Sequence[float],
    betas: Sequence[float],
    shots: int = 2048,
) -> np.ndarray:
    """Sample bitstrings from the optimized QAOA state."""

    num_wires = len(cost_h.wires)
    wires = range(num_wires)
    dev = qml.device("default.qubit", wires=num_wires, shots=shots)

    @qml.qnode(dev)
    def circuit():
        for wire in wires:
            qml.Hadamard(wires=wire)
        for layer in range(len(gammas)):
            qml.qaoa.cost_layer(gammas[layer], cost_h)
            qml.qaoa.mixer_layer(betas[layer], wires=wires)
        return qml.sample(wires=wires)

    samples = circuit()
    return samples


# ---------------------------------------------------------------------------
# Energy evaluation and post-processing
# ---------------------------------------------------------------------------

def evaluate_qubo(
    bitstring: Iterable[int],
    linear: Dict[str, float],
    quadratic: Dict[Tuple[str, str], float],
    offset: float,
    index_to_var: Dict[int, str],
) -> float:
    """Compute QUBO energy for a bitstring."""

    energy = offset
    bits = list(bitstring)
    var_to_index = {var: idx for idx, var in index_to_var.items()}

    for idx, bit in enumerate(bits):
        var = index_to_var[idx]
        energy += linear.get(var, 0.0) * bit

    for (var_a, var_b), coeff in quadratic.items():
        idx_a = var_to_index[var_a]
        idx_b = var_to_index[var_b]
        energy += coeff * bits[idx_a] * bits[idx_b]

    return energy


def best_sampled_solution(
    samples: np.ndarray,
    linear: Dict[str, float],
    quadratic: Dict[Tuple[str, str], float],
    offset: float,
    index_to_var: Dict[int, str],
) -> Tuple[List[int], float]:
    """Pick the lowest-energy sample seen."""

    energy_by_sample = {}
    for bitstring in samples:
        key = tuple(int(b) for b in bitstring)
        if key not in energy_by_sample:
            energy_by_sample[key] = evaluate_qubo(key, linear, quadratic, offset, index_to_var)

    best_bits, best_energy = min(energy_by_sample.items(), key=lambda item: item[1])
    return list(best_bits), best_energy


def interpret_selection(bits: Sequence[int], index_to_var: Dict[int, str]) -> Dict[str, str]:
    """Map a bitstring back to selected course sections."""

    chosen: Dict[str, str] = {}
    for idx, bit in enumerate(bits):
        if bit:
            course_code, section_id, _ = index_to_var[idx].split("|")
            chosen[course_code] = section_id
    return chosen


# ---------------------------------------------------------------------------
# Optional integrations
# ---------------------------------------------------------------------------

def build_braket_device(num_wires: int, shots: int = 1000):
    """Return a PennyLane Braket device (requires AWS credentials).

    Replace ``device_arn`` with a managed simulator or a hardware ARN from
    https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html.
    """

    return qml.device(
        "braket.aws.qubit",
        device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
        wires=num_wires,
        shots=shots,
    )


def qiskit_qubo_program(linear: Dict[str, float], quadratic: Dict[Tuple[str, str], float]):
    """Create a Qiskit QuadraticProgram mirroring the QUBO.

    Running QAOA then requires qiskit-terra + qiskit-algorithms:

        from qiskit_algorithms import QAOA
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit.primitives import Estimator

        qp = qiskit_qubo_program(linear, quadratic)
        qaoa = QAOA(Estimator(), reps=1, optimizer=COBYLA())
        result = qaoa.compute_minimum_eigenvalue(qp.to_ising())
    """

    from qiskit.optimization import QuadraticProgram

    qp = QuadraticProgram()
    for var in linear:
        qp.binary_var(var)

    qp.minimize(linear=linear, quadratic=quadratic)
    return qp


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def main():
    courses = EXAMPLE_COURSES
    linear, quadratic, offset, index_to_var = build_qubo(courses)
    h, J, constant = qubo_to_ising(linear, quadratic, offset)
    cost_h = ising_to_pennylane_hamiltonian(h, J, constant, index_to_var)

    print("Variables (index -> course|section|local_id):")
    for idx, var in index_to_var.items():
        print(f"  {idx}: {var}")

    print("\nRunning QAOA locally (depth p=1)...")
    gammas, betas, expectation = run_local_qaoa(cost_h, num_layers=1, steps=60)
    print(f"Optimized expectation value: {expectation:.3f}")

    samples = sample_bitstrings(cost_h, gammas, betas, shots=1024)
    bits, best_energy = best_sampled_solution(samples, linear, quadratic, offset, index_to_var)
    selection = interpret_selection(bits, index_to_var)

    print("Best sampled assignment (bitstring -> course section):")
    print(bits)
    for course, section in selection.items():
        print(f"  Course {course} -> Section {section}")
    print(f"Sampled QUBO energy: {best_energy:.3f}")

    print("\nTo run on Braket hardware, swap the device:")
    print("  dev = build_braket_device(num_wires=len(cost_h.wires), shots=1000)")
    print("  qnode = qml.QNode(circuit, dev)")


if __name__ == "__main__":
    main()
