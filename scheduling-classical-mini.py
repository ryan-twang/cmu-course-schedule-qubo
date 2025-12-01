#!/usr/bin/env python
#
# scheduling_classical_v2.py
#
# Classical Baseline Solver for Quantum Optimization Project (QUBO Phase 1)
# Uses clean CSV data to build and solve a feasibility model in Gurobi.
#

# Mini Version

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import random
from datetime import datetime
import math
from typing import List, Dict, Set, Tuple, Any


# --- 1. CONFIGURATION ---

# NOTE: The course list is already filtered to ECE ('18') in the CSV.
LOCAL_SCHEDULE_FILEPATH = 'F24-Schedule-ECE.csv' 
MAX_ROOMS_ALLOWED = 2
MAX_NUM_COURSES = 3
TIME_SLOT_HOURS = range(8, 11) # Use 8 AM to 10 PM (22:00) for time slots

# --- 2. DATA LOADING AND FEATURE CONSTRUCTION ---

def build_time_slots(time_hours: range) -> List[str]:
    """Generates a list of discrete hourly time slots (e.g., 'Mon_09:00')."""
    # days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    days = ["Mon"]
    hours = [f"{h:02d}:00" for h in time_hours]
    return [f"{day}_{hour}" for day in days for hour in hours]

def calculate_duration_slots(begin_str: str, end_str: str) -> int:
    """
    Calculates duration in hours, rounding up greedily.
    Example: 8:00AM to 9:20AM = 1h 20m -> Rounds up to 2 slots.
    """
    if pd.isna(begin_str) or pd.isna(end_str) or begin_str == 'TBA':
        return 1 # Default to 1 hour if unknown
    
    fmt = "%I:%M%p" # Format: 08:00AM
    try:
        t_start = datetime.strptime(begin_str, fmt)
        t_end = datetime.strptime(end_str, fmt)
        
        # Calculate difference in minutes
        diff_minutes = (t_end - t_start).total_seconds() / 60
        
        # Greedily round up to nearest 60 minutes
        # 1 min to 60 min = 1 slot; 61 min = 2 slots
        slots = math.ceil(diff_minutes / 60)
        return max(1, slots) # Ensure at least 1 slot
    except Exception:
        return 1

def load_and_map_features(filepath: str) -> Tuple[List[str], List[str], List[str], Dict[str, List[str]], List[str], Dict[str, int], Dict[str, int]]:
    """Loads CSV, cleans data, and generates feature sets for the optimization model."""
    print(f"[Data] Loading and mapping data from {filepath}...")
    
    # Use Pandas to load the CSV file
    df = pd.read_csv(filepath)
    df.columns = [col.strip() for col in df.columns]

    # Drop rows that are TBA or Cancelled
    df = df[df['DAYS'] != 'TBA']
    df = df[df['INSTRUCTOR'].str.lower() != 'tba']
    df = df[~df['COURSE TITLE'].str.contains('Cancelled', case=False)]

    # MINI: Only take first 20 courses
    df_mini = df[:MAX_NUM_COURSES].copy()
    print(f"[DEBUG] Truncated dataset to {len(df)} sections for testing.")
    
    # Use the combination of Course Number and Section ID as the unique Meeting ID
    df_mini['MeetingID'] = df_mini['COURSE'].astype(str) + '-' + df_mini['SEC']

    # Handle multiple instructors in the 'INSTRUCTOR' column (e.g., "Last, First; Other, Name")
    df_mini['INSTRUCTORS'] = df_mini['INSTRUCTOR'].str.split(';').apply(lambda x: [i.strip() for i in x])
    
    # 1. Calculate Duration (The Greedy Rounding)
    # Instead of mapping to specific times (e.g. "Mon_09:00"), we calculate 
    # how many integer slots are needed (e.g. 80mins -> 2 slots).
    # Uses helper function 'calculate_duration_slots' above
    df_mini['DurationSlots'] = df_mini.apply(
        lambda row: calculate_duration_slots(row['BEGIN'], row['END']), axis=1
    )

    # 2. Create a clean "Meetings" dataframe
    # We drop duplicates so each Section (MeetingID) appears exactly once.
    # This creates the "master list" of what needs to be scheduled.
    initial_count = len(df_mini)
    meeting_df = df_mini.drop_duplicates(subset=['MeetingID'])
    dropped_count = initial_count - len(meeting_df)
    print(f"\n[Data] Dropped {dropped_count} duplicate rows.")

    # 3. Define primary scheduling dimensions
    all_meetings = meeting_df['MeetingID'].tolist()
    
    # Extract unique rooms and instructors directly from the raw data
    # (No need to explode by time slots anymore to get this list)
    all_rooms_raw = df_mini['BLDG/ROOM'].unique().tolist()
    all_instructors = df_mini['INSTRUCTORS'].explode().unique().tolist()

    # 4. Create the Duration Map
    # This is the key new dictionary the Gurobi solver needs to check conflicts.
    meeting_duration_map = meeting_df.set_index('MeetingID')['DurationSlots'].to_dict()
    
    # TODO: REMOVE / ADJUST --- Enforce License Limit ---
    limited_room_list = all_rooms_raw[:MAX_ROOMS_ALLOWED]
    print(f"[Map] Original unique rooms found: {len(all_rooms_raw)}. Using: {len(limited_room_list)} rooms.")
    
    # --- Final Mappings ---
    timeslots_list = build_time_slots(TIME_SLOT_HOURS)

    # Create mapping of meeting ID to list of instructors
    meeting_instructors_map = df_mini.set_index('MeetingID')['INSTRUCTORS'].to_dict()

    # Create map of MeetingID to required size (placeholder for now)
    # meeting_enrollment_map = {
    #     meeting: random.randint(15, 80) for meeting in all_meetings
    # }

    # ANS B10: Small (30) - Can only hold 18095-A or 18095-B
    # HH A104: Large (50) - The only room that can hold 18021-A1
    meeting_enrollment_map = {'18021-A1': 28, '18095-A': 20, '18100-Lec1': 45}

    # Create map of Room to capacity (placeholder for now)
    # room_capacity_map = {
        
    #     room: random.randint(30, 150) for room in limited_room_list
    # }

    room_capacity_map = {'ANS B10': 30, 'HH A104': 50}

    
    print(f"[Map] Total discrete time slots: {len(timeslots_list)}")
    print(f"[Map] Total unique meetings (sections/lectures) to schedule: {len(all_meetings)}")
    
    return all_meetings, limited_room_list, all_instructors, meeting_instructors_map, timeslots_list, meeting_enrollment_map, room_capacity_map, meeting_duration_map


# --- 3. GUROBI MODEL (Classical Baseline Solver) ---

def build_and_run_model(
    meetings: List[str],
    rooms: List[str],
    timeslots: List[str],
    all_instructors: List[str],
    meeting_instructors: Dict[str, List[str]],
    meeting_enrollment: Dict[str, int],
    room_capacity: Dict[str, int],
    meeting_duration: Dict[str, int]
):
    """
    Builds and solves the Gurobi optimization model.
    """
    try:
        m = gp.Model("course_scheduling_classical_baseline")

        # --- Variables Calculation (DEBUGGING) ---
        num_meetings = len(meetings)
        num_rooms = len(rooms)
        num_timeslots = len(timeslots)
        total_variables = num_meetings * num_rooms * num_timeslots
        print(f"\n[DEBUG] Model Inputs: Meetings={num_meetings}, Rooms={num_rooms}, TimeSlots={num_timeslots}")
        print(f"[DEBUG] Total Binary Variables: {total_variables:,}")

        # days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        days = ["Mon"]
        slots_per_day = {day: [t for t in timeslots if t.startswith(day)] for day in days}
        
        # A typical Gurobi commercial license limit for unrestricted academic use is 2000 variables.
        # However, restricted (trial/limited academic) licenses may be much lower (e.g., 500-1000).
        # We must ensure this number is acceptable to Gurobi's license.

        x = {}
        objective_expr = gp.LinExpr()
        count_vars = 0

        # RESOURCE MINIMIZATION SETUP:
        # We designate the second half of the room list as "Overflow/Expensive" rooms.
        # This encourages the solver to PACK the first half of rooms before touching the rest.
        num_rooms = len(rooms)
        overflow_start_index = int(num_rooms * 0.5) 
        sorted_rooms = sorted(rooms, key=lambda r: room_capacity[r]) # Sort small to large
        
        print("[Gurobi] Generating variables with Space Optimization weights...")
        
        for meeting in meetings:
            duration = meeting_duration[meeting]
            enrollment = meeting_enrollment[meeting]

            for r_idx, room in enumerate(sorted_rooms):
                capacity = room_capacity[room]
                
                # 1. Hard Constraint Filter: Class must fit in room
                if capacity < enrollment:
                    continue
                    
                # --- CALCULATE SOFT CONSTRAINT COSTS ---
                
                # Cost A: Wasted Space (The "Snug Fit" Factor)
                # If you put 10 students in a 100-seat room, cost is 90. 
                wasted_space_cost = (capacity - enrollment)
                
                # Cost B: Resource Minimization (The "Overflow" Factor)
                # If this room is in the 'overflow' half of the list, add a huge penalty
                activation_cost = 0
                if r_idx >= overflow_start_index:
                    activation_cost = 1000 # High penalty to discourage using these rooms
                
                # Total Preference Weight for this specific assignment
                total_weight = wasted_space_cost + activation_cost
                
                for day in days:
                    day_slots = slots_per_day[day]
                    # A meeting with duration D can only start in the first (N - D + 1) slots of the day
                    valid_start_indices = len(day_slots) - duration + 1
                    
                    for i in range(valid_start_indices):
                        t_start = day_slots[i]

                        # Add Variable
                        var = m.addVar(vtype=GRB.BINARY, name=f"x_{meeting}_{t_start}_{room}")
                        x[meeting, t_start, room] = var
                        count_vars += 1

                        # Add to Objective Function
                        # We want to MINIMIZE the total wasted space + overflow usage
                        objective_expr.add(var, total_weight)
                        
        print(f"[DEBUG] Total Binary Variables Created: {count_vars:,}")

        # --- Define Variables ---
        # x[m, t, r] = 1 if meeting m is at time t in room r
        # x = m.addVars(meetings, timeslots, rooms, vtype=GRB.BINARY, name="x")

        # --- C1: Schedule Each Meeting Exactly Once ---
        print("[Gurobi] Adding C1: Schedule each meeting (section/lecture) exactly once.")
        # m.addConstrs(
        #     (x.sum(m, '*', '*') == 1 for m in meetings), 
        #     name="C1_ScheduleOnce"
        # )
        for meeting in meetings:
            # Sum over all valid t, r created for this meeting
            relevant_vars = [x[meeting, t, r] for t in timeslots for r in rooms if (meeting, t, r) in x]
            if not relevant_vars:
                print(f"Warning: Meeting {meeting} (Dur: {meeting_duration[meeting]}) has no valid placement options!")
                continue
            m.addConstr(gp.quicksum(relevant_vars) == 1, name=f"C1_{meeting}")

        def get_active_vars_at_time(target_t, target_r, meeting_subset=None):
            active_vars = []
            target_day = target_t.split('_')[0]
            day_slots = slots_per_day[target_day]
            
            try:
                t_index = day_slots.index(target_t)
            except ValueError:
                return [] # Should not happen

            candidate_meetings = meeting_subset if meeting_subset else meetings

            for meet in candidate_meetings:
                dur = meeting_duration[meet]
                # A meeting covers 'target_t' if it started at 'start_t'
                # such that: start_index <= t_index < start_index + dur
                # Rewritten: t_index - dur < start_index <= t_index
                
                min_start_index = t_index - dur + 1
                max_start_index = t_index
                
                for idx in range(max(0, min_start_index), max_start_index + 1):
                    start_t = day_slots[idx]
                    if (meet, start_t, target_r) in x:
                        active_vars.append(x[meet, start_t, target_r])
            return active_vars
        
        # --- C2: No Two Meetings in the Same Room at the Same Time ---
        print("[Gurobi] Adding C2: Max one meeting per room per time slot.")
        # m.addConstrs(
        #     (x.sum('*', t, r) <= 1 for t in timeslots for r in rooms), 
        #     name="C2_RoomConflict"
        # )
        for r in rooms:
            for t in timeslots:
                # Sum of all meetings that occupy room r at time t must be <= 1
                # This includes meetings that started at t, t-1, t-2 etc. depending on their duration
                active_vars = get_active_vars_at_time(t, r, meetings)
                if active_vars:
                    m.addConstr(gp.quicksum(active_vars) <= 1, name=f"C2_Room_{r}_{t}")        

        # --- C3: Room Capacity must not be exceeded ---
        # Now redundant because we enforce room capacity before running solver

        # --- C4: No Instructor Conflict ---
        print("[Gurobi] Adding C4: No instructor conflicts.")
        for instructor in all_instructors:
            # Find all meetings taught by this instructor
            meetings_by_instructor = [
                m for m in meetings if instructor in meeting_instructors.get(m, [])
            ]
            if not meetings_by_instructor:
                print("error line 309 - instructors")
                continue
            
            for t in timeslots:
                # Instructor cannot be in ANY room at time t
                # We need to sum over all rooms for the instructor's meetings active at t
                expr = gp.LinExpr()
                for r in rooms:
                    active_vars = get_active_vars_at_time(t, r, meetings_by_instructor)
                    for v in active_vars:
                        expr.add(v)
                
                m.addConstr(expr <= 1, name=f"C4_Inst_{instructor}_{t}")
        
        # --- Objective: Feasibility (Find any valid schedule) ---
        # m.setObjective(0, GRB.MINIMIZE)

        # --- Objective: Minimize Wasted Space & Resource Usage ---
        m.setObjective(objective_expr, GRB.MINIMIZE)

        print("\n[Gurobi] Starting optimization (Optimizing for Room Usage)...")
        m.optimize()

        if m.Status == GRB.OPTIMAL:
            print(f"\n--- ✅ Optimal Schedule Found! ---")
            print(f"Total 'Inefficiency' Score: {m.ObjVal}")
            print("(Lower score = tighter packing and less wasted seats)")
            print_schedule(m, x, meeting_duration, meeting_instructors, meeting_enrollment, room_capacity)

        print("\n[Gurobi] Starting solver...")
        m.Params.OutputFlag = 1
        m.optimize()

        print_schedule(m, x, meeting_duration, meeting_instructors, meeting_enrollment, room_capacity)

    except gp.GurobiError as e:
        print(f"\n--- ❌ Gurobi Error ---")
        print(f"Error code {e.errno}: {e}")
        if m.Status == GRB.INFEASIBLE:
            print("\n--- ❌ Model is Infeasible ---")
            print("The model is too constrained (e.g., not enough rooms/time slots, or too many large classes for small rooms).")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def print_schedule(model, x_vars, duration_map, meeting_instructors, meeting_enrollment, room_capacity):
    if model.Status == GRB.OPTIMAL:
        print("\n--- ✅ Schedule Generated ---")
        schedule = {}
        
        # Iterate through vars to find assignments
        for (m, t, r), v in x_vars.items():
            if v.X > 0.5:
                # Calculate all slots this meeting covers
                dur = duration_map[m]
                day, hour_str = t.split('_')
                hour = int(hour_str.split(':')[0])

                # Get stats for the printout
                cap = room_capacity[r]
                enr = meeting_enrollment[m]
                instr = meeting_instructors[m][0] if meeting_instructors[m] else "Staff"
                
                # Mark every covered slot in the schedule
                for i in range(dur):
                    current_hour = hour + i
                    slot_key = f"{day}_{current_hour:02d}:00"
                    
                    if slot_key not in schedule: schedule[slot_key] = []
                    
                    note = "(Cont.)" if i > 0 else "(Start)"
                    entry = f"{note} {m} [{dur}hr] | {instr} | {r} (room cap: {cap}, enrollment: {enr})"
                    schedule[slot_key].append(entry)

        # Sort and Print
        day_order = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4}
        sorted_times = sorted(schedule.keys(), key=lambda x: (day_order.get(x.split('_')[0], 99), x))
        
        for t in sorted_times:
            print(f"\n{t}:")
            for entry in schedule[t]:
                print(f"  {entry}")
    else:
        print("No feasible solution found.")

# --- 4. MAIN EXECUTION ---

if __name__ == "__main__":
    
    (all_meetings, room_list, all_instructors, meeting_instructors_map, 
     timeslot_list, enrollment_map, capacity_map, duration_map) = load_and_map_features(LOCAL_SCHEDULE_FILEPATH)
    
    if not all_meetings:
        print("\nExiting. No suitable meetings found after filtering.")
        exit()

    # Build and Solve
    build_and_run_model(
        all_meetings,
        room_list,
        timeslot_list,
        all_instructors,
        meeting_instructors_map,
        enrollment_map,
        capacity_map,
        duration_map
    )