# truth_check.py
# SBVS 2.0 - checks a hand-labeled ground-truth CSV before it is used (owner: Jessica)
#
# A person watches a clip and writes data/ground_truth/<session_id>.csv
# (see docs/LABELING_GUIDE.md). Typing mistakes in that file would make the
# accuracy numbers wrong, so check_truth() looks for them first.
#
#   check_truth(path) -> list of problem messages (an empty list means the file is OK)
#
# Use it in a notebook or Python shell started in the project folder:
#   import truth_check
#   truth_check.print_truth_report('data/ground_truth/2026-11-04_practice_1.csv')

import re
import pandas as pd
import settings


def check_truth(path):
    """Return a list of problems found in one ground-truth CSV file."""
    problems = []

    # Read every cell as text (dtype=str) so we can check how it was typed.
    # keep_default_na=False keeps blank cells as '' instead of turning them into NaN.
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except FileNotFoundError:
        problems.append(f"File not found: {path}")
        return problems

    # 1. The header row must be exactly the agreed columns, in order
    if list(df.columns) != settings.TRUTH_COLUMNS:
        problems.append(f"Wrong columns {list(df.columns)}; expected {settings.TRUTH_COLUMNS}")
        return problems   # the other checks need the right columns

    if len(df) == 0:
        problems.append("No shots labeled yet (the file only has the header row)")
        return problems

    previous_time = -1.0
    for i in range(len(df)):
        row = df.iloc[i]
        line = i + 2   # line number in the CSV file (line 1 is the header)
        shot_no = row['shot_no'].strip()
        time_text = row['time_s'].strip()
        result = row['result']

        # 2. shot_no must be a whole number, counting 1, 2, 3, ...
        if not re.search(r'^\d+$', shot_no):
            problems.append(f"Line {line}: shot_no '{shot_no}' is not a whole number")
        elif int(shot_no) != i + 1:
            problems.append(f"Line {line}: shot_no is {shot_no}, expected {i + 1}")

        # 3. time_s must not be blank and must be a number like 15 or 15.2
        if time_text == '':
            problems.append(f"Line {line}: time_s is blank")
        elif not re.search(r'^\d+(\.\d+)?$', time_text):
            problems.append(f"Line {line}: time_s '{time_text}' is not a number of seconds")
        else:
            # 4. Times must go up: each shot happens after the one before it
            time_s = float(time_text)
            if time_s <= previous_time:
                problems.append(f"Line {line}: time_s {time_s} is not after the previous shot ({previous_time})")
            previous_time = time_s

        # 5. result must be exactly 'made' or 'missed' (lowercase, no spaces)
        if result != 'made' and result != 'missed':
            problems.append(f"Line {line}: result '{result}' must be 'made' or 'missed'")

    return problems


def print_truth_report(path):
    """Print the problems for one file in a readable way."""
    problems = check_truth(path)
    if len(problems) == 0:
        print(f"OK: {path}")
    else:
        print(f"{len(problems)} problem(s) in {path}:")
        for problem in problems:
            print("  - " + problem)

