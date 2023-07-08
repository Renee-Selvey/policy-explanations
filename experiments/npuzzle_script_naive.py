import subprocess
import os.path as osp

THIS_DIR = osp.dirname(osp.abspath(__file__))
PROBLEM_PATH = osp.abspath(osp.join(THIS_DIR, "experiment_problems/gripper"))
EXP_PATH = osp.join(THIS_DIR, "../compute_explanation_naive.py")
LOG_PATH = osp.abspath(osp.join(THIS_DIR, "../results/gripper/"))

weights = osp.join(PROBLEM_PATH, "gripper-sparse-weights.pkl")
domain = osp.join(PROBLEM_PATH, "domain.pddl")
problems = [
    osp.join(PROBLEM_PATH, "problem01.pddl"),
    osp.join(PROBLEM_PATH, "problem02.pddl"),
    osp.join(PROBLEM_PATH, "problem03.pddl"),
    osp.join(PROBLEM_PATH, "problem04.pddl"),
    osp.join(PROBLEM_PATH, "problem05.pddl"),
    osp.join(PROBLEM_PATH, "problem06.pddl"),
    osp.join(PROBLEM_PATH, "problem07.pddl"),
    osp.join(PROBLEM_PATH, "problem08.pddl"),
    osp.join(PROBLEM_PATH, "problem09.pddl"),
    osp.join(PROBLEM_PATH, "problem10.pddl"),
    osp.join(PROBLEM_PATH, "problem11.pddl"),
    osp.join(PROBLEM_PATH, "problem12.pddl"),
    osp.join(PROBLEM_PATH, "problem13.pddl"),
    osp.join(PROBLEM_PATH, "problem14.pddl"),
    osp.join(PROBLEM_PATH, "problem15.pddl"),
    osp.join(PROBLEM_PATH, "problem16.pddl"),
    osp.join(PROBLEM_PATH, "problem17.pddl"),
    osp.join(PROBLEM_PATH, "problem18.pddl"),
    osp.join(PROBLEM_PATH, "problem19.pddl"),
    osp.join(PROBLEM_PATH, "problem20.pddl"),
    osp.join(PROBLEM_PATH, "problem25.pddl"),
    osp.join(PROBLEM_PATH, "problem30.pddl"),
    osp.join(PROBLEM_PATH, "problem35.pddl"),
    osp.join(PROBLEM_PATH, "problem40.pddl"),
    osp.join(PROBLEM_PATH, "problem45.pddl"),
    osp.join(PROBLEM_PATH, "problem50.pddl"),
    osp.join(PROBLEM_PATH, "problem55.pddl"),
    osp.join(PROBLEM_PATH, "problem60.pddl"),
    osp.join(PROBLEM_PATH, "problem65.pddl"),
    osp.join(PROBLEM_PATH, "problem70.pddl"),
    osp.join(PROBLEM_PATH, "problem75.pddl"),
    osp.join(PROBLEM_PATH, "problem80.pddl"),
    osp.join(PROBLEM_PATH, "problem85.pddl"),
    osp.join(PROBLEM_PATH, "problem90.pddl"),
    osp.join(PROBLEM_PATH, "problem95.pddl"),
]

for prob in problems:
    try:
        subprocess.run(
            [
                "python",
                EXP_PATH,
                weights,
                domain,
                prob,
                "--type",
                "sub_min_exp",
                "--path",
                LOG_PATH,
            ],
            timeout = 100000
        )
        
    except subprocess.TimeoutExpired:
        raise TimeoutError
        print("timed out")