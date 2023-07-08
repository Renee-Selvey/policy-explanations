import subprocess
import os.path as osp

THIS_DIR = osp.dirname(osp.abspath(__file__))
PROBLEM_PATH = osp.abspath(osp.join(THIS_DIR, "experiment_problems/parking"))
EXP_PATH = osp.join(THIS_DIR, "../compute_explanation.py")
LOG_PATH = osp.abspath(osp.join(THIS_DIR, "../results/parking/"))

weights = osp.join(PROBLEM_PATH, "parking-sparse-weights.pkl")
domain = osp.join(PROBLEM_PATH, "parking-typed.pddl")
problems = [
    osp.join(PROBLEM_PATH, "parking-2-1.pddl"),
    osp.join(PROBLEM_PATH, "parking-2-2-1.pddl"),
    osp.join(PROBLEM_PATH, "parking-2-2.pddl"),
    osp.join(PROBLEM_PATH, "parking-3-1.pddl"),
    osp.join(PROBLEM_PATH, "parking-3-1-1.pddl"),
    osp.join(PROBLEM_PATH, "parking-3-2-1.pddl"),
    osp.join(PROBLEM_PATH, "parking-3-2.pddl"),
    osp.join(PROBLEM_PATH, "parking-3-3-1.pddl"),
    osp.join(PROBLEM_PATH, "parking-3-3-2.pddl"),
    osp.join(PROBLEM_PATH, "parking-3-3-3.pddl"),
    osp.join(PROBLEM_PATH, "parking-3-3-4.pddl"),
    osp.join(PROBLEM_PATH, "parking-3-3-5.pddl"),
    osp.join(PROBLEM_PATH, "parking-3-3.pddl"),
    osp.join(PROBLEM_PATH, "parking-3-4-1.pddl"),
    osp.join(PROBLEM_PATH, "parking-3-4-2.pddl"),
    osp.join(PROBLEM_PATH, "parking-3-4.pddl"),
    osp.join(PROBLEM_PATH, "parking-4-1.pddl"),
    osp.join(PROBLEM_PATH, "parking-4-1-1.pddl"),
    osp.join(PROBLEM_PATH, "parking-4-1-2.pddl"),
    osp.join(PROBLEM_PATH, "parking-4-2.pddl"),
    osp.join(PROBLEM_PATH, "parking-4-2-1.pddl"),
    osp.join(PROBLEM_PATH, "parking-4-2-2.pddl"),
    osp.join(PROBLEM_PATH, "parking-4-3-1.pddl"),
    osp.join(PROBLEM_PATH, "parking-4-3-2.pddl"),
    osp.join(PROBLEM_PATH, "parking-4-3-3.pddl"),
    osp.join(PROBLEM_PATH, "parking-4-3-4.pddl"),
    osp.join(PROBLEM_PATH, "parking-4-3.pddl"),
    osp.join(PROBLEM_PATH, "parking-4-4.pddl"),
    osp.join(PROBLEM_PATH, "parking-4-5.pddl"),
    osp.join(PROBLEM_PATH, "parking-4-6.pddl"),
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
            timeout = 10800
        )

    except subprocess.TimeoutExpired:
        print("timed out")