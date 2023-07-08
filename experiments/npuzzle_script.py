import subprocess
import os.path as osp

THIS_DIR = osp.dirname(osp.abspath(__file__))
PROBLEM_PATH = osp.abspath(osp.join(THIS_DIR, "experiment_problems/npuzzle/all-2puzzle"))
EXP_PATH = osp.join(THIS_DIR, "../compute_explanation.py")
LOG_PATH = osp.abspath(osp.join(THIS_DIR, "../results/npuzzle/with-fixed-goal"))

weights = osp.join(PROBLEM_PATH, "n-puzzle-sparse-weights.pkl")
domain = osp.join(PROBLEM_PATH, "n-puzzle-typed.pddl")
problems = [
    osp.join(PROBLEM_PATH, "npuzzle2x2-19.pddl"),
    osp.join(PROBLEM_PATH, "npuzzle2x2-01.pddl"),
    osp.join(PROBLEM_PATH, "npuzzle2x2-23.pddl"),
    osp.join(PROBLEM_PATH, "npuzzle2x2-14.pddl"),
    osp.join(PROBLEM_PATH, "npuzzle2x2-03.pddl"),
    osp.join(PROBLEM_PATH, "npuzzle2x2-09.pddl"),
    osp.join(PROBLEM_PATH, "npuzzle2x2-11.pddl"),
    osp.join(PROBLEM_PATH, "npuzzle2x2-05.pddl"),
    osp.join(PROBLEM_PATH, "npuzzle2x2-17.pddl"),
    osp.join(PROBLEM_PATH, "npuzzle2x2-15.pddl"),
    osp.join(PROBLEM_PATH, "npuzzle2x2-16.pddl"),
    osp.join(PROBLEM_PATH, "npuzzle2x2-04.pddl"),
    osp.join(PROBLEM_PATH, "npuzzle2x2-06.pddl"),
    osp.join(PROBLEM_PATH, "npuzzle2x2-22.pddl"),
    osp.join(PROBLEM_PATH, "npuzzle2x2-13.pddl"),
    osp.join(PROBLEM_PATH, "npuzzle2x2-02.pddl"),
    osp.join(PROBLEM_PATH, "npuzzle2x2-20.pddl"),
    osp.join(PROBLEM_PATH, "npuzzle2x2-07.pddl"),
    osp.join(PROBLEM_PATH, "npuzzle2x2-18.pddl"),
    osp.join(PROBLEM_PATH, "npuzzle2x2-12.pddl"),
    osp.join(PROBLEM_PATH, "npuzzle2x2-24.pddl"),
    osp.join(PROBLEM_PATH, "npuzzle2x2-10.pddl"),
    osp.join(PROBLEM_PATH, "npuzzle2x2-08.pddl"),
    osp.join(PROBLEM_PATH, "npuzzle2x2-21.pddl"),
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