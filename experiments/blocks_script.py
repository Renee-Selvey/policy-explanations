import subprocess
import os.path as osp

THIS_DIR = osp.dirname(osp.abspath(__file__))
PROBLEM_PATH = osp.abspath(osp.join(THIS_DIR, "experiment_problems/blocks"))
EXP_PATH = osp.join(THIS_DIR, "../compute_explanation.py")
LOG_PATH = osp.abspath(osp.join(THIS_DIR, "../results/blocks/"))

weights = osp.join(PROBLEM_PATH, "blocksworld-sparse-weights.pkl")
domain = osp.join(PROBLEM_PATH, "domain.pddl")
problems = [
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk2-seed1175397191-seq0.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk2-seed1257879174-seq0.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk2-seed1257879174-seq1.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk2-seed1257879174-seq2.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk2-seed1257879174-seq4.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk2-seed1257879174-seq5.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk3-seed627976805-seq1.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk2-seed1175397191-seq1.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk2-seed1175397191-seq2.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk2-seed1257879174-seq3.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk2-seed1257879174-seq6.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk3-seed627976805-seq2.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk3-seed1914988565-seq4.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk3-seed1914988565-seq1.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk4-seed1798170167-seq0.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk3-seed627976805-seq0.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk3-seed1914988565-seq2.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk3-seed1914988565-seq3.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk3-seed1914988565-seq5.pddl"),    
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk3-seed1914988565-seq6.pddl"),    
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk3-seed1914988565-seq0.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk3-seed1914988565-seq6.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk4-seed1798170167-seq0.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk4-seed1798170167-seq1.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk4-seed1798170167-seq2.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk4-seed1798170167-seq3.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk4-seed1798170167-seq4.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk4-seed1798170167-seq5.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk4-seed1798170167-seq6.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk4-seed830202877-seq0.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk4-seed830202877-seq1.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk4-seed830202877-seq2.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk5-seed1915185165-seq0.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk5-seed1915185165-seq1.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk5-seed1915185165-seq2.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk5-seed1915185165-seq3.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk5-seed1915185165-seq4.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk5-seed1915185165-seq5.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk5-seed1915185165-seq6.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk5-seed256232149-seq0.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk5-seed256232149-seq1.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk5-seed256232149-seq2.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk6-seed2083640315-seq0.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk6-seed2083640315-seq1.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk6-seed2083640315-seq2.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk6-seed2083640315-seq3.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk6-seed2083640315-seq4.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk6-seed2083640315-seq5.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk6-seed2083640315-seq6.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk6-seed2083640315-seq7.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk6-seed2083640315-seq8.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk6-seed2083640315-seq9.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk7-seed995275251-seq0.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk7-seed995275251-seq1.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk7-seed995275251-seq2.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk7-seed995275251-seq3.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk7-seed995275251-seq4.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk7-seed995275251-seq5.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk7-seed995275251-seq6.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk7-seed995275251-seq7.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk7-seed995275251-seq8.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk7-seed995275251-seq9.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk8-seed1741991660-seq0.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk8-seed1741991660-seq1.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk8-seed1741991660-seq2.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk8-seed1741991660-seq3.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk8-seed1741991660-seq4.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk8-seed1741991660-seq5.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk8-seed1741991660-seq6.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk8-seed1741991660-seq7.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk8-seed1741991660-seq8.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk8-seed1741991660-seq9.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk9-seed555044516-seq0.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk9-seed555044516-seq1.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk9-seed555044516-seq2.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk9-seed555044516-seq3.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk9-seed555044516-seq4.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk9-seed555044516-seq5.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk9-seed555044516-seq6.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk9-seed555044516-seq7.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk9-seed555044516-seq8.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk9-seed555044516-seq9.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk10-seed1982156651-seq0.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk10-seed1982156651-seq1.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk10-seed1982156651-seq2.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk10-seed1982156651-seq3.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk10-seed1982156651-seq4.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk10-seed1982156651-seq5.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk10-seed1982156651-seq6.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk10-seed1982156651-seq7.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk10-seed1982156651-seq8.pddl"),
    osp.join(PROBLEM_PATH, "prob-blocks-blocks-nblk10-seed1982156651-seq9.pddl"),
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