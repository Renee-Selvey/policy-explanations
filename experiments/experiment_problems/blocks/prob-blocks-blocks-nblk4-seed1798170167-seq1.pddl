(define (problem blocks-nblk4-seed1798170167-seq1)
    (:domain blocks)
    (:objects b1 b2 b3 b4 - block)
    (:init (handempty) (on b1 b4) (on b2 b3) (ontable b3) (ontable b4) (clear b1) (clear b2))
    (:goal (and (handempty) (ontable b1) (ontable b2) (on b3 b2) (ontable b4) (clear b1) (clear b3) (clear b4))))
