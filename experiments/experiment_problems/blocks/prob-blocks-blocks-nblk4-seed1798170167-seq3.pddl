(define (problem blocks-nblk4-seed1798170167-seq3)
    (:domain blocks)
    (:objects b1 b2 b3 b4 - block)
    (:init (handempty) (on b1 b3) (on b2 b1) (ontable b3) (on b4 b2) (clear b4))
    (:goal (and (handempty) (on b1 b3) (ontable b2) (on b3 b2) (on b4 b1) (clear b4))))
