(define (problem blocks-nblk8-seed1741991660-seq3)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 - block)
    (:init (handempty) (on b1 b3) (on b2 b4) (on b3 b5) (on b4 b7) (ontable b5) (ontable b6) (on b7 b6) (on b8 b1) (clear b2) (clear b8))
    (:goal (and (handempty) (on b1 b8) (on b2 b6) (on b3 b2) (on b4 b7) (on b5 b1) (on b6 b5) (ontable b7) (on b8 b4) (clear b3))))
