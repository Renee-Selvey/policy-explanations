(define (problem blocks-nblk8-seed1741991660-seq1)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 - block)
    (:init (handempty) (ontable b1) (on b2 b6) (on b3 b8) (on b4 b7) (on b5 b1) (on b6 b4) (on b7 b5) (ontable b8) (clear b2) (clear b3))
    (:goal (and (handempty) (on b1 b5) (on b2 b4) (ontable b3) (on b4 b8) (ontable b5) (on b6 b2) (on b7 b6) (on b8 b3) (clear b1) (clear b7))))
