(define (problem blocks-nblk8-seed1741991660-seq9)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 - block)
    (:init (handempty) (ontable b1) (ontable b2) (on b3 b7) (on b4 b6) (on b5 b3) (on b6 b5) (on b7 b1) (on b8 b2) (clear b4) (clear b8))
    (:goal (and (handempty) (on b1 b4) (ontable b2) (on b3 b5) (ontable b4) (on b5 b8) (on b6 b3) (ontable b7) (on b8 b1) (clear b2) (clear b6) (clear b7))))