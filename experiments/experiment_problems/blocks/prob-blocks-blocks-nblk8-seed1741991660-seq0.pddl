(define (problem blocks-nblk8-seed1741991660-seq0)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 - block)
    (:init (handempty) (on b1 b4) (on b2 b5) (ontable b3) (ontable b4) (on b5 b6) (on b6 b7) (ontable b7) (on b8 b1) (clear b2) (clear b3) (clear b8))
    (:goal (and (handempty) (on b1 b7) (on b2 b3) (on b3 b6) (on b4 b1) (on b5 b4) (ontable b6) (ontable b7) (ontable b8) (clear b2) (clear b5) (clear b8))))
