(define (problem blocks-nblk10-seed1982156651-seq9)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 - block)
    (:init (handempty) (ontable b1) (on b2 b8) (ontable b3) (on b4 b6) (ontable b5) (ontable b6) (on b7 b1) (on b8 b7) (ontable b9) (on b10 b5) (clear b2) (clear b3) (clear b4) (clear b9) (clear b10))
    (:goal (and (handempty) (on b1 b7) (on b2 b6) (ontable b3) (on b4 b8) (on b5 b10) (on b6 b9) (ontable b7) (on b8 b5) (on b9 b1) (on b10 b2) (clear b3) (clear b4))))