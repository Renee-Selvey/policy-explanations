(define (problem blocks-nblk10-seed1982156651-seq5)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 - block)
    (:init (handempty) (on b1 b10) (on b2 b7) (on b3 b9) (ontable b4) (on b5 b1) (on b6 b3) (on b7 b8) (on b8 b4) (on b9 b5) (ontable b10) (clear b2) (clear b6))
    (:goal (and (handempty) (on b1 b8) (on b2 b5) (on b3 b6) (ontable b4) (on b5 b3) (on b6 b10) (ontable b7) (on b8 b2) (on b9 b1) (on b10 b4) (clear b7) (clear b9))))
