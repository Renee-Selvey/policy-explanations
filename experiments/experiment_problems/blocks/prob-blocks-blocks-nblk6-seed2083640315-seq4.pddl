(define (problem blocks-nblk6-seed2083640315-seq4)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 - block)
    (:init (handempty) (ontable b1) (on b2 b3) (on b3 b1) (ontable b4) (on b5 b4) (on b6 b5) (clear b2) (clear b6))
    (:goal (and (handempty) (on b1 b4) (ontable b2) (ontable b3) (on b4 b5) (on b5 b6) (on b6 b3) (clear b1) (clear b2))))
