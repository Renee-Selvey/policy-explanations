(define (problem blocks-nblk9-seed555044516-seq4)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 - block)
    (:init (handempty) (on b1 b3) (ontable b2) (on b3 b5) (on b4 b9) (on b5 b8) (on b6 b7) (on b7 b4) (ontable b8) (on b9 b1) (clear b2) (clear b6))
    (:goal (and (handempty) (on b1 b8) (on b2 b7) (on b3 b9) (ontable b4) (on b5 b2) (on b6 b1) (on b7 b6) (on b8 b4) (ontable b9) (clear b3) (clear b5))))