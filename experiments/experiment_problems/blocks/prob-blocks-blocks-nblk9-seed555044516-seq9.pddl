(define (problem blocks-nblk9-seed555044516-seq9)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 - block)
    (:init (handempty) (on b1 b5) (on b2 b1) (on b3 b6) (on b4 b3) (on b5 b7) (ontable b6) (on b7 b8) (ontable b8) (on b9 b4) (clear b2) (clear b9))
    (:goal (and (handempty) (on b1 b8) (on b2 b9) (on b3 b6) (ontable b4) (on b5 b1) (on b6 b2) (ontable b7) (on b8 b3) (ontable b9) (clear b4) (clear b5) (clear b7))))
