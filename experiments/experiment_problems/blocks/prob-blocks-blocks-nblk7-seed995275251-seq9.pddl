(define (problem blocks-nblk7-seed995275251-seq9)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 - block)
    (:init (handempty) (on b1 b4) (on b2 b5) (on b3 b6) (on b4 b7) (ontable b5) (ontable b6) (on b7 b2) (clear b1) (clear b3))
    (:goal (and (handempty) (on b1 b3) (on b2 b6) (on b3 b4) (ontable b4) (ontable b5) (on b6 b7) (on b7 b1) (clear b2) (clear b5))))
