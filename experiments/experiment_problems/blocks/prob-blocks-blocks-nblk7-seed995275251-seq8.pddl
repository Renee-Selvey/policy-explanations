(define (problem blocks-nblk7-seed995275251-seq8)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 - block)
    (:init (handempty) (on b1 b7) (ontable b2) (on b3 b4) (on b4 b5) (on b5 b1) (ontable b6) (on b7 b6) (clear b2) (clear b3))
    (:goal (and (handempty) (on b1 b4) (ontable b2) (ontable b3) (on b4 b6) (on b5 b1) (on b6 b3) (ontable b7) (clear b2) (clear b5) (clear b7))))