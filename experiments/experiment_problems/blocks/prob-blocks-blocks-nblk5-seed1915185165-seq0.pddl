(define (problem blocks-nblk5-seed1915185165-seq0)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 - block)
    (:init (handempty) (on b1 b5) (ontable b2) (on b3 b4) (ontable b4) (ontable b5) (clear b1) (clear b2) (clear b3))
    (:goal (and (handempty) (on b1 b5) (ontable b2) (on b3 b1) (on b4 b2) (on b5 b4) (clear b3))))
