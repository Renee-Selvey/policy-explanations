(define (problem blocks-nblk3-seed1914988565-seq6)
    (:domain blocks)
    (:objects b1 b2 b3 - block)
    (:init (handempty) (ontable b1) (on b2 b3) (on b3 b1) (clear b2))
    (:goal (and (handempty) (on b1 b2) (ontable b2) (ontable b3) (clear b1) (clear b3))))
