(define (problem blocks-nblk4-seed830202877-seq1)
    (:domain blocks)
    (:objects b1 b2 b3 b4 - block)
    (:init (handempty) (on b1 b3) (on b2 b1) (on b3 b4) (ontable b4) (clear b2))
    (:goal (and (handempty) (on b1 b4) (ontable b2) (on b3 b1) (ontable b4) (clear b2) (clear b3))))
