(define (problem blocks-nblk3-seed627976805-seq1)
    (:domain blocks)
    (:objects b1 b2 b3 - block)
    (:init (handempty) (on b1 b2) (on b2 b3) (ontable b3) (clear b1))
    (:goal (and (handempty) (ontable b1) (on b2 b3) (ontable b3) (clear b1) (clear b2))))
