(define (problem blocks-nblk3-seed627976805-seq0)
    (:domain blocks)
    (:objects b1 b2 b3 - block)
    (:init (handempty) (ontable b1) (ontable b2) (on b3 b1) (clear b2) (clear b3))
    (:goal (and (handempty) (on b1 b3) (on b2 b1) (ontable b3) (clear b2))))