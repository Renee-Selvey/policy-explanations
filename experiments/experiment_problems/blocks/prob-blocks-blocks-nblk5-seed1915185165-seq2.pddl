(define (problem blocks-nblk5-seed1915185165-seq2)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 - block)
    (:init (handempty) (ontable b1) (on b2 b3) (ontable b3) (on b4 b1) (ontable b5) (clear b2) (clear b4) (clear b5))
    (:goal (and (handempty) (on b1 b3) (on b2 b4) (ontable b3) (on b4 b5) (ontable b5) (clear b1) (clear b2))))
