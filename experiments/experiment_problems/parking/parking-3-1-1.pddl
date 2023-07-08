(define   (problem parking-3-1)
  (:domain parking-typed)
  (:objects
     car_0 - car
     curb_0 curb_1 curb_2 - curb
  )
  (:init
    (at-curb car_0)
    (at-curb-num car_0 curb_0)
    (car-clear car_0)
    (curb-clear curb_1)
    (curb-clear curb_2)
  )
  (:goal
    (and
      (at-curb-num car_0 curb_1)
    )
  )
)
; =========== INIT =========== 
;  curb_0: car_0 
;  curb_1: 
;  curb_2: 
; ========== /INIT =========== 

; =========== GOAL =========== 
;  curb_0: car_0 
;  curb_1: 
;  curb_2: 
; =========== /GOAL =========== 
