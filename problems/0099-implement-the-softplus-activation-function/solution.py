import math
def softplus(x: float) -> float:
    if x > 100:
        return x
    if x < -100:
        return 0.0
    
    return round (math.log(1.0 + math.exp(x)), 4)
	
