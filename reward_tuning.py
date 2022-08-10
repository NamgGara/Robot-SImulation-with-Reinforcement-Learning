
def overlapping_punishment(list):
    if not list:
        return -1

class Reward_for_standing():
    def __init__(self) -> None:
        
        self.threshold = 0
        self.reward = 5
        self.punishment = -1
    
    def set_threshold(self, value):
        self.threshold = value

    def __call__(self,progress):
        if progress > self.threshold:
            self.threshold = progress
            return self.reward
        return self.punishment

reward = Reward_for_standing()