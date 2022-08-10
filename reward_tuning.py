
class Reward_for_standing():
    def __init__(self) -> None:
        
        self.threshold = 0
        self.reward = 5
        self.punishment = -1

    def __call__(self,progress):
        if progress > self.threshold:
            self.threshold = progress
            return self.reward
        return self.punishment

reward = Reward_for_standing()