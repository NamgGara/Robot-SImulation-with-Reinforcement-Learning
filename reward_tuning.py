class Reward_for_standing():
    def __init__(self) -> None:
        
        self.threshold = 0
        self.reward = 5
        self.standing_reward = 1
        self.punishment = -1
        self.gap = 2
    
    def set_threshold(self, value):
        self.threshold = value
    
    def reset(self):
        self.threshold = 0

    def overlapping_punishment(self,list):
        if not list:
            return -1
        return 0
# put in switch logic for contact that is for overlapping
    def __call__(self,progress, contact, idx):
        # if idx%2==0:
        #     return self.overlapping_punishment(contact)
        
        if progress > self.threshold:
            self.threshold = progress
            return self.reward
        elif progress < self.threshold and progress > self.threshold - self.gap:
            return self.standing_reward
        return self.punishment

reward = Reward_for_standing()