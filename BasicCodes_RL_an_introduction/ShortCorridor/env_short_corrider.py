MOVE_RIGHT = 1
MOVE_LEFT = 0

class ShortCorrider(object):
    def __init__(self):
        self.state = 0
        self.actions = [MOVE_LEFT, MOVE_RIGHT]

    def reset(self):
        self.state = 0
    
    def step(self, action):
        if self.state != 1:
            if action == MOVE_RIGHT:
                self.state += 1
            else :
                self.state = max(0, self.state-1)
        else:
            if self.state == 1:
                if action == MOVE_RIGHT:
                    self.state -= 1
                else :
                    self.state += 1
            
        if self.state == 3:
            return 0, True
        else :
            return -1, False


