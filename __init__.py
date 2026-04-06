class MyEnv:
    def __init__(self):
        self.state_data = {}

    def reset(self):
        self.state_data = {
            "input": "This is a complaint message",
            "progress": 0
        }
        return self.state()

    def state(self):
        return self.state_data

    def step(self, action):
        reward = 0
        done = False

        if action == "complaint":
            reward = 1
            self.state_data["progress"] = 100
            done = True
        else:
            reward = -0.2

        return self.state(), reward, done, {}
