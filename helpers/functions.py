import matplotlib.pyplot as plt


def calculate_returns(rollouts, gamma):
    for rollout in rollouts:
        discounted = 0
        for i in reversed(range(len(rollout))):
            discounted = gamma * discounted + rollout[i]["reward"]
            rollout[i]["return"] = discounted


def graph(data):
    print(data)
    plt.plot(data)
    plt.savefig("fig")
    plt.clf()
