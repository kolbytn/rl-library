from experiments.cartpole import cartpole_dqn
from utils.functions import graph


def main():
    data = cartpole_dqn()
    graph(data)


if __name__ == "__main__":
    main()
