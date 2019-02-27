from experiments.cartpole import train_cartpole
from helpers.functions import graph


def main():
    returns = train_cartpole()
    graph(returns)


if __name__ == "__main__":
    main()
