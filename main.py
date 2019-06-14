from experiments.cartpole import cartpole_ppo
from utils.functions import graph


def main():
    data = cartpole_ppo()
    graph(data)


if __name__ == "__main__":
    main()
