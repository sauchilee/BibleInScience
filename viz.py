import matplotlib.pyplot as plt

def plot_bar(data, headline, left_axis, right_axis, save=False):
    plt.bar(range(0, len(data)), data)
    plt.show()