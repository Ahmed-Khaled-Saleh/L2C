# plot accuraccies and losses for 500 epochs in a subplot
# make the accuracies and the losses two lists


import matplotlib.pyplot as plt
import numpy as np


accuracies = [80, 82, 85, 86, 89]
losses = [0.5, 0.4, 0.3, 0.2, 0.2]


def plot_accuracies_and_losses(accuracies, losses):
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.plot([100, 200, 300, 400, 500], accuracies, label='Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    #plt.legend("L2C", loc='upper left')
    plt.subplot(1, 2, 2)
    plt.plot([100, 200, 300, 400, 500], losses, label='Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    #plt.legend('Learning to collaborate', loc='upper right')

    plt.show()
f = plot_accuracies_and_losses(accuracies, losses)