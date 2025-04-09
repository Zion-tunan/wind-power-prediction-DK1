# Loss Visualization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Config import config

config = config()

data = pd.read_csv("../data/loss.csv")
train_loss = data[['train_loss']]
test_loss = data[['test_loss']]
y1 =np.array(train_loss)
y2 = np.array(test_loss)

plt.plot(range(1, config.epochs+1), y1, label="train_loss")
plt.plot(range(1, config.epochs+1), y2, label="test_loss")
plt.xlabel('epochs')
plt.ylabel('loss')
# plt.ylim(0.00025, 0.00225)
plt.legend()
plt.show()
