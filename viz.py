import pandas
from IPython import embed
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import cloudpickle

# plot win rate
#
df = pandas.read_csv('reward_histories/vgfmri3_sokoban_reward_history_repeated_trial1.csv')
winrate = np.cumsum(df['win']) / np.array(df['steps'])
#print(winrate)

fig, ax = plt.subplots(2, 2)

x = np.array(df['steps'])
y = winrate
ax[0,0].plot(x, y, "-b")
ax[0,0].set_title('win rate')
ax[0,0].set_xlabel('steps')
ax[0,0].set_ylabel('# wins/step')

ax[0,1].plot(x, 1./y, "-b")
ax[0,1].set_title('(win) episode duration')
ax[0,1].set_xlabel('steps')
ax[0,1].set_ylabel('# steps/win')

# plot action values
with open('bookkeeping.pkl', 'rb') as f:
    d = cloudpickle.load(f)

# policy net
ys = np.vstack(d['policy_net_action_value_history'])
x = list(range(0, ys.shape[0]))
ax[1,0].plot(x, ys)
ax[1,0].set_title('action values')
ax[1,0].set_xlabel('steps')
ax[1,0].set_ylabel('Q (policy)')
ax[1,0].legend(['0', 'right', 'left', 'up', 'down', 'space'])


# target net
ys = np.vstack(d['target_net_action_value_history'])
x = list(range(0, ys.shape[0]))
ax[1,1].plot(x, ys)
ax[1,1].set_title('action values')
ax[1,1].set_xlabel('steps')
ax[1,1].set_ylabel('Q (target)')
ax[1,1].legend(['0', 'right', 'left', 'up', 'down', 'space'])

plt.show()
fig.savefig('viz.png')

