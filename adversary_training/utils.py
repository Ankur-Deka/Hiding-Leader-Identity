# visualize some results
import os
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 42})

def plot_trajectories(traj, leaderID, pred, initSteps = 10, leader_viz = True, fname = None):
    plt.figure(figsize=(10,10))
    trajLen = traj.shape[0]
    nAgents = traj.shape[1]//2

    for i in range(nAgents):
        if leader_viz and i == leaderID:
            # init steps in grey
            plt.plot(traj[:initSteps+1,i*2], traj[:initSteps+1,i*2+1], color = 'orange', alpha = 0.9)    # trajectory
            # prediction steps - red if correct, black if wrong
            for j in range(initSteps,trajLen-1):
                color = 'red' if pred[j-initSteps] == leaderID else 'black'
                plt.plot(traj[[j,j+1],i*2], traj[[j,j+1],i*2+1], color = color, alpha = 0.8, label = 'True')    
            # add an arrow for direction
            color = 'red' if pred[-1] == leaderID else 'black'
            plt.arrow(traj[-2,i*2], traj[-2,i*2+1], traj[-1,i*2]-traj[-2,i*2], traj[-1,i*2+1]-traj[-2,i*2+1], head_width=0.03, head_length=0.03, color = color)
        else:
            plt.plot(traj[:,i*2], traj[:,i*2+1], color = 'blue', alpha = 0.2)    # trajectory
            plt.arrow(traj[-2,i*2], traj[-2,i*2+1], traj[-1,i*2]-traj[-2,i*2], traj[-1,i*2+1]-traj[-2,i*2+1], head_width=0.03, head_length=0.03, color = 'blue', alpha = 0.3)
    # legend
    custom_lines = [Line2D([0], [0], color='orange', lw=4),
                    Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='black', lw=4),
                    Line2D([0], [0], color='blue', alpha = 0.3, lw=4)]
    
    plt.legend(custom_lines, ['Leader initial observation', 'Leader correct prediction', 'Leader wrong prediction', 'Followers'], fontsize=20)
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    plt.tight_layout()
    if not fname is None:
        plt.savefig(fname, dpi=300)
    plt.close()


def lossFunc(outputs,labels,criterion):
    outDim = outputs.shape[-1]
    loss = criterion(outputs.contiguous().view(-1,outDim), labels.contiguous().view(-1))
    # if self.stacked:
    #   loss *= self.timeWindow     # so that loss plots are comparable
    return loss