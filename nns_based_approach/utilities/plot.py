import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation, rc
from IPython.display import HTML

rc('animation', html='html5')

def animated_imshow(data):

    frames = []
    fig = plt.figure()
    plt.axis('off')
    
    for k in range(data.shape[0]):
        frames.append([plt.imshow(data[k, ...], cmap="gray", animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True,
                                    repeat_delay=1000)
    plt.close()
    
    HTML(ani.to_html5_video())
    return ani