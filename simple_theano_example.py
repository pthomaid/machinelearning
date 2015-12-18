
"""A simple scipt that uses Theano to learn a simple model of a 1D function
"""

import theano
from theano import tensor as T
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

trX = np.linspace(-1, 1, 101)
trY = -2*trX + np.random.randn(*trX.shape)*0.33
X = T.scalar()
Y = T.scalar()

def model(X, w):
    return X*w

w = theano.shared(np.asarray(0., dtype=theano.config.floatX))
y = model(X, w)

cost = T.mean(T.sqr(y - Y))
gradient = T.grad(cost*cost, wrt=w)
updates = [[w, w - gradient*0.01]]

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
test = theano.function(inputs=[X], outputs=y, allow_input_downcast=True)

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(-2, 2), ylim=(-2, 2))
line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

def animate(i):
    for x, y in zip(trX, trY):
        train(x, y)
    
    line.set_data(trX, [test(xi) for xi in trX])
    return line,
    
# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=2000, interval=60, blit=True)

plt.plot(trX, trY)
plt.show()
