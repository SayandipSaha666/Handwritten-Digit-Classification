import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import warnings
import tensorflow as tf
import math


def widgvis(fig):
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
def plt_learning_curve(history):
    fig,ax = plt.subplots(1,1,figsize=(5,4))
    widgvis(fig)
    ax.plot(history.history['loss'],label='loss')
    ax.set_ylim([0,2])
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Learning Curve')
    ax.legend()
    plt.grid(True)
    plt.show()
def display_digit(x):
    fig,ax = plt.subplots(1,1,figsize=(0.5,0.5))
    widgvis(fig)
    x_reshaped = x.reshape((20,20)).T
    ax.imshow(x_reshaped,cmap='gray')
    plt.show()
def count_error(model, X, y):
    f = model.predict(X)
    yhat = np.argmax(f, axis=1)
    idxs = np.where(yhat != y[:, 0])[0]

    if len(idxs) == 0:
        print("No errors found")
    else:
        cnt = len(idxs)
        x = math.ceil(math.sqrt(cnt))
        
        # Create subplots
        fig, ax = plt.subplots(x, x, figsize=(5, 5))
        fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.80])
        
        # Convert ax into a 2D array if needed
        ax = np.array(ax).reshape(x, x)
        
        for i in range(cnt):
            j = idxs[i]
            row, col = divmod(i, x)  # Convert i into 2D index

            X_reshaped = X[j].reshape((20, 20)).T
            ax[row, col].imshow(X_reshaped, cmap="gray")

            # Predict again for label
            X_sample = X[j].reshape(1, -1)  # Ensure correct shape
            prediction = model.predict(X_sample)
            prediction_p = tf.nn.softmax(prediction)
            yhat = np.argmax(prediction_p)

            # Display label
            ax[row, col].set_title(f"{y[j,0]},{yhat}", fontsize=10)
            ax[row, col].set_axis_off()

        fig.suptitle("Label, yhat", fontsize=12)
        plt.show()  # Show the figure
    
    return len(idxs)

def calculate_accuracy(y_train,y_predicted):
    accuracy = np.mean(y_train == y_predicted) * 100
    return accuracy

