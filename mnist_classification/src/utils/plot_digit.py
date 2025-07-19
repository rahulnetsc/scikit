import matplotlib.pyplot as plt

def plt_digit(digit):

    img_array = digit.reshape(28, 28)
    plt.imshow(img_array,cmap="binary")
    plt.show()

