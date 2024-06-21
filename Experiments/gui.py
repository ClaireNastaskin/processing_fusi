import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
import matplotlib.animation as animation
from IPython.display import HTML

def MakeAnimation(imgs, output_file=None, fps=10, depth=[], lateral=[], time=[]):
    """
    Create an animation from a list of images.

    Args:
    imgs (list): A list of images to animate. ("lateral" x "depth" x "time").
    output_file (str): The file path where the animation will be saved.
    fps (int): Frames per second for the output video.
    depth (list): A list of depth values for the images.
    lateral (list): A list of lateral values for the images.
    time (list): A list of time values for the images.

    returns:
    ani (matplotlib.animation.FuncAnimation): The animation.
    """
    # Create a figure and axis
    if len(depth) == 0:
        depth = np.arange(imgs.shape[1])
    if len(lateral) == 0:
        lateral = np.arange(imgs.shape[0])
    matplotlib.rcParams['animation.embed_limit'] = 2**64 # enable plotting large animations
    fig, ax = plt.subplots()
    fig.set_size_inches(imgs.shape[0]/50, imgs.shape[1]/50)
    cax = ax.imshow(imgs[..., 0].T, 
                    cmap='hot', 
                    extent=[lateral[0], lateral[-1], depth[-1], depth[0]])

    # Set axis labels
    ax.set_xlabel('lateral (mm)')
    ax.set_ylabel('depth (mm)')

    def update(frame):
        cax.set_data(imgs[..., frame].T)
        if len(time) > 0:
            ax.set_title(f'Frame # {frame} ({time[frame]:.1f} s)')
        else:
            ax.set_title(f'Frame # {frame}')

    ani = animation.FuncAnimation(fig, update, frames=imgs.shape[-1], interval=1000./fps)

    # Close the figure to prevent it from displaying statically
    plt.close(fig)

    # Save the animation as a GIF or MP4
    if output_file is not None:
        output_file = str(output_file)
        # if output_file is .mp4, save as mp4
        if output_file[-4:] == '.mp4':
            ani.save(output_file, writer='ffmpeg')
        # if output_file is .gif, save as gif
        elif output_file[-4:] == '.gif':
            ani.save(output_file, writer='pillow')
    
    return ani

def MakeAnimation_old(imgs, output_file=None,fps=10):
    """
    Create an animation from a list of images.

    Args:
    imgs (list): A list of images to animate. ("lateral" x "depth" x "time")
    output_file (str): The file path where the animation will be saved.
    fps (int): Frames per second for the output video.

    returns:
    ani (matplotlib.animation.ArtistAnimation): The animation.
    """
    # Create a figure and axis
    matplotlib.rcParams['animation.embed_limit'] = 2**64 # enable plotting large animations
    fig, ax = plt.subplots()
    fig.set_size_inches(imgs.shape[0]/50, imgs.shape[1]/50)
    ax.set_axis_off()

    # remove white space from animation
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    
    # Create the animation
    frames = [] # for storing the generated images
    for i in range(imgs.shape[-1]):
        frames.append([plt.imshow(imgs[...,i].T)], cmap='hot')
    ani = animation.ArtistAnimation(fig, frames, interval=1000./fps, blit=True, repeat_delay=0)
    
    # Save the animation as a GIF or MP4
    if output_file is not None:
        output_file = str(output_file)
        # if output_file is .mp4, save as mp4
        if output_file[-4:] == '.mp4':
            ani.save(output_file, writer='ffmpeg')
        # if output_file is .gif, save as gif
        elif output_file[-4:] == '.gif':
            ani.save(output_file, writer='pillow')

    # Display the animation as a GIF
    HTML(ani.to_jshtml())
    return ani

def MultipleAnimation(imgs, labels=None, output_file=None, fps=10, depth=[], lateral=[], time=[]):
    """
    Create multiple animations from a list of images based on the labels.

    Args:
    imgs (list): A list of images to animate. ("lateral" x "depth" x "time")
    labels (list): A list of labels for each image.
    output_file (str): The file path where the animation will be saved.
    fps (int): Frames per second for the output video.

    returns:
    all_ani (matplotlib.animation.ArtistAnimation): The combined animation.
    """

    if labels is None:
        labels = np.zeros(imgs.shape[-1])
    unique_labels = np.unique(labels)

    all_ani = None
    for i in range(len(unique_labels)):
        
        # select images with the same label
        img_subset = imgs[..., labels == unique_labels[i]]

        # Save the animation as a GIF or MP4
        if output_file is not None:
            output_file_rename = str(output_file)
            output_file_rename = output_file_rename[:-4] + '_' + str(unique_labels[i]) + output_file_rename[-4:]
        print(f"Creating animation for label = {unique_labels[i]}")
        ani = MakeAnimation(img_subset, 
                            output_file=output_file_rename, 
                            fps=fps, depth=depth, lateral=lateral, time=time
                            )
        
        if all_ani is None:
           all_ani = ani
        else:
            all_ani = [all_ani, ani]

    return all_ani

class ImageViewerGUI:
  """
  A class to create an interactive image viewer GUI.
  """
  
  def __init__(self, start_index=0):
    self.fig, self.ax = plt.subplots()
    self.current_index = start_index
    self.image_slider = None
    self.next_button = None
    self.prev_button = None
    self.imgs = None

  def show_image(self, index):
    self.ax.imshow(self.imgs[..., index].T, extent=[0, 1, 0, 1])
    self.fig.canvas.draw_idle()
    self.ax.set_title("Frame "+str(self.current_index))


  def on_slider_changed(self, val):
    self.current_index = int(val)
    self.show_image(self.current_index)

  def on_button_clicked_next(self, event):
    if self.current_index < self.imgs.shape[-1] - 1:
      self.current_index += 1
    self.show_image(self.current_index)

  def on_button_clicked_prev(self, event):
    if self.current_index > 0:
      self.current_index -= 1
    self.show_image(self.current_index)

  def __call__(self, imgs):
    self.imgs = imgs

    # Slider to control image index
    slider_ax = self.fig.add_axes([0.2, 0.0, 0.65, 0.03])  # Position the slider
    self.image_slider = Slider(ax=slider_ax,
                   label='Image Index',
                   valmin=0,
                   valmax=self.imgs.shape[-1] - 1,
                   valinit=self.current_index,
                   valfmt='%0.0f')

    # Button to go to the next image
    next_button_ax = self.fig.add_axes([0.85, 0.1, 0.1, 0.03])  # Position the button
    self.next_button = Button(ax=next_button_ax, label='Next')

    # Button to go to the previous image
    prev_button_ax = self.fig.add_axes([0.85, 0.05, 0.1, 0.03])  # Position the button
    self.prev_button = Button(ax=prev_button_ax, label='Prev')

    # Connect slider and button events
    self.image_slider.on_changed(self.on_slider_changed)
    self.next_button.on_clicked(self.on_button_clicked_next)
    self.prev_button.on_clicked(self.on_button_clicked_prev)

    # connect the button to the slider
    self.next_button.on_clicked(lambda x: self.image_slider.set_val(self.image_slider.val + 1))
    self.prev_button.on_clicked(lambda x: self.image_slider.set_val(self.image_slider.val - 1))

    self.show_image(self.current_index)  # Show initial image
    plt.show()
