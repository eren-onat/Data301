# importing required libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

#want interactivity in a jupyter notebook?
#install ipympl
# !conda install -c conda-forge ipympl -y
#and then turn on the widget by running this magic
#%matplotlib ipympl

#determines color of points plotted, typically cluster number
colors1 = {-2:"brown",
            -1:"black",
            0:"cyan",
           1:"orange", 
           2:"purple",
           3:"green",
          4:"yellow",
          5:"red",
          6:"blue",
           7:"teal",
           8:"pink",
           9:"gold",
           10:"royalblue",
           11:"bisque",
           12:"brown"
          }

def plot_3D(x,y,z,hue, labels=['x','y','z','Title'], clrs=colors1):
    '''
    A 3d plot
    x pd.series for x axis
    y pd.series for y axis
    z pd.series for z axis
    hue pd.series of ints for hue, mapped to clrs, if hue has more than (len(colors1)) values then extend colors1
    clrs dict (see above) 
    '''
    # creating figure
    fig = plt.figure();

    ax = Axes3D(fig,auto_add_to_figure=False)
    ignore=fig.add_axes(ax)
    ignore=ax.set_facecolor("white");
    ignore=ax.grid(color="black");
    
    # creating the plot
    ignore=ax.scatter(x, y, z, c=hue.map(clrs))
    
    # setting title and labels
    ignore=ax.set_title(labels[3])
    ignore=ax.set_xlabel(labels[0])
    ignore=ax.set_ylabel(labels[1])
    ignore=ax.set_zlabel(labels[2])
    plt.show()

def palette_to_dict(palette_name):
  """Converts a seaborn color palette to a dictionary.
  Args:
    palette_name: The name of the seaborn color palette (string),
                  or the palette itself.
  Returns:
    A dictionary where keys are indices (0, 1, 2, ...) and values are
    the corresponding RGB tuples (in the range 0-1).
  Example usage:
    palette_dict = palette_to_dict("deep") 
    print(palette_dict)
  Example with a custom palette
    custom_palette = ["red", "green", "blue"]
    palette_dict_custom = palette_to_dict(custom_palette)
    print(palette_dict_custom)
  """
  palette = sns.color_palette(palette_name)
  return {i: color for i, color in enumerate(palette)}

#example using above to use seaborns 
#ut.plot_3D(x=features_pca[0], y=features_pca[1], z=features_pca[2], hue=pd.Series(kmeans.labels_),clrs=ut.palette_to_dict("deep") )