from IPython.display import clear_output


def update_progress(progress):
    '''Displays a nice text progress bar in Jupyter Notebooks.
       See [here](https://www.mikulskibartosz.name/how-to-display-a-progress-bar-in-jupyter-notebook/)
    '''
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait=True)
    text = "Progress: [{0}] {1:.1f}%".format(
        "#" * block + "-" * (bar_length - block), progress * 100
    )
    print(text)
