__author__ = "Kramer84"

__all__ = ["plotSobolIndicesWithErr"]

import numpy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plotSobolIndicesWithErr(
    S, errS, varNames, n_dims, Stot=None, errStot=None, dimNames=None, figsize=(20, 10)
):
    """Plot Sobol' indices with error bars to visualize estimator uncertainty.

    Supports plotting for scalar outputs (1D array of indices) using discrete 
    errorbar markers, and vector outputs (2D array of indices) using either subplots 
    or comprehensive heatmaps via an `ImageGrid`.

    Parameters
    ----------
    S : numpy.ndarray
        First-order Sobol' indices. Can be shape `(n_dims,)` for scalar outputs 
        or `(n_dims, dimOut)` for vector outputs.
    errS : numpy.ndarray
        Confidence intervals or errors corresponding to `S`. Must match the shape of `S`.
    varNames : list of str
        Names of the input variables (length must match `n_dims`).
    n_dims : int
        The total number of input variables/dimensions analyzed.
    Stot : numpy.ndarray, optional
        Total-order Sobol' indices. Must match the shape of `S`. Default is None.
    errStot : numpy.ndarray, optional
        Confidence intervals or errors corresponding to `Stot`. Must match the shape of `Stot`.
    dimNames : list of str, optional
        Names of the output vector dimensions. Only evaluated for vector outputs in 
        heatmap configurations. Default is None.
    figsize : tuple of int, optional
        Width and height of the target matplotlib figure in inches. Default is (20, 10).

    Notes
    -----
    The function adapts dynamically based on the dimensionality of `S`. For high-dimensional 
    vector outputs, it uses heatmaps to manage visual density.
    """
    plt.style.use("classic")
    S, errS = numpy.squeeze(S), numpy.squeeze(errS)
    if Stot is not None and errStot is not None:
        Stot, errStot = numpy.squeeze(Stot), numpy.squeeze(errStot)

    if len(S.shape) == 1:
        print("The output is scalar")
        print(
            "The sensitivity is measured accordingly to the",
            n_dims,
            "input variables, namely:\n",
            " and ".join(varNames),
        )

        lgd_elems = [
            mpatches.Circle((0, 0), radius=7, color="r", label="first order indices"),
            mpatches.Circle((0, 0), radius=7, color="b", label="total order indices"),
        ]

        x = numpy.arange(n_dims)
        y = S
        yerr = errS  # to have 95%                                 #####

        fig, ax = plt.subplots(figsize=figsize)
        ax.errorbar(x, y, yerr=yerr, fmt="s", color="r", ecolor="r")
        if Stot is not None and errStot is not None:
            y2 = numpy.squeeze(Stot)
            y2err = numpy.squeeze(errStot)  #####
            ax.errorbar(x + 0.05, y2, yerr=y2err, fmt="o", color="b", ecolor="b")
        else:
            lgd_elems.pop()
        ax.legend(handles=lgd_elems, loc="upper right")
        ax.set_xticks(ticks=x)
        ax.set_xticklabels(labels=varNames)

        ax.tick_params(axis="x", rotation=45)

        ax.axis(xmin=-0.5, xmax=x.max() + 0.5, ymin=-0.1, ymax=1.1)
        plt.grid()
        plt.show()

    if len(S.shape) == 2:
        print("The output is a vector")
        plt.ion()
        fig = plt.figure(figsize=figsize)
        # Here we dinamically build our grid according to the number of input
        # dims
        if n_dims <= 5:
            n_cols = case = 1
        elif n_dims > 5 and n_dims <= 10:
            n_cols = case = 2
        else:
            case = 3

        graphList = list()
        if case == 1:
            colspan = 5
            rowspan = 2
            colTot = 5
            rowTot = 2 * n_dims
            for i in range(n_dims):
                graphList.append(
                    plt.subplot2grid(
                        (rowTot, colTot),
                        (i * rowspan, 0),
                        colspan=colspan,
                        rowspan=rowspan,
                        fig=fig,
                    )
                )
                graphList[i].set_title(varNames[i], fontsize=10)

                dimOut = S.shape[1]
                x = numpy.arange(dimOut)
                y = S[i, ...]
                yerr = errS[i, ...]

                graphList[i].errorbar(x, y, yerr, color="r", ecolor="b")
                graphList[i].axis(
                    xmin=-0.5, xmax=x.max() + 0.5, ymin=y.min() - 0.1, ymax=y.max() + 0.1
                )
            fig.subplots_adjust(hspace=0.25, wspace=0.25)
            plt.tight_layout()
            fig.canvas.draw()
            plt.show()

        if case == 2:
            colspan = 5
            rowspan = 2
            colTot = 5 * 2
            rowTot = 5  # (cause we fill up at least the full left side)
            for i in range(n_dims):
                col = 0
                if i > 5:
                    col = 1
                graphList.append(
                    plt.subplot2grid(
                        (rowTot, colTot),
                        (i * rowspan, col * 5),
                        colspan=colspan,
                        rowspan=rowspan,
                        fig=fig,
                    )
                )
                graphList[i].set_title(varNames[i], fontsize=10)

                dimOut = S.shape[1]
                x = numpy.arange(dimOut)
                y = S[i, ...]
                yerr = errS[i, ...]

                graphList[i].errorbar(x, y, yerr, color="r", ecolor="b")
                graphList[i].axis(
                    xmin=-0.5, xmax=x.max() + 0.5, ymin=y.min() - 0.01, ymax=y.max() + 0.01
                )
            fig.subplots_adjust(hspace=0.25, wspace=0.25)
            plt.tight_layout()
            fig.canvas.draw()
            plt.show()

        if case == 3:
            if Stot is not None:
                from mpl_toolkits.axes_grid1 import ImageGrid

                fig = plt.figure(figsize=figsize)
                grid = ImageGrid(
                    fig,
                    111,  # similar to subplot(111)
                    nrows_ncols=(2, 2),  # creates 2x2 grid of axes
                    axes_pad=0.2,  # pad between axes in inch.
                )

                S_im, S_cbar = heatmap(
                    S, dimNames, varNames, ax=grid[0], cmap="plasma", cbarlabel="Sobol' indices"
                )
                S_texts = annotate_heatmap(S_im, valfmt="{x:.2f} t")

                errS_im, errS_cbar = heatmap(
                    errS, dimNames, varNames, ax=grid[1], cmap="plasma", cbarlabel="Sobol' indices"
                )
                errS_texts = annotate_heatmap(errS_im, valfmt="{x:.2f} t")

                Stot_im, Stot_cbar = heatmap(
                    Stot,
                    dimNames,
                    varNames,
                    ax=grid[2],
                    cmap="plasma",
                    cbarlabel="Total Sobol' indices",
                )
                Stot_texts = annotate_heatmap(Stot_im, valfmt="{x:.2f} t")

                errStot_im, errStot_cbar = heatmap(
                    errStot,
                    dimNames,
                    varNames,
                    ax=grid[3],
                    cmap="plasma",
                    cbarlabel="Total Sobol' indices",
                )
                errStot_texts = annotate_heatmap(errStot_im, valfmt="{x:.2f} t")

                fig.tight_layout()
                plt.show()

    if len(S.shape) == 3:
        pass

    plt.style.use("default")


# From : https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """Create a heatmap from a 2D numpy array and two lists of axis labels.

    Parameters
    ----------
    data : numpy.ndarray
        A 2D numpy array of shape (N, M) containing the matrix values to plot.
    row_labels : list or array-like of str
        A list or array of length N containing labels for the rows.
    col_labels : list or array-like of str
        A list or array of length M containing labels for the columns.
    ax : matplotlib.axes.Axes, optional
        An existing axes instance to draw the heatmap onto. If None, the current 
        active axes (`plt.gca()`) will be used or generated. Default is None.
    cbar_kw : dict, optional
        A dictionary containing keyword arguments forwarded directly to 
        `matplotlib.figure.Figure.colorbar`. Default is None.
    cbarlabel : str, optional
        The text label assigned to the colorbar's y-axis. Default is "".
    **kwargs : dict
        All other keyword arguments are forwarded directly to `ax.imshow`.

    Returns
    -------
    im : matplotlib.image.AxesImage
        The generated image plot mapping data values to the colormap.
    cbar : matplotlib.colorbar.Colorbar
        The colorbar instance tied to the image plotting axes.
    """
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(numpy.arange(data.shape[1]))
    ax.set_yticks(numpy.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(numpy.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(numpy.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im, data=None, valfmt="{x:.3f}", textcolors=("black", "white"), threshold=None, **textkw
):
    """Annotate a text layer over a pre-existing matplotlib heatmap image.

    Loops through the individual cells of a heatmap grid and renders an explicit text 
    string displaying the cell's underlying value, adjusting text contrast color dynamically 
    based on a defined threshold map.

    Parameters
    ----------
    im : matplotlib.image.AxesImage
        The active heatmap image axes instance to be annotated.
    data : list or numpy.ndarray, optional
        The numerical data used to populate the text labels. If None, the function 
        extracts data directly from `im.get_array()`. Default is None.
    valfmt : str or matplotlib.ticker.Formatter, optional
        The formatting blueprint used for the text overlays. Accepts a standard 
        string format scheme (e.g., "${x:.2f}") or an instance of a Matplotlib Formatter. 
        Default is "{x:.3f}".
    textcolors : tuple of str, optional
        A structural pair of color definitions. The first index string specifies 
        the color used for values sitting below the threshold; the second index 
        handles values sitting above. Default is ("black", "white").
    threshold : float, optional
        The value limit in raw data units determining when text contrast colors switch. 
        If None, the separation boundary defaults precisely to the center of the 
        colormap's range. Default is None.
    **textkw : dict
        All other keyword arguments are forwarded directly into each individual 
        `ax.text` rendering invocation.

    Returns
    -------
    list of matplotlib.text.Text
        A list containing all text label instances rendered into the heatmap grid.
    """
    if not isinstance(data, (list, numpy.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
