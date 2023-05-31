import networkx as nx
import numpy as np
from deap.gp import graph
from matplotlib import gridspec, pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import seaborn as sns


def plot_tree(
        genes, coefs,
        ncols=4  # Number of plots per row
):
    function_name = {
        'multiply': 'MUL',
        'analytical_quotient': 'AQ',
        'subtract': 'SUB',
        'add': 'ADD',
    }
    n = len(genes)  # Total number of plots
    nrows = n // ncols if n % ncols == 0 else n // ncols + 1  # Calculate the number of rows
    last_row_cols = n % ncols  # Number of plots in the last row
    # Create a grid with nrows rows and ncols columns
    gs = gridspec.GridSpec(nrows, ncols)
    for i in range(len(genes)):
        gene = genes[i]
        coef = np.round(coefs[i], 2)
        for g in gene:
            if hasattr(g, 'value') and isinstance(g.value, str) and g.value.startswith('ARG'):
                g.value = g.value.replace('ARG', 'X')
            if g.name in function_name:
                g.name = function_name[g.name]

        # ax = plt.subplot(3, len(genes) // 3, i + 1)

        # Calculate row and column index for subplot
        row = i // ncols
        col = i % ncols
        # If it's the last row and there are fewer plots, we adjust the column index
        if row == nrows - 1 and last_row_cols != 0:
            empty_space = ncols - last_row_cols
            left_empty_space = empty_space // 2
            col += left_empty_space
        ax = plt.subplot(gs[row, col])

        nodes, edges, labels = graph(gene)
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        pos = graphviz_layout(g, prog="dot")

        # Generate a list of colors from the "mako" color palette, excluding the darker shades
        sns.set_palette("mako")
        color_palette = sns.color_palette("mako", n_colors=len(set(labels.values())) * 2)
        light_color_palette = color_palette[len(color_palette) // 2:]  # Use only the second half of the colors
        assert len(light_color_palette) == len(set(labels.values()))
        # Generate a dictionary to map node text to lighter colors from the "mako" color palette
        node_text_colors = {text: color for text, color in zip(set(labels.values()), light_color_palette)}
        # Generate a list of colors based on the node text
        node_colors = [node_text_colors[labels[node]] for node in nodes]

        nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colors, node_size=500)
        nx.draw_networkx_edges(g, pos, ax=ax)
        nx.draw_networkx_labels(g, pos, labels, ax=ax)

        plt.title(f"Feature #{i + 1} (W:{coef})")
    plt.tight_layout()
