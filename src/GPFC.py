import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sympy import simplify
import sympy as sp
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.common_tool import plot_tree
from src.data_loader import load_data, load_one_column_data, y_columns
from evolutionary_forest.component.fitness import MTLR2
from evolutionary_forest.component.selection import MTLAutomaticLexicase
from evolutionary_forest.forest import EvolutionaryForestRegressor, model_to_string
from evolutionary_forest.component.test_function import MTLTestFunction
from evolutionary_forest.model.MTL import MTLRidgeCV
from evolutionary_forest.utils import reset_random
import seaborn as sns

def file_copy():
    pass

def post_train_on_missing_data(est: EvolutionaryForestRegressor, scaler: StandardScaler):
    missing_columns = [
        'Maisonneuve_1', 'Pont_Jacques_Cartier', 'Saint-Laurent U-Zelt Test', 'Parc U-Zelt Test'
    ]
    for c in missing_columns:
        X, y = load_one_column_data(c)
        X, y = clean_data(X, y, scaler)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        X = est.feature_generation(np.array(X), est.hof[0])
        pipe = Pipeline([
            ('Scaler', StandardScaler()),
            ('Ridge', RidgeCV()),
        ])
        pipe.fit(X, y)

        model = model_to_string(est.hof[0].gene, pipe['Ridge'], pipe['Scaler'])
        # Parse the expression from the string
        expression = simplify(model)
        print(f'if task=="{c}":')
        print('\t' + str(expression))

        num_variables = x_test.shape[1]
        variables = sp.symbols(f'x:{num_variables}')
        # Convert the expression to a Python function
        func = sp.lambdify(variables, expression, 'numpy')
        result = func(*x_test.T)
        score = r2_score(y_test, result)
        print(c, score)


def clean_data(X, y, scaler):
    X = np.array(X)
    y = np.array(y)
    X = scaler.transform(X)
    # Find rows with NaN values in y
    nan_rows = np.isnan(y)
    # Remove corresponding rows from X and y
    X = X[~nan_rows]
    y = y[~nan_rows]
    return X, y


def task(model='GPFC'):
    reset_random(0)
    X, y = load_data()
    indices = np.arange(0, len(X))
    X, y, indices = shuffle(X, y, indices)
    X = np.array(X)
    y = np.array(y)
    indices = np.array(indices)
    x_train, x_test, y_train, y_test, index_train, index_test = \
        train_test_split(X, y, indices, test_size=0.2, random_state=0)

    # fit and transform
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    if model == 'GPFC':
        est = EvolutionaryForestRegressor(n_gen=20, n_pop=200, select=MTLAutomaticLexicase(y.shape[1]),
                                          cross_pb=0.9, mutation_pb=0.1, max_height=3,
                                          boost_size=1, initial_tree_size='0-2', gene_num=5,
                                          mutation_scheme='EDA-Terminal-PM',
                                          basic_primitives=','.join(['Add', 'Sub', 'Mul', 'Div']),
                                          base_learner=MTLRidgeCV(), verbose=True, normalize=False,
                                          score_func=MTLR2(y.shape[1]),
                                          external_archive=1, root_crossover=True)
        funs = [MTLTestFunction(x_train, y_train, est, y.shape[1]), MTLTestFunction(x_test, y_test, est, y.shape[1])]
        est.test_fun = funs
        est.fit(x_train, y_train)
        test_r2 = r2_score(y_test, est.predict(x_test))
        print(test_r2)

        post_train_on_missing_data(est, sc)
        plot_predictions(est, x_test, y_test, index_test)
        plot_trend(est, x_test, y_test)
        print_expressions(est, x_test, y_test)
        plot_symbolic_trees(est)

        colors = sns.color_palette("mako", 2)
        fig, axes = plt.subplots(2, 3, figsize=(10 * 0.75, 8 * 0.5))
        for task_id, ax in zip(range(6), axes.flatten()):
            task_name = y_columns[task_id]
            single_est = EvolutionaryForestRegressor(n_gen=30, n_pop=200, select='AutomaticLexicase',
                                                     cross_pb=0.9, mutation_pb=0.1, max_height=3,
                                                     boost_size=1, initial_tree_size='0-2', gene_num=5,
                                                     mutation_scheme='EDA-Terminal-PM',
                                                     basic_primitives=','.join(['Add', 'Sub', 'Mul', 'Div']),
                                                     base_learner='RidgeCV', verbose=True, normalize=False,
                                                     score_func='R2',
                                                     external_archive=1, root_crossover=True)
            single_est.fit(x_train, y_train[:, task_id])
            single_test_r2 = r2_score(y_test[:, task_id], single_est.predict(x_test))
            multi_test_r2 = r2_score(y_test[:, task_id], est.predict(x_test)[:, task_id])

            x = np.arange(2)
            bar_width = 0.35
            bars = ax.bar(x, [single_test_r2, multi_test_r2], bar_width, color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(['Single', 'Multi'])
            ax.set_title(task_name)

            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 3), ha='center', va='bottom')

        file_name = './multi_task.eps'
        plt.tight_layout()
        plt.savefig(file_name, format='eps')
        file_copy(file_name)
        plt.show()

        return est, test_r2
    else:
        est, test_r2 = train_machine_learning_model(model, x_train, x_test, y_train, y_test)
        return est, test_r2


def plot_predictions(est, x_test, y_test, index_test):
    y_pred = est.predict(x_test)
    # Select the first 15 columns of y_pred and y_test
    y_pred_subset = y_pred[:, :15]
    y_test_subset = y_test[:, :15]
    titles_subset = y_columns[:15]
    # Set Seaborn's color palette to "mako"
    sns.set_palette("mako")
    # Define custom colors for the lines
    colors = sns.color_palette("mako", 2)
    # Plotting the line graph
    fig, axs = plt.subplots(3, 5, figsize=(12, 5))
    for i, ax in enumerate(axs.flat):
        sns.lineplot(index_test, y_pred_subset[:, i], label='Prediction', ax=ax, color=colors[0])
        sns.lineplot(index_test, y_test_subset[:, i], label='Ground Truth', ax=ax, color=colors[1],
                     linestyle='dashed')
        ax.set_title(titles_subset[i])
        ax.get_legend().remove()

    # Adjusting the layout
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Adding a shared legend at the bottom
    lines, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.52, 0))

    file_name = './lanes_prediction.eps'
    plt.savefig(file_name, format='eps', bbox_inches='tight')
    file_copy(file_name)
    plt.show()


def train_machine_learning_model(model, x_train, x_test, y_train, y_test):
    if model == 'LR':
        est = LinearRegression()
    elif model == 'DT':
        est = DecisionTreeRegressor()
    elif model == 'RF':
        est = RandomForestRegressor()
    elif model == 'KNN':
        est = KNeighborsRegressor()
    elif model == 'XGB':
        est = MultiOutputRegressor(XGBRegressor())
    elif model == 'LGBM':
        est = MultiOutputRegressor(LGBMRegressor())
    else:
        raise ValueError("Invalid model specified.")
    est.fit(x_train, y_train)
    test_r2 = r2_score(y_test, est.predict(x_test))
    print(test_r2)
    return est, test_r2


def plot_symbolic_trees(est):
    # plot
    mtl_id = 0
    best_ind = est.hof[0]
    learner: MTLRidgeCV = best_ind.pipe['Ridge']
    plt.figure(figsize=(10 * 1.2, 1.5 * 1.2))
    plot_tree(copy.deepcopy(best_ind.gene), learner.mtl_ridge.estimators_[mtl_id].coef_, ncols=5)
    file_name = './model.eps'
    plt.savefig(file_name, format='eps')
    file_copy(file_name)
    plt.show()


def print_expressions(est, x_test, y_test):
    all_r2 = []
    for mtl_id in range(y_test.shape[1]):
        # Define the variables
        num_variables = x_test.shape[1]
        variables = sp.symbols(f'x:{num_variables}')
        # Parse the expression from the string
        expression = simplify(est.model(mtl_id=mtl_id))
        print(f'if task=="{y_columns[mtl_id]}":')
        print('\t' + str(expression))

        # Convert the expression to a Python function
        func = sp.lambdify(variables, expression, 'numpy')
        result = func(*x_test.T)
        score = r2_score(y_test[:, mtl_id], result)
        all_r2.append(score)
    print(np.mean(all_r2))


def plot_trend(est, x_test, y_test):
    # `constructed_features` is a 2D array with 5 columns
    constructed_features = est.feature_generation(x_test, est.hof[0])
    number_of_genes = len(est.hof[0].gene)
    y_pred = est.predict(x_test)
    # Create a new figure with one row and five columns
    fig, axes = plt.subplots(1, number_of_genes, figsize=(10 * 1.5, 1.5 * 1.5))
    # Flatten the axes array to make it easier to iterate
    axes = axes.flatten()
    # Iterate over each dimension of constructed_features
    for i in range(number_of_genes):
        # Create a DataFrame for the current dimension
        data = pd.DataFrame({'Constructed Feature': constructed_features[:, i], 'Ground Truth': y_test[:, 0],
                             'Prediction': y_pred[:, 0]})
        colors = sns.color_palette("viridis", 3)
        # Select the current axis to plot on
        ax = axes[i]
        # Create a scatter plot for Ground Truth using seaborn
        sns.scatterplot(data=data, x='Constructed Feature', y='Ground Truth', ax=ax, label='Ground Truth',
                        color=colors[0])
        # Create a scatter plot for Prediction using seaborn
        sns.scatterplot(data=data, x='Constructed Feature', y='Prediction', ax=ax, label='Prediction',
                        color=colors[1])
        # Set the x-axis label
        ax.set_xlabel(str(est.hof[0].gene[i]))
        # Set the y-axis label
        ax.set_ylabel('Target')
        # Set the title
        ax.set_title(f'Feature {i + 1}')
        # Remove the legend for each subplot
        ax.get_legend().remove()
    # Adjust subplot spacing
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Adding a shared legend at the bottom
    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.52, 0))

    file_name = './correlation.eps'
    plt.savefig(file_name, format='eps')
    file_copy(file_name)
    # Show the plot
    plt.show()


def plot_comparison_results():
    # Plotting the R2 scores
    sns.set_palette("mako")
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=models, y=results, palette='mako')
    plt.xlabel('Model')
    plt.ylabel('R2 Score')
    plt.title('Comparison of R2 Scores for Different Models')
    # Add grid
    ax.grid(True, axis='y')
    # Add numeric labels on top of each bar
    for i, v in enumerate(results):
        ax.text(i, v, f"{v:.2f}", ha='center', va='bottom')
    file_name = './comparison.eps'
    plt.savefig(file_name, format='eps')
    file_copy(file_name)
    plt.show()


if __name__ == '__main__':
    models = ['GPFC', 'LR', 'DT', 'RF', 'KNN', 'XGB', 'LGBM']
    # models = ['LR', 'DT', 'RF', 'KNN', 'XGB', 'LGBM']

    # Run tasks and collect R2 scores
    results = []
    for model in models:
        est, r2 = task(model)
        results.append(r2)
    plot_comparison_results()
