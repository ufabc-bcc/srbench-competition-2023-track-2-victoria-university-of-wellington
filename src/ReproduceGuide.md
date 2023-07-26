### Steps:
1. Install the environment for "Evolutionary Forest" by running the following command:

  ```bash
  python -m venv competition
  source competition/bin/activate
  git clone git@github.com:ufabc-bcc/srbench-competition-2023-track-2-victoria-university-of-wellington.git
  pip install scipy hdfe numpy seaborn matplotlib deap sympy pandas scikit_learn dill lightgbm gplearn skorch umap-learn category_encoders pyade networkx torch tpot linear-tree sklearn2pmml shap mlxtend
  pip install deap==1.3.3
  export PYTHONPATH=./evolutionary_forest:./:$PYTHONPATH
  ```

2. Execute the `GPFC.py` script to obtain the results and corresponding figures.
