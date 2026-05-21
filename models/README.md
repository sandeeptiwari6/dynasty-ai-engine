# Modelling Layer
___
This folder contains the implementation of the following models

## Models
### 0 Base model
`base.py` —  the abstract foundation. Defines `DynastyModel` (ABC all models inherit from), `PredictionResult` (a standardized output schema so the agent layer always gets the same dict shape), MLflow tracking wrappers, and the SHAP-to-human-readable-strings utility shared by all three models

### 1 NFL Forecaster
This is the first model that uses **LightGBM quantile regression**, with four models, one per position (QB, RB, WR, TE). The quantile regression mode (three models at `alpha=0.1`, `0.5`, `0.9`) gives the floor and ceiling estimates for each fantasy-relevant player—this is essential for dynasty value, where a boom-bust WR and a consistent one can have the same mean projection but very different roster values.

### 2 Injury Risk
This is the second model, which uses **Calibrated logistic regression** with Platt scaling. Although LightGBM would yield a higher AUC, because we only have 15-20% playing injured, GBMs would produce poorly calibrated probabilities. The dynasty agent needs to say "this player has a 35% injury risk," and that 35% needs to be accurate, not just ranked correctly. Platt scaling (`CalibratedClassifierCV`) on logistic regression achieves genuine probability calibration. 

### 3 College Translator
`college_translator.py` - **RidgeCV + KNN comps** for prospect translation. Since we only have 5-30 historical draft prospects per position with both college data *and* 3-year NFL outcomes, LightGBM would badly overfit here. Ridge with L2 regularization shrinks correlated features (dominator rating and targets/game are correlated) toward each other rather than picking one arbitrarily like Lasso would. `RidgeCV` selects the regularization strength automatically via generalized cross-validation. The KNN comp lookup runs separately and provides the uncertainty range (variance of historical comp outcomes) plus the narrative that dynasty players actually trust.  

## Model Store
This acts as the unified inference interface. The LangGraph agent layer will import only this file, and it never touches any of the model types directly. It Contains a smart router (predict_player) that auto-selects between the NFL forecaster and college translator based on years of experience.