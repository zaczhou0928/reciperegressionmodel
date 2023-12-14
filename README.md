# Recipe N-steps Prediction Model 
by Qianjin Zhou and Haohan Zou

# Framing the problem

## Prediction Problem & Type

This project is centered around the prediction of the number of preparation steps, **‘n_steps’**, necessary for completing a recipe. This is a regression problem because **‘n_steps’** is a continuous variable.

## Response Variable: 

The response variable is **‘n_steps’**.

We have identified **‘n_steps’** as a critical determinant of a recipe’s intricacy and the time commitment it demands.

By accurately forecasting the number of steps a recipe requires, we can empower a diverse array of food enthusiasts – from home cooks to professional chefs, recipe developers, and restaurant owners – with valuable insights. Prior to the preparation or publication of a recipe, these individuals can benefit from understanding its complexity.

Moreover, since the number of steps serves as a proxy for the recipe’s difficulty, it is also valuable information for anyone looking for recipes that fit their time constraints and cooking experience.

## Metric for Evaluation: 

Our model's performance will be assessed using the R-squared (`R²`) statistic, which is particularly well-suited for regression problems. The R² metric measures the proportion of variance in the number of steps (`n_steps`) that can be explained by the independent variables in the model. 

R² is chosen over other potential metrics, such as Mean Squared Error (`MSE`) or Mean Absolute Error (`MAE`), because it provides an intuitive understanding of the model's predictive accuracy in percentage terms. A higher R² value would indicate a model that closely matches the observed data, which in this context means accurately predicting the number of steps needed for a recipe. It allows us to quantify the effectiveness of our feature selection and the model's ability to generalize from the provided data to unseen recipes, ensuring that users receive reliable step count estimates that can inform their cooking plans and expectations.

## Justification for Feature Selection:

By the time of making predictions, features such as **‘n_ingredients’**, **‘nutritions’**, and **‘tags’** are available because we believe those information are typically available before a recipe is prepared or published. However, the rating of a recipe, which is provided by users after cooking and tasting the dish, would not be available at the time of prediction and thus is excluded from the feature set.

## Data Cleaning:

Most data cleaning processes are done in our EDA project, which is included in [this website](https://zaczhou0928.github.io/recipeanalysis/). On top of that, we have implemented some additional data cleanings to facilitate our prediction.

Adding columns: We added **‘tags’** column back.

Encoding **‘tags’**: We transformed the **‘tag’** column from strings into lists, which is a more usable format.

# Baseline Model

In the baseline model, we employed a linear regression algorithm encapsulated within a pipeline to ensure a seamless transformation of features and training of the model. The pipeline integrates preprocessing steps for both numerical and categorical data, followed by the regression model for prediction.

## Features in the Model:

Our baseline model includes two features: **‘tags’** and **‘n_ingredients’**. The two features are categorical nominal feature and quantitative discrete feature respectively.

## Feature Encoding:

### One-Hot Encoding for 'tags': 

To incorporate the categorical **'tags'** data into our model, we first transformed it with a function that identifies whether a recipe is tagged as **'easy'**. We then used one-hot encoding to convert this binary classification into a numeric format that our linear regression model can interpret.

### Log Transformation for 'n_ingredients': 

Considering that the number of ingredients (**'n_ingredients'**) might have a nonlinear relationship with the number of steps, we applied a log transformation to this feature, which can help in stabilizing variance and normalizing the distribution. 

## Model Performance:

The model's performance was evaluated using the R-squared metric, and the results are as follow:

**R-Squared for training dataset: 0.2298318956507721**
**R-Squared for testing dataset: 0.23454040104808938**

These scores indicate that around 22% of the variance in the number of steps can be explained by the model for both the training and testing sets.

## Evaluation of Model Quality:

Based on the resulting metric, our baseline model seems not ‘good’ enough. R-squared value of approximately **0.22** may suggest that there is room for improvement, as our model currently explains only a limited portion of the variance in the response variable. 

However, we believe that determining whether the model is "good" is subjective and depends on the specific context and requirements of the task. For a baseline model, such a result is not unusual, and the goal will be to enhance this performance in the subsequent iterations of the model by engineering additional features, tuning hyperparameters, and possibly exploring more complex modeling techniques.

The relatively similar R-squared scores for both the training and testing datasets suggest that our model is generalizing adequately to unseen data, which is a positive indication that the model is not overfitting the training data.

# Final Model

In the final model development phase, we experimented with various features in conjunction with linear regression to determine their predictive power. These experiments included:

**Experiment 1 (Baseline + Log Minutes + Calories)**: This model expanded upon the baseline by integrating a logarithmic transformation of the **'minutes'** feature, which aimed to smooth out the skewness often present in time-related data, and the **'calories'** feature, to investigate a potential correlation with the complexity of the recipe.

Here's a plot visualization of the distrubtion of **'calories'** variable:

<iframe src="assests/calories-hist.html" width=800 height=600 frameBorder=0></iframe>


Here's a plot visualization of the distribution of **minutes** variable:
<iframe src="assests/minutes-hist.html" width=800 height=600 frameBorder=0></iframe>


**Experiment 2 (Baseline + Log Minutes + Time-to-Make)**: In the second model, we retained the log transformation for **'minutes'** and introduced the **'time-to-make'** tag as a categorical variable. This tag could imply the relative quickness or lengthiness of a recipe's preparation time, which we theorized might have a relationship with the number of steps involved.

Here's a plot visualization of the distrubtion of **'n_steps'** variable with and without tag **'time-to-make'**:
<iframe src="assests/time-to-make-hist.html" width=800 height=600 frameBorder=0></iframe>


**Experiment 3 (Quantile Transformed Features + Tags)**: The third experiment diverged by applying a Quantile Transformation to both **'n_ingredients'** and **'minutes'**, which normalizes their distribution and can reveal more subtle associations between features and the target variable. Additionally, we encoded **'easy'** and **'time-to-make'** tags to assess their individual contributions to predicting the number of preparation steps.

Below is a performance table for the three modified models we illustrated:

| Validation Fold   |   log(minutes) + log(n_ingred) + calories + easy |   log(minutes) + log(n_ingred) + time-to-make + easy |   quantile(minutes) + quantile(n_ingred) + time-to-make + easy |
|:------------------|-------------------------------------------------:|-----------------------------------------------------:|---------------------------------------------------------------:|
| Fold 1            |                                         0.275223 |                                             0.29054  |                                                       0.296322 |
| Fold 2            |                                         0.278079 |                                             0.290342 |                                                       0.299265 |
| Fold 3            |                                         0.276808 |                                             0.289269 |                                                       0.294817 |
| Fold 4            |                                         0.271779 |                                             0.286886 |                                                       0.291561 |
| Fold 5            |                                         0.278527 |                                             0.29212  |                                                       0.299506 |

After extensive trials and evaluations with linear regression, we transitioned to the Random Forest Regressor for our final model as we wish to improve our performance further. This decision was informed by the ensemble method's superior ability to handle the complexity and non-linearity in the data—a prowess that was not fully realized by linear regression despite the various feature transformations attempted.

## Features Added:

Finally, in our final model with the Random Forest Regressor, we decide to include the two features: **‘minutes’** and **‘time-to-make’**, on top of the features we chose in the baseline model.

These two features are chosen because we believe that they represent logical facets of the data-generating process. A recipe's complexity and the time required to complete it are intuitively connected to the number of steps it contains. 

In particular, we selected **'minutes'** as a feature under the premise that a direct and positive correlation exists between the duration needed to cook a recipe and the number of steps involved—the assumption being that recipes requiring more time are likely to have a greater number of steps. 

Similarly, the **'time-to-make'** tag was incorporated based on the hypothesis that recipes marked with this tag emphasize the significance of cooking duration. This tag potentially serves as an indicator of recipes where time commitment is a notable consideration, making it a meaningful attribute for analysis.

## Feature Engineering:

Quantile Transformation of **'n_ingredients'** and **'minutes'**: In our trials and experiments, we found that applying quantile transformation on these two variables makes the linear regression model perform better. Therefore, we changed the log transformation, which we used for our baseline model, to quantile transformation for our Random Forest Regression model.

One-Hot Encoding of Tags: We extract **'time-to-make'** and **'easy'** from the **‘tags’** column and one-hot encoded it.

## Hyperparameters

The hyperparameter tuning was methodically conducted using GridSearchCV with 4 folds, and we tuned a combination of the max depth, number of estimators, and the criterion for the Random Forest to find a combination that led a model that generalized the best to unseen data.

Here's the hyperparameters we tested for the Regressor:
```
hyperparameters = {'Forest__max_depth': np.arange(1, 21, 1),
                   'Forest__n_estimators': np.arange(100, 501, 125),
                   'Forest__criterion': ['friedman_mse', 'poisson'],
                   'Forest__n_jobs': [-1]}
```

Below is a performance table summarizing the test results for different hyperparameters:

|     | Hyperparameters                                                                                                   |     Scores |
|----:|:------------------------------------------------------------------------------------------------------------------|-----------:|
|  83 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 1, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}       | 0.00569427 |
|  81 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 1, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}       | 0.00571241 |
|  80 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 1, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}       | 0.00572624 |
|  82 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 1, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}       | 0.0058239  |
|  84 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 2, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}       | 0.0139127  |
|  85 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 2, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}       | 0.0142872  |
|  87 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 2, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}       | 0.0142992  |
|  86 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 2, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}       | 0.0145077  |
|  88 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 3, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}       | 0.0248702  |
|  89 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 3, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}       | 0.0249815  |
|  90 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 3, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}       | 0.025272   |
|  91 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 3, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}       | 0.0255174  |
|  92 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 4, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}       | 0.0373343  |
|  95 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 4, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}       | 0.0375256  |
|  93 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 4, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}       | 0.03782    |
|  94 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 4, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}       | 0.0378637  |
|  97 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 5, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}       | 0.0507139  |
|  98 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 5, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}       | 0.0509922  |
|  96 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 5, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}       | 0.0511655  |
|  99 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 5, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}       | 0.0517425  |
| 101 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 6, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}       | 0.0656034  |
| 102 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 6, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}       | 0.0659031  |
| 103 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 6, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}       | 0.0663253  |
| 100 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 6, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}       | 0.0673782  |
| 105 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 7, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}       | 0.0816332  |
| 106 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 7, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}       | 0.0820332  |
| 107 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 7, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}       | 0.082377   |
| 104 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 7, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}       | 0.0827125  |
| 109 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 8, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}       | 0.097189   |
| 110 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 8, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}       | 0.0971959  |
| 108 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 8, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}       | 0.097245   |
| 111 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 8, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}       | 0.0973199  |
| 113 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 9, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}       | 0.110956   |
| 115 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 9, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}       | 0.111329   |
| 114 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 9, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}       | 0.111418   |
| 112 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 9, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}       | 0.111599   |
| 117 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 10, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}      | 0.122566   |
| 119 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 10, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}      | 0.122798   |
| 118 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 10, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}      | 0.123262   |
| 116 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 10, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}      | 0.124204   |
| 122 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 11, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}      | 0.134402   |
| 120 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 11, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}      | 0.134634   |
| 121 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 11, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}      | 0.134707   |
| 123 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 11, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}      | 0.135367   |
| 124 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 12, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}      | 0.145304   |
| 127 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 12, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}      | 0.146692   |
| 126 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 12, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}      | 0.146945   |
| 125 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 12, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}      | 0.146954   |
| 129 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 13, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}      | 0.158355   |
| 128 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 13, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}      | 0.158554   |
| 131 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 13, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}      | 0.158832   |
| 130 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 13, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}      | 0.159531   |
| 132 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 14, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}      | 0.169087   |
| 134 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 14, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}      | 0.169439   |
| 135 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 14, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}      | 0.17015    |
| 133 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 14, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}      | 0.170239   |
| 138 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 15, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}      | 0.17896    |
| 139 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 15, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}      | 0.179544   |
| 137 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 15, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}      | 0.179602   |
| 136 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 15, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}      | 0.17975    |
| 142 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 16, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}      | 0.187697   |
| 140 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 16, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}      | 0.18812    |
| 141 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 16, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}      | 0.188216   |
| 143 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 16, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}      | 0.188403   |
|   0 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 1, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}  | 0.196035   |
| 144 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 17, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}      | 0.196258   |
| 145 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 17, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}      | 0.196294   |
| 146 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 17, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}      | 0.196452   |
| 147 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 17, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}      | 0.196874   |
|   2 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 1, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}  | 0.197184   |
|   3 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 1, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}  | 0.197802   |
|   1 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 1, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}  | 0.199086   |
| 149 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 18, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}      | 0.204419   |
| 151 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 18, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}      | 0.20563    |
| 150 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 18, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}      | 0.205669   |
| 148 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 18, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}      | 0.207034   |
| 152 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 19, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}      | 0.213431   |
| 155 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 19, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}      | 0.213534   |
| 154 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 19, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}      | 0.213738   |
| 153 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 19, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}      | 0.213995   |
| 156 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 20, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}      | 0.221506   |
| 157 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 20, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}      | 0.222957   |
| 158 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 20, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}      | 0.223116   |
| 159 | {'Forest__criterion': 'poisson', 'Forest__max_depth': 20, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}      | 0.223246   |
|   4 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 2, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}  | 0.261635   |
|   7 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 2, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}  | 0.261837   |
|   6 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 2, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}  | 0.261959   |
|   5 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 2, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}  | 0.262094   |
|  77 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 20, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1} | 0.279843   |
|  76 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 20, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1} | 0.279897   |
|  78 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 20, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1} | 0.28007    |
|  79 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 20, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1} | 0.280138   |
|  72 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 19, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1} | 0.280497   |
|  73 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 19, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1} | 0.281245   |
|  74 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 19, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1} | 0.2813     |
|  75 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 19, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1} | 0.281595   |
|  69 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 18, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1} | 0.282716   |
|  68 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 18, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1} | 0.282802   |
|  70 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 18, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1} | 0.283223   |
|  71 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 18, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1} | 0.283377   |
|  64 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 17, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1} | 0.285245   |
|  65 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 17, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1} | 0.285798   |
|  66 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 17, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1} | 0.285965   |
|  67 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 17, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1} | 0.286137   |
|  60 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 16, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1} | 0.28897    |
|  61 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 16, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1} | 0.289024   |
|  63 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 16, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1} | 0.289401   |
|  62 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 16, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1} | 0.289576   |
|  56 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 15, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1} | 0.293141   |
|  57 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 15, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1} | 0.29319    |
|  59 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 15, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1} | 0.293703   |
|  58 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 15, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1} | 0.293857   |
|   8 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 3, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}  | 0.295014   |
|  11 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 3, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}  | 0.295124   |
|  10 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 3, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}  | 0.295324   |
|   9 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 3, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}  | 0.295338   |
|  52 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 14, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1} | 0.298265   |
|  53 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 14, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1} | 0.298403   |
|  55 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 14, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1} | 0.298574   |
|  54 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 14, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1} | 0.29882    |
|  48 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 13, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1} | 0.303377   |
|  50 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 13, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1} | 0.303492   |
|  51 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 13, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1} | 0.303787   |
|  49 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 13, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1} | 0.303817   |
|  12 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 4, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}  | 0.307085   |
|  15 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 4, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}  | 0.307263   |
|  14 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 4, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}  | 0.307277   |
|  13 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 4, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}  | 0.307332   |
|  44 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 12, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1} | 0.308824   |
|  46 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 12, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1} | 0.308846   |
|  45 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 12, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1} | 0.30895    |
|  47 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 12, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1} | 0.309007   |
|  42 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 11, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1} | 0.313575   |
|  40 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 11, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1} | 0.313632   |
|  41 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 11, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1} | 0.313691   |
|  43 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 11, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1} | 0.313723   |
|  18 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 5, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}  | 0.31473    |
|  16 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 5, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}  | 0.314764   |
|  17 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 5, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}  | 0.314773   |
|  19 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 5, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}  | 0.314781   |
|  37 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 10, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1} | 0.317305   |
|  38 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 10, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1} | 0.317665   |
|  36 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 10, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1} | 0.317739   |
|  39 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 10, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1} | 0.317741   |
|  20 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 6, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}  | 0.318626   |
|  21 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 6, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}  | 0.318804   |
|  22 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 6, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}  | 0.318841   |
|  23 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 6, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}  | 0.318898   |
|  32 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 9, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}  | 0.3198     |
|  33 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 9, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}  | 0.319948   |
|  34 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 9, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}  | 0.320051   |
|  35 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 9, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}  | 0.320152   |
|  25 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 7, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}  | 0.320686   |
|  24 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 7, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}  | 0.320768   |
|  28 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 8, 'Forest__n_estimators': 100, 'Forest__n_jobs': -1}  | 0.320782   |
|  26 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 7, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}  | 0.320838   |
|  27 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 7, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}  | 0.320868   |
|  29 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 8, 'Forest__n_estimators': 225, 'Forest__n_jobs': -1}  | 0.320942   |
|  30 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 8, 'Forest__n_estimators': 350, 'Forest__n_jobs': -1}  | 0.32096    |
|  31 | {'Forest__criterion': 'friedman_mse', 'Forest__max_depth': 8, 'Forest__n_estimators': 475, 'Forest__n_jobs': -1}  | 0.321078   |

**Note:** This table is sorted by the R-square score.

The best-performing hyperparameters were:

**max_depth: 8**, controlling the depth of each tree and thus preventing the model from becoming overly complex.

**n_estimators: 475**, determining the number of trees in the forest and providing a balance between model accuracy and computational efficiency.

**criterion: 'friedman_mse'**, a modification of the mean squared error that is more suitable for boosting and tends to give better results.

These parameters were selected because they balance the bias-variance trade-off and enhance the model's generalization capabilities.

## Performance Improvement

The performance of our final model is as follow:
 
**R-Squared for training dataset: 0.33910338100218707**
**R-Squared for testing dataset: 0.32649396208975057**

The performance demonstrates a clear improvement over the baseline model. With an R-Squared score of approximately 0.33 on both the training and testing datasets, the final model accounts for a third of the variance in the number of recipe steps, a substantial increase from the baseline model's 0.22. This improvement indicates that the additional features and the robustness of the Random Forest Regressor have captured more of the underlying patterns within our data.

# Fairness Analysis

In assessing the fairness of our final model, we posed the question: Does our model predict the number of steps (`n_steps`) for recipes more accurately when they are tagged 'healthy' compared to those not tagged 'healthy'? To address this, we implemented a permutation test with the following parameters:

- **Group X**: Recipes tagged with 'Healthy'.
- **Group Y**: Recipes not tagged with 'Healthy'.

We selected the **R-squared value** as our **evaluation metric** because it quantifies the proportion of variance in the predicted variable that is captured by the model.

Our **null hypothesis** posited that the model is equitable in its predictions: The R-squared values for recipes in both groups would be equivalent, indicating unbiased performance.

Conversely, the **alternative hypothesis** suggested that the model is biased: The R-squared values for recipes would differ significantly between the two groups.

The **test statistic** was the absolute difference in R-squared values between Group X and Group Y.

We set a **significance level** at 0.05, which is a standard threshold for detecting a meaningful discrepancy in statistical testing.

## Results:
Below is a distribution of our test statistics from the permutation test.

<iframe src="assests/permutation.html" width=800 height=600 frameBorder=0></iframe>

- **Observed Difference**: 0.017742070525936127
- **P-value**: 0.018

The p-value obtained from our permutation test was 0.02, which is below the significance level of 0.05. This result suggests there is statistical evidence to reject the null hypothesis in favor of the alternative hypothesis, indicating a **potential** disparity in the model's performance between healthy and non-healthy recipes. However, this does not irrefutably prove bias; it simply implies that, under the framework of our test, there is a **likelihood** that the model performs differently for the two groups.

Given these findings, we might consider examining the model further to understand the sources of this discrepancy and explore ways to ensure that it performs equitably across different recipe categories.

## Conclusion:

In conclusion, while our permutation test results indicate a statistically significant difference in the performance of our model between recipes tagged 'healthy' and those not, this does not conclusively establish bias. It does, however, warrant a closer look into the model's behavior and suggests that adjustments may be necessary to achieve fairer outcomes.
