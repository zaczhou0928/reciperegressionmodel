# Recipe N-steps Prediction Model 

# Framing the problem

## Prediction problem & type

This project is centered around the prediction of the number of preparation steps ‘n_steps’ necessary for completing a recipe. This is a regression problem because ‘n_steps’ is a continuous variable.

## Response Variable: 

The response variable is ‘n_steps’.

We have identified ‘n_steps’ as a critical determinant of a recipe’s intricacy and the time commitment it demands.

By accurately forecasting the number of steps a recipe requires, we can empower a diverse array of food enthusiasts – from home cooks to professional chefs, recipe developers, and restaurant owners – with valuable insights. Prior to the preparation or publication of a recipe, these individuals can benefit from understanding its complexity.

Moreover, since the number of steps serves as a proxy for the recipe’s difficulty, it is also valuable information for anyone looking for recipes that fit their time constraints and cooking experience.

## Metric for Evaluation: 

Our model's performance will be assessed using the R-squared (R²) statistic, which is particularly well-suited for regression problems. The R² metric measures the proportion of variance in the number of steps (`n_steps`) that can be explained by the independent variables in the model. 

R² is chosen over other potential metrics, such as Mean Squared Error (MSE) or Mean Absolute Error (MAE), because it provides an intuitive understanding of the model's predictive accuracy in percentage terms. A higher R² value would indicate a model that closely matches the observed data, which in this context means accurately predicting the number of steps needed for a recipe. It allows us to quantify the effectiveness of our feature selection and the model's ability to generalize from the provided data to unseen recipes, ensuring that users receive reliable step count estimates that can inform their cooking plans and expectations.

## Justification for Feature Selection:

By the time of making predictions, features such as ‘n_ingredients’, ‘nutritions’, and ‘tags’ are available because we believe those information are typically available before a recipe is prepared or published. However, the rating of a recipe, which is provided by users after cooking and tasting the dish, would not be available at the time of prediction and thus is excluded from the feature set.

## Data Cleaning:

Most data cleaning processes are done in our EDA project, which is included in this website. On top of that, we have implemented some additional data cleanings to facilitate our prediction.

Dropping/Adding columns: We dropped ‘rating’ column because it is irrelevant to our prediction problem, and we added ‘tags’ column back.

Encoding ‘tags’: We transformed the ‘tag’ column from strings into lists, which is a more usable format. This is accomplished with one-hot encoding.

# Baseline Model

In the baseline model, we employed a linear regression algorithm encapsulated within a pipeline to ensure a seamless transformation of features and training of the model. The pipeline integrates preprocessing steps for both numerical and categorical data, followed by the regression model for prediction.

## Features in the Model:

Our baseline model includes two features: ‘tags’ and ‘n_ingredients’. The two features are categorical nominal and quantitative discrete features respectively.

## Feature Encoding:

### One-Hot Encoding for 'tags': 

To incorporate the categorical 'tags' data into our model, we first transformed it with a function that identifies whether a recipe is tagged as 'easy'. We then used one-hot encoding to convert this binary classification into a numeric format that our linear regression model can interpret.

### Log Transformation for 'n_ingredients': 

Considering that the number of ingredients ('n_ingredients') might have a nonlinear relationship with the number of steps, we applied a log transformation to this feature, which can help in stabilizing variance and normalizing the distribution. 

## Model Performance:

The model's performance was evaluated using the R-squared metric, and the results are as follow:
R-Squared for training dataset: 0.22499407649086567
R-Squared for testing dataset: 0.22170387130263502

These scores indicate that around 22% of the variance in the number of steps can be explained by the model for both the training and testing sets.

## Evaluation of Model Quality:

Based on the resulting metric, our baseline model seems not ‘good’ enough. R-squared value of approximately 0.22 may suggest that there is room for improvement, as our model currently explains only a limited portion of the variance in the response variable. 

However, we believe that determining whether the model is "good" is subjective and depends on the specific context and requirements of the task. For a baseline model, such a result is not unusual, and the goal will be to enhance this performance in the subsequent iterations of the model by engineering additional features, tuning hyperparameters, and possibly exploring more complex modeling techniques.

The relatively similar R-squared scores for both the training and testing datasets suggest that our model is generalizing adequately to unseen data, which is a positive indication that the model is not overfitting the training data.

# Final Model

In the final model development phase, we experimented with various features in conjunction with linear regression to determine their predictive power. These experiments included:

Experiment 1 (Baseline + Log Minutes + Calories): This model expanded upon the baseline by integrating a logarithmic transformation of the 'minutes' feature, which aimed to smooth out the skewness often present in time-related data, and the 'calories' feature, to investigate a potential correlation with the complexity of the recipe.

Experiment 2 (Baseline + Log Minutes + Time-to-Make): In the second model, we retained the log transformation for 'minutes' and introduced the 'time-to-make' tag as a categorical variable. This tag could imply the relative quickness or lengthiness of a recipe's preparation time, which we theorized might have a relationship with the number of steps involved.

Experiment 3 (Quantile Transformed Features + Tags): The third experiment diverged by applying a Quantile Transformation to both 'n_ingredients' and 'minutes', which normalizes their distribution and can reveal more subtle associations between features and the target variable. Additionally, we encoded 'easy' and 'time-to-make' tags to assess their individual contributions to predicting the number of preparation steps.


After extensive trials and evaluations with linear regression, we transitioned to the Random Forest Regressor for our final model. This decision was informed by the ensemble method's superior ability to handle the complexity and non-linearity in the data—a prowess that was not fully realized by linear regression despite the various feature transformations attempted.

## Features Added:

Finally, in our final model with the Random Forest Regressor, we decide to include the two features: ‘minutes’ and ‘time-to-make’, on top of the features we chose in the baseline model.

These two features are chosen because we believe that they represent logical facets of the data-generating process. A recipe's complexity and the time required to complete it are intuitively connected to the number of steps it contains. 

In particular, we selected 'minutes' as a feature under the premise that a direct and positive correlation exists between the duration needed to cook a recipe and the number of steps involved—the assumption being that recipes requiring more time are likely to have a greater number of steps. 

Similarly, the 'time-to-make' tag was incorporated based on the hypothesis that recipes marked with this tag emphasize the significance of cooking duration. This tag potentially serves as an indicator of recipes where time commitment is a notable consideration, making it a meaningful attribute for analysis.

## Feature Engineering:

Quantile Transformation of 'n_ingredients' and 'minutes': In our trials and experiments, we found that applying quantile transformation on these two variables makes the linear regression model perform better. Therefore, we changed the log transformation, which we used for our baseline model, to quantile transformation for our Random Forest Regression model.

One-Hot Encoding of Tags: We extract ‘'time-to-make '' from the ‘tags’ column and one-hot encoded it.

## Hyperparameters

The hyperparameter tuning was methodically conducted using GridSearchCV with 4 folds, and we tuned a combination of the max depth, number of estimators, and the criterion for the Random Forest to find a combination that led a model that generalized the best to unseen data.

The best-performing hyperparameters were:

max_depth: 8, controlling the depth of each tree and thus preventing the model from becoming overly complex.
n_estimators: 475, determining the number of trees in the forest and providing a balance between model accuracy and computational efficiency.
criterion: 'friedman_mse', a modification of the mean squared error that is more suitable for boosting and tends to give better results.

These parameters were selected because they balance the bias-variance trade-off and enhance the model's generalization capabilities.

## Performance Improvement

The performance of our final model is as follow:
 
R-Squared for training dataset: 0.3342440985781123
R-Squared for testing dataset: 0.3338380115944084

The performance demonstrates a clear improvement over the baseline model. With an R-Squared score of approximately 0.33 on both the training and testing datasets, the final model accounts for a third of the variance in the number of recipe steps, a substantial increase from the baseline model's 0.22. This improvement indicates that the additional features and the robustness of the Random Forest Regressor have captured more of the underlying patterns within our data.
