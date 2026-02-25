# Polynomial Regression

**I'm currently learning machine learning, and this is my first little project on polynomial regression.** 

Nothing fancy — just wanted to understand how linear models can handle curved data. Turns out, the trick is to add squared terms as features. This notebook is basically my playground for figuring out how it all works.

## What's this about?

I generated some fake data that follows a quadratic pattern (y = 2 + X + 0.5X²), threw in some random noise to make it realistic, and tried to recover the original relationship using a degree-2 polynomial regression.

**What's inside:**
- Creating synthetic data (because real data is messy when you're just learning)
- Splitting into train/test (gotta check if it actually generalizes)
- Using scikit-learn's Pipeline with PolynomialFeatures (makes the code so much cleaner)
- Looking at MSE, R², and the learned coefficients (did it find the 0.5 and 1?)
- Residual plots (because metrics don't tell the whole story)

## The Numbers

After running the model, here's what came out:

| Metric       | Value          | What it means                                         |
|--------------|----------------|-------------------------------------------------------|
| MSE          | 0.7748         | Average squared error - pretty decent given the noise |
| R²           | 0.8339         | Explains ~83% of variance in test data                |
| Coefficients | [1.028, 0.503] | Should be [1, 0.5] - got really close!                |
| Intercept    | 1.9262         | True value was 2.0 - off by ~0.07                     |

Not bad for a simple model with noisy data, right? The coefficients are almost exactly the true values I used to generate the data.

## What it looks like

The notebook has a couple of plots:
- **Main plot**: Shows the training points (blue), test points (green), and the model's prediction (red curve). You can visually see the curve fits nicely.
- **Residual plots**: These are diagnostic - they show if the model is making systematic errors. The random scatter around zero tells me the model isn't missing any obvious patterns.

## What I learned

Polynomial regression isn't magic - it just adds x², x³ as new columns and runs linear regression

Choosing the right degree matters a lot. Degree 1 (plain linear) would miss the curve entirely. Degree 15 would probably go crazy trying to fit every noise point.

Pipeline in scikit-learn is awesome - keeps preprocessing and modeling together so you don't mess up.

## Limitations / What's missing

Didn't try different degrees (maybe degree 3 would overfit?)

No cross-validation - just a single train/test split

The data is synthetic - real-world data is never this clean

Could add confidence intervals to show uncertainty

## Want to try it?

```bash

git clone https://github.com/uwaspwned/polynomial-regression.git
cd polynomial-regression
pip install -r requirements.txt
jupyter notebook polynomial_regression.ipynb
