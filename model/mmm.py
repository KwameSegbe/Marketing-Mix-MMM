import pandas as pd
import numpy as np

# from pymc_marketing.mmm.delayed_saturated_mmm import DelayedSaturatedMMM
# from pymc_marketing.mmm import DelayedSaturatedMMM
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_marketing.paths import data_dir
import pandas as pd
import pytimetk as tk
import os

import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
os.makedirs("outputs", exist_ok=True)

# Generate Robyn-equivalent simulated dataset
np.random.seed(123)
n = 208  # 4 years of weekly data

dates = pd.date_range(start="2015-11-23", periods=n, freq="W")

data = pd.DataFrame({
    "DATE": dates,
    "revenue":        np.random.normal(25000, 5000, n).clip(5000),
    "tv_S":           np.random.uniform(0, 10000, n),
    "ooh_S":          np.random.uniform(0, 5000, n),
    "print_S":        np.random.uniform(0, 3000, n),
    "facebook_I":     np.random.uniform(0, 200000, n),
    "facebook_S":     np.random.uniform(0, 8000, n),
    "search_S":       np.random.uniform(0, 6000, n),
    "search_clicks_P":np.random.uniform(0, 50000, n),
    "competitor_sales_B": np.random.normal(50000, 10000, n).clip(0),
    "events":         np.random.choice(["0", "event1", "event2"], n),
    "newsletter":     np.random.randint(0, 50000, n),
})

data.to_csv("data/dc_simulated_data.csv", index=False)
print(data.shape)
print(data.head())


# DATA DEFINITION --------
# use robys dataset

data = pd.read_csv("data/dc_simulated_data.csv", parse_dates=['DATE'])

# data.glimpse()

# Data Definition:
# - revenue          = sales, target variable
# - ooh_s            = Out of Home (Billboards, etc.) Spend
# - tv_s             = Television Spend
# - print_s          = Print Media (Newspapers, Magazines) Spend
# - facebook_i       = Facebook Ads, Impressions
# - facebook_s       = Facebook Ads, Spend
# - search_s         = Google Search Ads, Search Spend
# - search_clicks_p  = Google Search Ads Performance (Number of Clicks)
# - competitor_sales_b = Competitor Sales Baseline

# EXPLORATORY DATA ANALYSIS --------

# 1.0 EXPLORATORY DATA ANALYSIS --------

df = data.copy()

df.columns = [col.lower() for col in df.columns]

print(df.columns.tolist())

# Visualize Revenue and Spend by Marketing Channels
fig = df \
    .melt(
        id_vars=["date"],
        value_vars=["revenue", "ooh_s", "tv_s", "print_s",
                    "facebook_i", "facebook_s", "search_s",
                    "search_clicks_p", "competitor_sales_b"]
    ) \
    .groupby("variable") \
    .plot_timeseries(
        "date", "value",
        color_column="variable",
        facet_ncol=2,
        width=600,
        height=800,
        legend_show=False
    )

# fig.show()

# Total Spend and Revenue Analysis
total_spend = df[['tv_s', 'ooh_s', 'print_s', 'facebook_s',
                  'search_s']].sum(axis=0).sum()

total_revenue = df['revenue'].sum()

print(f"Total Spend: {total_spend:,.0f}")
print(f"Total Revenue: {total_revenue:,.0f}")


# Total Spend and Revenue Analysis
total_spend = df[['tv_s', 'ooh_s', 'print_s',
                  'facebook_s', 'search_s']].sum(axis=0).sum()

total_revenue = df['revenue'].sum()

print(f"Total Revenue / Total Spend: {total_revenue / total_spend:.2f}")

# Monthly Ad Spend Analysis
median_spend = df[['tv_s', 'ooh_s', 'print_s',
                   'facebook_s', 'search_s']].median()

mean_spend = df[['tv_s', 'ooh_s', 'print_s',
                 'facebook_s', 'search_s']].mean()
print(f"Mean Spend: {mean_spend.sum():,.0f}")

df[['tv_s', 'ooh_s', 'print_s', 'facebook_s',
    'search_s']] \
    .describe() \
    .apply(lambda x: x.apply(lambda y: "{:,.0f}".
    format(y)))

# print(f"Mean Spend: {mean_spend.sum():,.0f}")
print(df[['tv_s', 'ooh_s', 'print_s', 'facebook_s', 'search_s']]
    .describe()
    .applymap(lambda y: "${:,.0f}".format(y))
    .to_string())

# QUESTION - IS THIS THE MOST PROFITABLE BASED ON
# THE CONTRIBUTION OF EACH CHANNEL?
# - LET'S TACKLE THIS QUESTION BY BUILDING A
# MARKETING MIX MODEL

# 2.0 FEATURE ENGINEERING --------

# Time Series Features
df_features = df \
    .assign(
        year       = lambda x: x["date"].dt.year,
        month      = lambda x: x["date"].dt.month,
        dayofyear  = lambda x: x["date"].dt.dayofyear,
    ) \
    .assign(
        trend = lambda x: df.index,
    )

df_features = df_features[['date', 'revenue',
    'tv_s', 'ooh_s', 'print_s', 'facebook_s',
    'search_s', 'trend', 'year', 'month',
    'dayofyear']]

# df_features.glimpse()

# 3.0 MODEL SET UP --------
# - Reference: https://www.pymc-marketing.io/en/
#   stable/notebooks/mmm/mmm_example.html#id1
# - DelayedSaturatedMMM handles scaling
#   transformations internally
# - Uses MaxAbsScaler transformer from
#   sklearn
# - Specify the priors in the scaled space i.e.


# Create Priors from Business Knowledge
total_spend_per_channel = df_features[['tv_s', 'ooh_s', 'print_s',
                                        'facebook_s', 'search_s']].sum(axis=0)

spend_proportion = total_spend_per_channel / total_spend_per_channel.sum()

HALFNORMAL_SCALE = 1 / np.sqrt(1 - 2 / np.pi)

n_channels = 5

prior_sigma = HALFNORMAL_SCALE * n_channels * spend_proportion

prior_sigma.tolist()

# Create a Model Specification
X = df_features.drop("revenue", axis=1)
y = df_features["revenue"]

print(X)
print(y)

# Default Model Configuration
dummy_model = MMM(
    date_column = "date",
    channel_columns = ['tv_s', 'ooh_s', 'print_s', 'facebook_s', 'search_s'],
    control_columns = [
        "trend",
        "year",
        "month",
    ],
    adstock = GeometricAdstock(l_max=8),
    saturation = LogisticSaturation(),
)

my_model_config = {
    'intercept': {
        'dist': 'Normal',
        'kwargs': {
            'mu': 0,
            'sigma': 2
        }
    },
    'beta_channel': {
        'dist': 'HalfNormal',
        'kwargs': {
            'sigma': 2
        }
    },
    # 'beta_channel': {
    #     'dist': 'LogNormal',
    #     'kwargs': {
    #         'mu': np.array([5,1]),
    #         'sigma': prior_sigma.to_numpy()
    #     }
    # },
    'likelihood': {
        'dist': 'Normal',
        'kwargs': {
            'sigma': {
                'dist': 'HalfNormal',
                'kwargs': {
                    'sigma': 2
                }
            }
        }
    },
    'alpha': {
        'dist': 'Beta',
        'kwargs': {'alpha': 1, 'beta': 3}
    },
    'lam': {
        'dist': 'Gamma',
        'kwargs': {'alpha': 3, 'beta': 1}
    },
    'gamma_control': {
        'dist': 'Normal',
        'kwargs': {'mu': 0, 'sigma': 2}
    },
    'gamma_fourier': {
        'dist': 'Laplace',
        'kwargs': {'mu': 0, 'b': 1}
    },
}

my_sampler_config = {
    "progressbar": True,
    "cores": 3,
}

# * DelayedSaturatedMMM Model

# * MMM Model
mmm = MMM(
    model_config = my_model_config,
    sampler_config = my_sampler_config,
    date_column = "date",
    channel_columns = ['tv_s', 'ooh_s', 'print_s', 'facebook_s', 'search_s'],
    control_columns = [
        "trend",
        "year",
        "month",
    ],
    adstock = GeometricAdstock(l_max=8),
    saturation = LogisticSaturation(),
    yearly_seasonality = 2,
)

mmm.model_config

mmm.default_model_config

# 4.0 MODEL FITTING --------
# - Model fitting (takes 20 minutes with 1 core)

# Fit the Model
# 4.0 MODEL FITTING --------

if __name__ == '__main__':

    # Fit the Model - SKIP, already saved
    # mmm.fit(
    #     X, y,
    #     target_accept=0.95,
    #     random_seed=888,
    #     draws=200,
    #     tune=500,
    #     chains=1,
    #     max_treedepth=15,
    # )

    # Save - SKIP, already saved
    # mmm.save("model/mmm_adspend_model.pkl")

    # Load already saved model
    loaded_mmm = MMM.load("model/mmm_adspend_model.pkl")

    # Model Summary (arviz object)
    print(loaded_mmm.idata)

    # * 5.0 POST MODEL ANALYSIS & VISUALIZATIONS --------

    # Plot Components Contributions
    fig = loaded_mmm.plot_components_contributions()
    fig.savefig("outputs/components_contributions.png", dpi=150, bbox_inches="tight")
    # fig.show()

    # Plot Graphical MMM Model
    loaded_mmm.graphviz().render("outputs/mmm_graph", format="png", view=True)

    # Plot Direct Contribution Curves
    fig = loaded_mmm.plot_direct_contribution_curves()
    fig.savefig("outputs/direct_contribution_curves.png", dpi=150, bbox_inches="tight")
    # fig.show()

    # Plot Channel Contributions - matplotlib
    fig = loaded_mmm.plot_channel_contribution_grid(
        start=0,
        stop=1.3,
        num=12,
        absolute_xrange=True
        )
    fig.savefig("outputs/channel_contribution_grid.png", dpi=150, bbox_inches="tight")

    print("All plots saved to outputs folder!")


# * 6.0 BUSINESS APPLICATIONS: RETURN ON ADSPEND --------

# Estimate Return on Ad Spend (ROAS) by Channel
get_mean_contributions_over_time_df = loaded_mmm \
    .compute_mean_contributions_over_time(original_scale=True)

channel_contribution_original_scale = loaded_mmm \
    .compute_channel_contribution_original_scale()

roas_samples = (
    channel_contribution_original_scale.stack(sample=["chain", "draw"]).sum("date")
    / X[['tv_s', 'ooh_s', 'print_s', 'facebook_s', 'search_s']].sum().to_numpy()[..., None]
)


# Visualize Estimated ROAS by Channel
# Visualize Estimated ROAS by Channel
fig, ax = plt.subplots(figsize=(15, 6))
for channel in ['tv_s', 'ooh_s', 'print_s', 'facebook_s',
                'search_s']:
    sns.histplot(
        roas_samples.sel(channel=channel).to_numpy(), 
        binwidth=0.05, 
        alpha=0.3, 
        kde=True, 
        ax=ax, 
        legend=True, 
        label=channel
    )
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax.set(title="Posterior ROAS Distribution", xlabel="ROAS")
fig.savefig("outputs/roas_distribution.png", dpi=150, bbox_inches="tight")

# ROAS Summary
roas_df = roas_samples.to_dataframe(name="roas")

roas_mean = roas_df.groupby("channel").mean()
print(roas_mean)

roas_summary = roas_df.groupby("channel")['roas'].describe(percentiles=[0.025, 0.975])
print(roas_summary)

# Save ROAS Summary as CSV
roas_mean.to_csv("outputs/roas_mean.csv")
roas_summary.to_csv("outputs/roas_summary.csv")

print("ROAS outputs saved!")



# # * 7.0 BUSINESS APPLICATIONS: BUDGET ALLOCATION --------
# # - PROBLEM:
# # 1. SOME CHANNELS PERFORM BETTER THAN OTHERS
# # 2. DECAY IN PERFORMANCE OF MARKETING CHANNELS
# # - SOLUTION: OPTIMIZE BUDGET ALLOCATION FOR MAXIMUM CONTRIBUTION
# # - Reference: https://www.pymc-marketing.io/en/stable/notebooks/
# #   mmm/mmm_budget_allocation_example.html

# # * 7.1 Response Curves (2 Types): Sigmoid and Michaelis-Menten

# # No Curve
# response_curve_fig = loaded_mmm.plot_direct_contribution_curves()
# response_curve_fig.show()

# # Sigmoid Shown
# sigmoid_response_curve_fig = mmm.plot_direct_contribution_curves(
#     show_fit=True
# )
# sigmoid_response_curve_fig.show()

# # Michaelis-Menten Shown
# sigmoid_response_curve_fig = mmm.plot_direct_contribution_curves(
#     show_fit=True, method='michaelis-menten'
# )
# sigmoid_response_curve_fig.show()

# # Curve Parameters
# sigmoid_params = mmm \
#     .compute_channel_curve_optimization_parameters_original_scale(
#     method='sigmoid')

# menten_params = mmm \
#     .compute_channel_curve_optimization_parameters_original_scale(
#     method='michaelis-menten')

# # * 7.2 Budget Allocation for Maximum Contribution
# # - We aim to optimize the allocation of budgets across multiple
# #   channels to maximize the overall contribution to key performance
# #   indicators (KPIs), such as sales or conversions.
# # - Each channel has its own sigmoid or michaelis-menten curve,
# #   representing the relationship between the amount spent and the
# #   resultant performance.


