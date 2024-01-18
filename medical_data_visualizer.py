import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = df['weight'] / (df['height'] / 100)**2
df['overweight'] = np.where(df['overweight'] > 25, 1, 0)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = np.where(df['cholesterol'] == 1, 0, 1)
df['gluc'] = np.where(df['gluc'] == 1, 0, 1)


# Draw Categorical Plot
def draw_cat_plot():
  # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
  df_cat = df.drop(
      columns=['id', 'age', 'sex', 'height', 'weight', 'ap_hi', 'ap_lo'])
  df_cat = df_cat.melt(id_vars='cardio').value_counts().to_frame()
  df_cat = df_cat.rename(columns={'count': 'total'})

  # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
  df_cat = df_cat.groupby(['cardio', 'variable',
                           'value']).value_counts().to_frame()
  df_cat = df_cat.drop(columns=['count'])

  # Draw the catplot with 'sns.catplot()'

  # Get the figure for the output
  fig = sns.catplot(data=df_cat,
                    x='variable',
                    y='total',
                    kind='bar',
                    hue='value',
                    col='cardio').fig

  # Do not modify the next two lines
  fig.savefig('catplot.png')
  return fig


# Draw Heat Map
def draw_heat_map():
  # Clean the data
  df_heat = df[(df['ap_lo'] <= df['ap_hi'])
               & (df['height'] >= df['height'].quantile(0.025))
               & (df['height'] <= df['height'].quantile(0.975))
               & (df['weight'] >= df['weight'].quantile(0.025))
               & (df['weight'] <= df['weight'].quantile(0.975))]
    
  # Calculate the correlation matrix
  corr = df_heat.corr()

  # Generate a mask for the upper triangle
  mask = np.zeros_like(corr)
  mask[np.triu_indices_from(mask)] = True

  # Set up the matplotlib figure
  fig, ax = plt.subplots(figsize=(8, 8))

  # Draw the heatmap with 'sns.heatmap()'
  sns.heatmap(data=corr,
              center=0,
              annot=True,
              fmt='.1f',
              mask=mask,
              vmin=-.16,
              vmax=.32,
              square=True,
              linewidths=0.5,
              cbar=True,
              cbar_kws={
                  'fraction': .025,
                  'spacing': 'uniform',
                  'ticks': [-0.08, 0.00, 0.08, 0.16, 0.24],
                  'format': '%.2f'
              })

  # Do not modify the next two lines
  fig.savefig('heatmap.png')
  return fig
