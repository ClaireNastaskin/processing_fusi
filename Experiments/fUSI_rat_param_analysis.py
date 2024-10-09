from pathlib import Path
import numpy as np
import pandas as pd
import json

import nibabel as nib
import scipy.stats
from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_design_matrix
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

root = Path('fUS_data') / 'fUSI_Rat_BIDS'

glm_path = root / 'derivatives' / 'glm'
plot_dir = glm_path / 'plots'
plot_dir.mkdir(exist_ok=True)

df = pd.read_csv(glm_path / 'experiment_data_proc.csv')
df['shifts'] = [eval('np.array(' + shift + ')')
                for shift in df['shifts']]
df['max_zmaps'] = [eval('np.array(' + max_zmaps + ')')
                   for max_zmaps in df['max_zmaps']]

df2 = df.copy()
df = df[~df['empty_zmap']]

# %%
# Plot max z-scores
datasets = list(np.unique(df['dataset']))
subjects = list(np.unique(df['rat']))
fig, axs = plt.subplots(len(datasets), len(subjects), figsize=(8, 8))
for i, dataset in enumerate(datasets):
    for j, subject in enumerate(subjects):
        axs[i, j].set_title(f'{dataset} rat {subject}')
axs[-1, 0].set_xlabel('Time Shift')
axs[-1, 0].set_ylabel('Maximum z-score')
for _, row in df.iterrows():
    ax = axs[datasets.index(row['dataset']),
             subjects.index(row['rat'])]
    ax.plot(row['shifts'], row['max_zmaps'])
fig.tight_layout()
for ext in ('png', 'eps'):
    ax.figure.savefig(plot_dir / f'max_zmaps_all.{ext}', dpi=300)
plt.close(fig)

ax = sns.displot(df, x="max_zmap2", hue="dataset", palette='tab10')
for ext in ('png', 'eps'):
    ax.figure.savefig(plot_dir / f'max_zmaps.{ext}', dpi=300)
plt.close(ax.figure)

# %%
# Plot session time shifts
ax = sns.displot(df, x="shift", hue="dataset", palette='tab10')
for ext in ('png', 'eps'):
    ax.figure.savefig(plot_dir / f'shifts.{ext}', dpi=300)
plt.close(ax.figure)

# %%
# Plot interaction
ax = sns.kdeplot(df, x="shift", y="max_zmap2", hue="dataset",
                 palette='tab10')
ax.figure.show()
for ext in ('png', 'eps'):
    ax.figure.savefig(plot_dir / f'shifts_max_zmaps.{ext}', dpi=300)
plt.close(ax.figure)

# %%
# Experiment plots: For each experiment, use the session name to find
# what has been changed relative to the others, then plot the differences
# in those parameters.
df_comp = pd.DataFrame(
    columns=['rat', 'dataset', 'plane', 'run', 'n_tissue_components',
             'comparison', 'sequence', 'values', 'max_zmaps'])
df_gb = df.groupby(['rat', 'dataset', 'plane', 'run', 'n_tissue_components'])
generic_max_zmaps = dict()
for group, df_group in df_gb:
    generic = [idx for idx in df_group.index if
               'generic' in df_group.loc[idx, 'sequence']]
    if not generic or len(df_group) < 3:
        print(f'Incomplete group {group}:#{len(df_group)} '
              f'generic: {len(generic)} skipping')
        continue
    assert len(generic) == 1
    generic = generic[0]
    generic_max_zmaps.update({idx: df_group.loc[generic, 'max_zmap2']
                             for idx in df_group.index})
    # find names with one factor different
    comps = dict()  # comparisons
    for i, idx in enumerate(df_group.index):
        if idx == generic:
            continue
        parts1 = df_group.loc[idx, 'sequence'].split('_')
        assert parts1[0] == 'RF'
        parts1 = parts1[1:]
        parts1[0] += 'RF'
        for idx2 in df_group.index[1:]:
            if idx2 == generic:
                continue
            parts2 = df_group.loc[idx2, 'sequence'].split('_')
            assert parts2[0] == 'RF'
            parts2 = parts2[1:]
            parts2[0] += 'RF'
            assert len(parts1) == len(parts2)
            diff_idx = [j for j in range(len(parts1))
                        if parts1[j] != parts2[j]]
            if len(diff_idx) == 1: # we have a single difference
                name = '_'.join([part for j, part in enumerate(parts1)
                                    if j != diff_idx[0]])
                diff = ''.join([letter for letter in parts1[diff_idx[0]]
                                if not letter.isdigit() and
                                letter not in ('-', '.', '+')])
                value1 = ''.join([letter for letter in parts1[diff_idx[0]]
                                    if letter.isdigit() or
                                    letter in ('-', '.', '+')])
                value2 = ''.join([letter for letter in parts2[diff_idx[0]]
                                    if letter.isdigit() or
                                    letter in ('-', '.', '+')])
                if name not in comps:
                    comps[(name, diff)] = set()
                comps[(name, diff)].add((idx, value1))
                comps[(name, diff)].add((idx2, value2))
    for (name, diff), picks in comps.items():
        picks = list(picks)
        order = np.argsort([value for _, value in picks])
        picks = [picks[j] for j in order]
        df_comp.loc[len(df_comp.index)] = \
            group + (diff,
                     [df_group.loc[idx, 'sequence'] for idx, _ in picks],
                     [value for _, value in picks],
                     [df_group.loc[idx, 'max_zmap2'] for idx, _ in picks])

# plots
for comp in np.unique(df_comp['comparison']):
    fig, ax = plt.subplots()
    ax.set_title(comp)
    for i, row in df_comp[df_comp['comparison'] == comp].iterrows():
        ax.plot(row['values'], row['max_zmaps'])
    ax.set_xlabel('Parameter Value')
    ax.set_ylabel('Max T-Statistic')
    for ext in ('png', 'eps'):
        fig.savefig(plot_dir / f'{comp}_AB.{ext}', dpi=300)
    plt.close(fig)

# stats + plots
df_comp['values'] = [tuple(values) for values in df_comp['values']]
df_comp_gb = df_comp.groupby(['comparison', 'values'])
for (comp, values), group in df_comp_gb:
    if len(group) < 3:
        print(f'{comp} {values} not enough data points, {len(group)} found')
        continue
    max_zmaps = {value: list() for value in values}
    for _, row in group.iterrows():
        for value, max_zmap in zip(row['values'], row['max_zmaps']):
            max_zmaps[value].append(max_zmap)
    res = scipy.stats.ttest_rel(*max_zmaps.values())
    print(name, values, res)
    fig, ax = plt.subplots()
    comp_means = np.mean(list(max_zmaps.values()), axis=1)
    comp_stds = np.std(list(max_zmaps.values()), axis=1)
    ax.bar(values, comp_means, yerr=comp_stds)
    ax.set_xlabel(comp)
    ax.set_ylabel('Max Z-score')
    if res.pvalue < 0.05:
        y = np.max(comp_means + comp_stds)
        ax.plot([0, 0, 1, 1], [y * 1.05, y * 1.1, y * 1.1, y * 1.05],
                color='black')
        ax.text(0.5, y * 1.15,
                'p<0.001' if res.pvalue < 0.001 else f'p={res.pvalue.round(3)}',
                ha='center')
        ax.set_ylim([ax.get_ylim()[0], y * 1.2])
    values_name = '-'.join(values)
    for ext in ('png', 'eps'):
        fig.savefig(plot_dir / f'{comp}_{values_name}_bar.{ext}', dpi=300)
    plt.close(fig)


df['max_zmap_rel'] = [df.loc[idx, 'max_zmap2'] - generic_max_zmaps[idx]
                       if idx in generic_max_zmaps else np.nan
                       for idx in df.index]
group_max_zmaps = {idx: df_group['max_zmap2'].mean()
                    for _, df_group in df_gb for idx in df_group.index}
df['max_zmap_rel2'] = [df.loc[idx, 'max_zmap2'] - group_max_zmaps[idx]
                        if idx in group_max_zmaps else np.nan
                        for idx in df.index]

# %%
# Count plot better than generic
(glm_path / 'plots' / 'generic_comp').mkdir(parents=True, exist_ok=True)
for col in df.columns[7:-10]:
    ax = sns.countplot(df[df['max_zmap_rel'] > 0], x=col)
    ax.figure.savefig(glm_path / 'plots' / 'generic_comp' / f'{col}.png')
    plt.close(ax.figure)


# %%
# Overall plots. For all the experiments, look at the trend of z-scores
# relative to generic across parameters.
for factor in ('frequency', 'az_aperture', 'el_aperture', 'gain',
               'n_tissue_components'):
    if isinstance(np.unique(df[factor])[0], str):
        grid = sns.catplot(df, x=factor, y="max_zmap2",
                           row="rat", col="dataset",
                           palette="tab10")
    else:
        grid = sns.displot(df, x="max_zmap2", hue=factor,
                           row="rat", col="dataset",
                           palette="tab10", kde=True)
    for ext in ('png', 'eps'):
        grid.figure.savefig(plot_dir / f'{factor}_factor.{ext}', dpi=300)
    plt.close(fig)

# %%
# Check for generic okay, no GLM zmap
df_gb2 = df.groupby(['rat', 'dataset', 'plane', 'run', 'n_tissue_components'])
killed_zmap = list()
for group, df_group in df_gb2:
    generic = [idx for idx in df_group.index if
               'generic' in df_group.loc[idx, 'sequence']]
    if not generic or len(df_group) < 3:
        print(f'Incomplete group {group}:#{len(df_group)} '
              f'generic: {len(generic)} skipping')
        continue
    generic = generic[0]
    # check generic in keep group
    if df_group.loc[generic, 'power_doppler_path'] not in \
            list(df['power_doppler_path']):
        continue
    # check if each row is in keep group
    for i, row in df_group.iterrows():
        if row['power_doppler_path'] not in list(df['power_doppler_path']):
            killed_zmap.append(i)

# empty, don't plot
print(killed_zmap)

df.to_csv(glm_path / 'experiment_data_rel.csv')

# %%
# Second-level analysis
design_matrix = df[
    ['dataset', 'rat', 'run', 'plane', 'n_tissue_components',
     'angles', 'frequency', 'az_aperture', 'el_aperture',
     'gain', 'power_mode', 'cycles']
]
drop = list()
for col in design_matrix.columns:
    if np.unique(df[col]).size == 1:
        drop.append(col)
design_matrix = design_matrix.drop(drop, axis='columns')

# handle categorical variables
dummy_cols = ['dataset', 'rat', 'plane', 'gain', 'el_aperture']
for col in dummy_cols:
    design_matrix = pd.concat(
        [design_matrix,
         pd.get_dummies(design_matrix[col], prefix=col,
                        drop_first=True)],
        axis='columns')
    design_matrix = design_matrix.drop(col, axis='columns')
design_matrix['const'] = 1

# handle interactions
interactions = ['gain', 'frequency', 'el_aperture']
for i, interaction0 in enumerate(interactions):
    for col0 in design_matrix:
        if not col0.startswith(interaction0):
            continue
        for interaction1 in interactions[i + 1:]:
            for col1 in design_matrix:
                if not col1.startswith(interaction1):
                    continue
                design_matrix = pd.concat([
                    design_matrix,
                    pd.DataFrame({f'{col0}*{col1}': design_matrix[col0] * design_matrix[col1]})
                ], axis='columns')

model = sm.GLM(df['max_zmap2'], design_matrix.astype(float))
res = model.fit()
print(res.summary())
