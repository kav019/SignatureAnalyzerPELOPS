import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Union
import numpy as np
import re

from ..utils import compl, sbs_annotation_converter
from ..spectra import context96, context78, context83

def stacked_bar(H: pd.DataFrame, figsize: tuple = (8,8)):
    """
    Plot stacked barchart & normalized stacked barchart.
    --------------------------------------
    Args:
        * H: matrix output from NMF
        * figsize: size of figure (int,int)

    Returns:
        * figure

    Example usage:
        plot_bar(H)
    """

    sig_vs_color = {'SBS1': (0.5075427220776314, 0.48983070987251776, 0.5231722949335792),
 'SBS2': (0.9981113891476213, 0.4883858125036441, 0.964800238937131),
 'SBS3': (0.649003392020299, 0.9881312331645069, 0.4926471847334411),
 'SBS4': (0.4996659062811246, 0.7561004259905724, 0.9664903124446428),
 'SBS5': (0.9474571668317493, 0.6970050269302019, 0.48664998554375666),
 'SBS6': (0.6473240871930996, 0.4831016896667449, 0.9304143520281729),
 'SBS7a': (0.9945022838375985, 0.9892603234025525, 0.4997689744973316),
 'SBS7b': (0.6686955509516681, 0.9998081484182196, 0.9791630927471403),
 'SBS7c': (0.864246367351471, 0.741102816325034, 0.8730131426780279),
 'SBS7d': (0.6424475208029509, 0.7321298959499095, 0.5950030378416087),
 'SBS8': (0.8731310682825001, 0.4739681638973685, 0.641399490689568),
 'SBS9': (0.8286249723430873, 0.9485729835626333, 0.7376874772972845),
 'SBS10a': (0.5050049158986379, 0.9162541243982796, 0.7244411831812033),
 'SBS10b': (0.48682514889349265, 0.6044243881616924, 0.8190237217928684),
 'SBS11': (0.7482725210793446, 0.6002767186366353, 0.7633036714734334),
 'SBS12': (0.6795388986053303, 0.8359582955709574, 0.8399583237064875),
 'SBS13': (0.993694629620879, 0.8397930210558615, 0.6931714127852786),
 'SBS14': (0.745031139294868, 0.5860508548097905, 0.50562779522088),
 'SBS15': (0.6716258514346118, 0.6517549777506593, 0.9735137766772968),
 'SBS16': (0.8235442275973339, 0.8402875912261593, 0.501062000806576),
 'SBS17a': (0.9717102534066694, 0.6020398296608485, 0.760751818327542),
 'SBS17b': (0.604427390298564, 0.48571702878005407, 0.7029187914367854),
 'SBS18': (0.48325724345549903, 0.8115142072875402, 0.4762792864655845),
 'SBS19': (0.4935504431551681, 0.9194156406075932, 0.9771055128523108),
 'SBS20': (0.8260419459512497, 0.9081327234684671, 0.966695040495792),
 'SBS21': (0.8289332237830808, 0.5467672537033088, 0.9805898505377405),
 'SBS22': (0.5313953117310168, 0.7546908971037839, 0.7610414556700299),
 'SBS23': (0.4821995965591, 0.6545646114525481, 0.48902064199393624),
 'SBS24': (0.9678335751679419, 0.6473774781895023, 0.9831296953725248),
 'SBS25': (0.8403689804515929, 0.7408201907496069, 0.6611756505445014),
 'SBS26': (0.6534773825860357, 0.9846436884809396, 0.6894918154630599),
 'SBS27': (0.4814684851537843, 0.5444676287342157, 0.9987043098017208),
 'SBS28': (0.996400467915433, 0.9806841833578662, 0.7361996443860706),
 'SBS29': (0.9827523960058161, 0.8264958337893654, 0.9847186338868784),
 'SBS30': (0.9763393064987019, 0.5065051620861762, 0.47948974175570364),
 'SBS31': (0.8285795603357811, 0.9882139001403626, 0.5420374707572978),
 'SBS32': (0.9896386723563307, 0.8471616992885598, 0.49921408812447),
 'SBS33': (0.47378322587074, 0.9890137623908979, 0.5810404223538987),
 'SBS34': (0.539711626653106, 0.6041269804231538, 0.6290674677783374),
 'SBS35': (0.6119494404484218, 0.8613269888129649, 0.6069213982907483),
 'SBS36': (0.7068086866716621, 0.7767383127360645, 0.9844630188225708),
 'SBS37': (0.8554836958704924, 0.4857308953027787, 0.8292502287500081),
 'SBS38': (0.6053421932253933, 0.9589151046088743, 0.8409938683887337),
 'SBS39': (0.6955344210415525, 0.7212456349877099, 0.7723327013983747),
 'SBS40': (0.6614061514938538, 0.48152218503475497, 0.4797031271809131),
 'SBS41': (0.8790571860014412, 0.6164354093467941, 0.6132906397570184),
 'SBS42': (0.5206781958874885, 0.48107740212400973, 0.8417496729132106),
 'SBS43': (0.9989970871489228, 0.7498934497917789, 0.8101267736300954),
 'SBS44': (0.9126344151746295, 0.9876794672191991, 0.8727230096585483),
 'SBS45': (0.7778274284876916, 0.7075843945931642, 0.4835033033845069),
 'SBS46': (0.9937390534337537, 0.4752518029037119, 0.763284803652669),
 'SBS47': (0.89696783802622, 0.8629116374455151, 0.8157561868507648),
 'SBS48': (0.7524446280026813, 0.8531693438694318, 0.6736862118408228),
 'SBS49': (0.9045242592429168, 0.9116225484972185, 0.6167441433956802),
 'SBS50': (0.6242866557844425, 0.5768206802002098, 0.8338613145689074),
 'SBS51': (0.7339446862470985, 0.5247828726831892, 0.6441276685937332),
 'SBS52': (0.4996517528125477, 0.7125448029675576, 0.6281098990792723),
 'SBS53': (0.9904086656699096, 0.6940908855929439, 0.6595406679526067),
 'SBS54': (0.8178148759989008, 0.4752950780867655, 0.4890927475093093),
 'SBS55': (0.6956942399571084, 0.8749191147673148, 0.47722049324773674),
 'SBS56': (0.7277817734573747, 0.4826610180666014, 0.7823609382966009),
 'SBS57': (0.47768122749314035, 0.8571344781010312, 0.8580279190062153),
 'SBS58': (0.6292744668023674, 0.6489097190032521, 0.47839294451256953),
 'SBS59': (0.6154649675661102, 0.8613494945239685, 0.994092800389792),
 'SBS60': (0.8640071500533965, 0.6019868117830725, 0.8570977337831751)}

    H = H.iloc[:,:-3].copy()
    H['sum'] = H.sum(1)
    H = H.sort_values('sum', ascending=False)

    fig,axes = plt.subplots(2,1,figsize=figsize, sharex=True)

    sbs_columns = [col for col in H.columns[:-1] if 'SBS' in col]
    sbs_sig_name = [x[3:] for x in sbs_columns]
    plot_colors = [sig_vs_color[x] for x in sbs_sig_name]

    H.iloc[:,:-1].plot(
        kind='bar',
        stacked=True,
        ax=axes[0],
        width=1.0,
        rasterized=True,
        color=plot_colors
    )

    axes[0].set_xticklabels([])
    axes[0].set_xticks([])
    axes[0].set_ylabel('Counts', fontsize=20)

    H_norm = H.iloc[:,:-1].div(H['sum'].values,axis=0)
    H_norm.plot(
        kind='bar',
        stacked=True,
        ax=axes[1],
        width=1.0,
        rasterized=True,
        color=plot_colors
    )

    axes[1].set_xticklabels([])
    axes[1].set_xticks([])
    axes[1].set_xlabel('Samples', fontsize=16)
    axes[1].set_ylabel('Fractions', fontsize=20)
    axes[1].get_legend().remove()
    axes[1].set_ylim([0,1])

    return fig

def _map_sbs_sigs_back(df: pd.DataFrame) -> pd.Series:
    """
    Map Back Single-Base Substitution Signatures.
    -----------------------
    Args:
        * df: pandas.core.frame.DataFrame with index to be mapped

    Returns:
        * pandas.core.series.Series with matching indices to context96
    """
    def _check_to_flip(x, ref):
        if x in ref:
            return x
        else:
            return compl(x)

    if df.index.name is None: df.index.name = 'index'
    df_idx = df.index.name

    if ">" in df.index[0]:
        # Already in arrow format
        context_s = df.reset_index()[df_idx].apply(sbs_annotation_converter)
    else:
        # Already in word format
        context_s = df.reset_index()[df_idx]

    return context_s.apply(lambda x: _check_to_flip(x, context96.keys()))

def _map_id_sigs_back(df: pd.DataFrame) -> pd.Series:
    """
        Map Back Insertion-Deletion Signatures.
        -----------------------
        Args:
            * df: pandas.core.frame.DataFrame with index to be mapped

        Returns:
            * pandas.core.series.Series with matching indices to context83
        """
    if df.index.name is None: df.index.name = 'index'
    df_idx = df.index.name

    context_s = df.reset_index()[df_idx]

    def _convert_from_cosmic(x):
        if x in context83:
            return x
        i1, i2, i3, i4 = x.split('_')
        pre = i2 if i3 == '1' else i3
        main = i1.lower() + ('m' if i2 == 'MH' else '')
        if main == 'del':
            post = str(int(i4[0]) + 1) + i4[1:]
        else:
            post = i4
        return pre + main + post

    return context_s.apply(_convert_from_cosmic)

def signature_barplot(W: pd.DataFrame, contributions: Union[int, pd.Series] = 1):
    """
    Plots signatures from W-matrix for Single-Base Substitutions
    --------------------------------------
    Args:
        * W: W-matrix
        * contributions: Series of total contributions, np.sum(H), from each
            signature if W is normalized; else, 1

    Returns:
        * fig

    Example usage:
        signature_barplot(W, np.sum(H))
    """
    W = W.copy()
    W.index = _map_sbs_sigs_back(W)

    for c in context96:
        if c not in W.index:
            W.loc[c] = 0

    W.sort_index(inplace=True)

    sig_columns = [c for c in W if c.startswith('S')]

    if isinstance(contributions, pd.Series):
        W = W[sig_columns] * contributions[sig_columns]
    else:
        W = W[sig_columns] * contributions

    n_sigs = len(sig_columns)

    context_label = []
    change_map = {'CA': [], 'CG': [], 'CT': [], 'TA': [], 'TC': [], 'TG': []}
    for p in itertools.product('ACGT', 'ACGT'):
        context = ''.join(p)
        compl_context = compl(context, reverse=True)
        context_label.append('-'.join(context))
        for key in change_map:
            if key.startswith('C'):
                change_map[key].append(key + context)
            else:
                change_map[key].append(compl(key) + compl_context)
    color_map = {'CA': 'cyan', 'CG': 'red', 'CT': 'yellow', 'TA': 'purple', 'TC': 'green', 'TG': 'blue'}

    x_coords = range(16)
    fig, axes = plt.subplots(nrows=n_sigs, ncols=6, figsize=(20, 2.5 * n_sigs), sharex='col', sharey='row')
    for row, sig in enumerate(sig_columns):
        for col, chg in enumerate(['CA', 'CG', 'CT', 'TA', 'TC', 'TG']):
            if n_sigs == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            bar_heights = W[sig].loc[change_map[chg]]
            ax.bar(x_coords, bar_heights, width=.95, linewidth=1.5, edgecolor='gray', color=color_map[chg], rasterized=True)
            ax.set_xlim(-.55, 15.55)
            if row == 0:
                ax.set_title('>'.join(chg), fontsize=18)
                if col == 0:
                    ax.text(51.2 / 16, 1.3, 'Mutational Signatures', transform=ax.transAxes,
                            horizontalalignment='center', fontsize=24)
            if row < n_sigs - 1:
                ax.tick_params(axis='x', length=0)
            else:
                ax.set_xticks(x_coords)
                ax.set_xticklabels(context_label, fontfamily='monospace', rotation='vertical')
                if col == 0:
                    ax.text(51.2 / 16, -.4, 'Motifs', transform=ax.transAxes, horizontalalignment='center', fontsize=20,
                            fontweight='bold')
            if col > 0:
                ax.tick_params(axis='y', length=0)
            if col == 5:
                ax.text(1.05, .5, sig, fontsize=14, rotation=270, transform=ax.transAxes, verticalalignment='center')

    plt.subplots_adjust(wspace=.08, hspace=.15)
    fig.text(.08, .5, 'Contributions', rotation='vertical', verticalalignment='center', fontsize=20, fontweight='bold')

    return fig

def signature_barplot_DBS(W, contributions):
    """
    Plots signatures from W-matrix for Doublet-Base Substitutions
    --------------------------------------
    Args:
        * W: W-matrix
        * contributions: Series of total contributions, np.sum(H), from each
            signature if W is normalized; else, 1

    Returns:
        * fig

    Example usage:
        signature_barplot_DBS(W, np.sum(H))
    """
    W = W.copy()
    for c in context78:
        if c not in W.index:
            W.loc[c] = 0
    W.sort_index(inplace=True)
    sig_columns = [c for c in W if c.startswith('S')]
    if isinstance(contributions, pd.Series):
        W = W[sig_columns] * contributions[sig_columns]
    else:
        W = W[sig_columns] * contributions

    n_sigs = len(sig_columns)

    ref_map = {'AC': [], 'AT': [], 'CC': [], 'CG': [], 'CT': [], 'GC': [], 'TA': [], 'TC': [], 'TG': [], 'TT': []}
    for x in W.index:
        ref_map[x[:2]].append(x)
    x_coords = {ref: range(len(sigs)) for ref, sigs in ref_map.items()}

    color_map = {'AC': '#99CCFF', 'AT': '#0000FF', 'CC': '#CCFF99', 'CG': '#00FF00', 'CT': '#FF99CC',
                 'GC': '#FF0000', 'TA': '#FFCC99', 'TC': '#FF8000', 'TG': '#CC99FF', 'TT': '#8000FF'}
    fig, axes = plt.subplots(nrows=n_sigs, ncols=10, figsize=(20, 2.5 * n_sigs), sharex='col',
                             sharey='row', gridspec_kw={'width_ratios': (3, 2, 3, 2, 3, 2, 2, 3, 3, 3)})
    for row, sig in enumerate(sig_columns):
        for col, ref in enumerate(ref_map):
            if n_sigs == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            bar_heights = W[sig].loc[ref_map[ref]]
            ax.bar(x_coords[ref], bar_heights, width=.95, linewidth=1.5, edgecolor='gray', color=color_map[ref],
                   rasterized=True)
            ax.set_xlim(-.55, x_coords[ref][-1] + .55)
            if row == 0:
                ax.set_title(ref)
                if col == 0:
                    ax.text(44.5 / 6, 1.2, 'Mutational Signatures', transform=ax.transAxes,
                            horizontalalignment='center', fontsize=24)
            if row < n_sigs - 1:
                ax.tick_params(axis='x', length=0)
            else:
                xlabels = [x[3:] for x in ref_map[ref]]
                ax.set_xticks(x_coords[ref])
                ax.set_xticklabels(xlabels, fontfamily='monospace', rotation='vertical')
                if col == 0:
                    ax.text(44.5 / 6, -.3, 'Motifs', transform=ax.transAxes, horizontalalignment='center', fontsize=20,
                            fontweight='bold')
            if col > 0:
                ax.tick_params(axis='y', length=0)
            if col == 9:
                ax.text(1.05, .5, sig, fontsize=14, rotation=270, transform=ax.transAxes, verticalalignment='center')

    plt.subplots_adjust(wspace=.08, hspace=.15)
    fig.text(.08, .5, 'Contributions', rotation='vertical', verticalalignment='center', fontsize=20, fontweight='bold')

    return fig

def signature_barplot_ID(W, contributions):
    """
    Plots signatures from W-matrix for Insertions-Deletions
    --------------------------------------
    Args:
        * W: W-matrix
        * contributions: Series of total contributions, np.sum(H), from each
            signature if W is normalized; else, 1

    Returns:
        * fig

    Example usage:
        signature_barplot_ID(W, np.sum(H))
    """
    W = W.copy()
    W.index = _map_id_sigs_back(W)
    for c in context83:
        if c not in W.index:
            W.loc[c] = 0
    W = W.loc[context83]
    sig_columns = [c for c in W if c.startswith('S')]
    if isinstance(contributions, pd.Series):
        W = W[sig_columns] * contributions[sig_columns]
    else:
        W = W[sig_columns] * contributions

    n_sigs = len(sig_columns)
    group_map = {'Cdel': [], 'Tdel': [], 'Cins': [], 'Tins': [],
                 '2del': [], '3del': [], '4del': [], '5+del': [],
                 '2ins': [], '3ins': [], '4ins': [], '5+ins': [],
                 '2delm': [], '3delm': [], '4delm': [], '5+delm': []}
    for x in W.index:
        group = re.search('.+?(?=[\d])', x).group(0)
        group_map[group].append(x)
    x_coords = {group: range(len(sigs)) for group, sigs in group_map.items()}

    color_map = {'Cdel': '#FFCC99', 'Tdel': '#FF8000', 'Cins': '#00FF00', 'Tins': '#00BB00',
                 '2del': '#FF99CC', '3del': '#FF3377', '4del': '#FF0000', '5+del': '#880000',
                 '2ins': '#99CCFF', '3ins': '#3377FF', '4ins': '#0000FF', '5+ins': '#000088',
                 '2delm': '#CC99FF', '3delm': '#9966FF', '4delm': '#8000FF', '5+delm': '#6000AA'}

    fig, axes = plt.subplots(nrows=n_sigs, ncols=16, figsize=(20, 2.5 * n_sigs), sharex='col',
                             sharey='row', gridspec_kw={'width_ratios': (6,) * 12 + (1, 2, 3, 5)})
    for row, sig in enumerate(sig_columns):
        for col, group in enumerate(group_map):
            if n_sigs == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            bar_heights = W[sig].loc[group_map[group]]
            ax.bar(x_coords[group], bar_heights, width=.95, linewidth=1.5, edgecolor='gray', color=color_map[group],
                   rasterized=True)
            ax.set_xlim(-.55, x_coords[group][-1] + .55)
            if row == 0:
                ax.set_title(re.search('[\d+CT]+', group).group(0), color=color_map[group])
                if col == 0:
                    ax.text(44.5 / 6, 1.3, 'Mutational Signatures', transform=ax.transAxes,
                            horizontalalignment='center', fontsize=24)
                if group == 'Tdel':
                    ax.text(-.02, 1.16, '1bp deletions at repeats', fontsize=10, transform=ax.transAxes,
                            horizontalalignment='center', color=color_map[group])
                if group == 'Tins':
                    ax.text(-.02, 1.16, '1bp insertions at repeats', fontsize=10, transform=ax.transAxes,
                            horizontalalignment='center', color=color_map[group])
                if group == '4del':
                    ax.text(-.02, 1.16, '>1bp deletions at repeats', fontsize=10, transform=ax.transAxes,
                            horizontalalignment='center', color=color_map[group])
                if group == '4ins':
                    ax.text(-.02, 1.16, '>1bp insertions at repeats', fontsize=10, transform=ax.transAxes,
                            horizontalalignment='center', color=color_map[group])
                if group == '4delm':
                    ax.text(.8, 1.16, '>1bp deletions with microhomology', fontsize=10, transform=ax.transAxes,
                            horizontalalignment='center', color=color_map[group])
            if row < n_sigs - 1:
                ax.tick_params(axis='x', length=0)
            else:
                xlabels = [re.search('[\d+]+$', x).group(0) for x in group_map[group]]
                ax.set_xticks(x_coords[group])
                ax.set_xticklabels(xlabels, fontfamily='monospace')
                if col == 0:
                    ax.text(44.5 / 6, -.3, 'Motifs', transform=ax.transAxes, horizontalalignment='center', fontsize=20,
                            fontweight='bold')
            if col > 0:
                ax.tick_params(axis='y', length=0)
            if col == 15:
                ax.text(1.05, .5, sig, fontsize=14, rotation=270, transform=ax.transAxes, verticalalignment='center')

    plt.subplots_adjust(wspace=.08, hspace=.15)
    fig.text(.08, .5, 'Contributions', rotation='vertical', verticalalignment='center', fontsize=20, fontweight='bold')

    return fig
