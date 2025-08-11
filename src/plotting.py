import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os
from pathlib import Path

st.set_page_config(
    page_title="Adversarial MARL Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration ---
BASE_RESULTS_PATH = Path(
    "/home/amine.andam/lustre/vr_outsec-vh2sz1t4fks/users/amine.andam/epymarl/results/attacks"
)

class MARLDataLoader:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.cache = {}

    def parse_name(self, name):
        d = {}
        for part in name.split('-'):
            if '=' in part:
                k, v = part.split('=')
                if k == 'num': d['num_agents'] = int(v)
                elif k == 'eps': d['epsilon'] = float(v)
                elif k == 'frms': d['frames'] = int(v)
                elif k == 'K': d['K'] = int(v)
                elif k == 'envst': d['step'] = int(v)
                elif k == 'alpha': d['alpha'] = float(v)
        return d

    def scan(self):
        rows = []
        # Baseline (no attack) data
        evals_path = self.base_path / 'evals'
        if evals_path.exists():
            stats_file = evals_path / 'stats.json'
            if stats_file.exists():
                try:
                    js = json.load(open(stats_file, 'r'))
                    ret = js['returns']; ep = js['ep_lengths']
                    rows.append({
                        'attack': 'baseline',
                        'epsilon': 0.0,
                        'path': str(stats_file),
                        'iqm_return': ret['iqm'],
                        'iqm_return_CI_lo': ret['iqm_ci'][0],
                        'iqm_return_CI_hi': ret['iqm_ci'][1],
                        'iqm_ep_length': ep['iqm'],
                        'iqm_ep_CI_lo': ep['iqm_ci'][0],
                        'iqm_ep_CI_hi': ep['iqm_ci'][1]
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    st.warning(f"Error reading {stats_file}: {e}")
        # Attacks
        for attack in ['align', 'noise', 'whitebox']:
            atk_path = self.base_path / attack
            if not atk_path.exists(): continue
            if attack == 'align':
                for net in ['mlp', 'rnn', 'transformer']:
                    net_path = atk_path / net
                    if not net_path.exists(): continue
                    for step_dir in net_path.iterdir():
                        if not step_dir.is_dir(): continue
                        for exp in step_dir.iterdir():
                            if not exp.is_dir(): continue
                            meta = self.parse_name(exp.name)
                            meta.update({'attack': attack, 'network': net, 'step': int(step_dir.name)})
                            stats_file = exp / 'stats.json'
                            if stats_file.exists(): rows.append({**meta, 'path': str(stats_file)})
            else:
                subdirs = atk_path.iterdir() if attack == 'noise' else [atk_path]
                for subtype in subdirs:
                    if not subtype.is_dir(): continue
                    for exp in subtype.iterdir():
                        if not exp.is_dir(): continue
                        meta = self.parse_name(exp.name)
                        meta['attack'] = attack
                        if attack == 'noise': meta['noise_type'] = subtype.name
                        stats_file = exp / 'stats.json'
                        if stats_file.exists(): rows.append({**meta, 'path': str(stats_file)})
        return pd.DataFrame(rows)

    def load(self, path):
        if path in self.cache:
            return self.cache[path]
        data = json.load(open(path, 'r'))
        self.cache[path] = data
        return data


def main():
    st.title("🤖 Adversarial MARL Dashboard")
    # Algorithm & Environment selection
    st.sidebar.header("Select Setup")
    algos = [d.name for d in BASE_RESULTS_PATH.iterdir() if d.is_dir()]
    selected_algo = st.sidebar.selectbox("Algorithm", algos)
    envs_path = BASE_RESULTS_PATH / selected_algo
    envs = [d.name for d in envs_path.iterdir() if d.is_dir()]
    selected_env = st.sidebar.selectbox("Environment / Game", envs)
    full_path = envs_path / selected_env
    st.sidebar.markdown(f"**Results Path:** `{full_path}`")
    if not full_path.is_dir():
        st.error(f"Invalid path: {full_path}")
        return
    # Load and scan
    loader = MARLDataLoader(full_path)
    df = loader.scan()
    if df.empty:
        st.warning("No data found")
        return
    # Filters
    st.sidebar.header("Filters")
    sel_attack = st.sidebar.multiselect(
        "Attack Types", sorted(df['attack'].unique()), default=sorted(df['attack'].unique())
    )
    filtered = df[df['attack'].isin(sel_attack)]
    if 'align' in sel_attack:
        nets = sorted(filtered[filtered['attack']=='align']['network'].unique())
        sel_net = st.sidebar.multiselect("Network Types", nets, default=nets)
        filtered = filtered[~((filtered['attack']=='align') & (~filtered['network'].isin(sel_net)))]
        steps = sorted(filtered[filtered['attack']=='align']['step'].unique())
        sel_steps = st.sidebar.multiselect("Env Steps", steps, default=steps)
        filtered = filtered[~((filtered['attack']=='align') & (~filtered['step'].isin(sel_steps)))]
        if 'K' in filtered.columns:
            Ks = sorted(filtered[filtered['attack']=='align']['K'].unique())
            sel_K = st.sidebar.multiselect("K Iterations", Ks, default=Ks)
            filtered = filtered[~((filtered['attack']=='align') & (~filtered['K'].isin(sel_K)))]
    st.sidebar.markdown(f"**Entries: {len(filtered)}**")
    # Table
    st.subheader("Comparison Table: IQM Returns & Episode Lengths")
    rows = []
    for _, r in filtered.iterrows():
        record = {
            'attack': r['attack'],
            'network': r.get('network','-'),
            'noise': r.get('noise_type','-'),
            'step': r.get('step','-'),
            'epsilon': r.get('epsilon','-'),
            'K': r.get('K','-')
        }
        if r['attack']=='baseline':
            record.update({
                'iqm_return': r['iqm_return'],
                'iqm_return_CI_lo': r['iqm_return_CI_lo'],
                'iqm_return_CI_hi': r['iqm_return_CI_hi'],
                'iqm_ep_length': r['iqm_ep_length'],
                'iqm_ep_CI_lo': r['iqm_ep_CI_lo'],
                'iqm_ep_CI_hi': r['iqm_ep_CI_hi']
            })
        else:
            js = loader.load(r['path'])
            ret = js['returns']; ep = js['ep_lengths']
            record.update({
                'iqm_return': ret['iqm'],
                'iqm_return_CI_lo': ret['iqm_ci'][0],
                'iqm_return_CI_hi': ret['iqm_ci'][1],
                'iqm_ep_length': ep['iqm'],
                'iqm_ep_CI_lo': ep['iqm_ci'][0],
                'iqm_ep_CI_hi': ep['iqm_ci'][1]
            })
        rows.append(record)
    table = pd.DataFrame(rows)
    table.to_csv(f"/home/amine.andam/lustre/vr_outsec-vh2sz1t4fks/users/amine.andam/epymarl/src/{selected_algo}-{selected_env}.csv")
    st.dataframe(table, use_container_width=True, height=400)
    # Plot
    st.subheader("IQM Return vs Epsilon: Baseline vs Align vs Max-Noise vs Whitebox")
    fig = go.Figure()
    # Baseline
    if 'baseline' in table['attack'].values:
        df_b = table[table['attack']=='baseline']
        agg = df_b.groupby('epsilon', as_index=False)[['iqm_return','iqm_return_CI_lo','iqm_return_CI_hi']].mean()
        fig.add_trace(go.Scatter(
            x=agg['epsilon'], y=agg['iqm_return'], mode='markers+lines', name='Baseline',
            error_y=dict(
                type='data',
                array=agg['iqm_return_CI_hi']-agg['iqm_return'],
                arrayminus=agg['iqm_return']-agg['iqm_return_CI_lo']
            )
        ))
    # Align
    if 'align' in sel_attack and not table[table['attack']=='align'].empty:
        for (step,K), grp in table[table['attack']=='align'].groupby(['step','K']):
            grp = grp.sort_values('epsilon')
            fig.add_trace(go.Scatter(
                x=grp['epsilon'], y=grp['iqm_return'], mode='markers+lines',
                name=f"Align s={step},K={K}",
                error_y=dict(
                    type='data',
                    array=grp['iqm_return_CI_hi']-grp['iqm_return'],
                    arrayminus=grp['iqm_return']-grp['iqm_return_CI_lo']
                )
            ))
    # Noise
    if 'noise' in sel_attack and not table[table['attack']=='noise'].empty:
        df_n = table[table['attack']=='noise']
        idx = df_n.groupby('epsilon')['iqm_return'].idxmin()
        mn = df_n.loc[idx]
        fig.add_trace(go.Scatter(
            x=mn['epsilon'], y=mn['iqm_return'], mode='markers+lines', name='Max-Noise',
            error_y=dict(
                type='data',
                array=mn['iqm_return_CI_hi']-mn['iqm_return'],
                arrayminus=mn['iqm_return']-mn['iqm_return_CI_lo']
            )
        ))
    # Whitebox
    if 'whitebox' in sel_attack and not table[table['attack']=='whitebox'].empty:
        df_w = table[table['attack']=='whitebox']
        agg_w = df_w.groupby('epsilon', as_index=False)[['iqm_return','iqm_return_CI_lo','iqm_return_CI_hi']].mean()
        fig.add_trace(go.Scatter(
            x=agg_w['epsilon'], y=agg_w['iqm_return'], mode='markers+lines', name='Whitebox',
            error_y=dict(
                type='data',
                array=agg_w['iqm_return_CI_hi']-agg_w['iqm_return'],
                arrayminus=agg_w['iqm_return']-agg_w['iqm_return_CI_lo']
            )
        ))
    fig.update_layout(xaxis_title='Epsilon', yaxis_title='IQM Return', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
