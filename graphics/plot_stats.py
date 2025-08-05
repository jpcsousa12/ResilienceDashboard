import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import zoom
from loguru import logger

def extract_mode_hacp(foldername):
    m = re.match(r'mode_(\d+)_hacp_([\d.]+)', foldername)
    if m:
        return int(m.group(1)), float(m.group(2))
    return None, None

def plot_stats(base_path):
    """
    For each mode (x), draws and saves a plot of obj_value vs hacp (y).
    Also sums PV/BESS capacities and investment costs.
    """
    mode_data = {}

    folders = [f for f in os.listdir(base_path)
               if os.path.isdir(os.path.join(base_path, f))
               and f.startswith("mode_") and "_hacp_" in f]
    for folder in folders:
        x, y = extract_mode_hacp(folder)
        if x is None or y is None:
            continue
        main_outputs_path = os.path.join(base_path, folder, "main_outputs.csv")
        if os.path.exists(main_outputs_path):
            try:
                df = pd.read_csv(main_outputs_path)

                obj_value = float(df['obj_value'].iloc[0]) if 'obj_value' in df.columns else None

                # Capacity columns
                pv_columns = [col for col in df.columns if col.startswith('p_gn_new_')]
                bess_columns = [col for col in df.columns if col.startswith('e_bn_new_')]

                # Investment columns
                pv_cost_columns = [col for col in df.columns if col.startswith('l_gic_')]
                bess_cost_columns = [col for col in df.columns if col.startswith('l_bic_')]

                # Extract from row 0
                total_pv = df[pv_columns].iloc[0].sum() if pv_columns else 0.0
                total_bess = df[bess_columns].iloc[0].sum() if bess_columns else 0.0
                
               # Calculate PV investment: sum over (p_gn_new_* × l_gic_*)
                total_pv_costs = sum(
                    df[pv_col].iloc[0] * df[gic_col].iloc[0]
                    for pv_col, gic_col in zip(pv_columns, pv_cost_columns)
                    if pv_col.replace('p_gn_new_', '') == gic_col.replace('l_gic_', '')
                )

                # Calculate BESS investment: sum over (e_bn_new_* × l_bic_*)
                total_bess_costs = sum(
                    df[bess_col].iloc[0] * df[bic_col].iloc[0]
                    for bess_col, bic_col in zip(bess_columns, bess_cost_columns)
                    if bess_col.replace('e_bn_new_', '') == bic_col.replace('l_bic_', '')
                )


                mode_data.setdefault(x, {})[y] = {
                    'obj_value': obj_value,
                    'total_pv': total_pv,
                    'total_bess': total_bess,
                    'pv_cost': total_pv_costs,
                    'bess_cost': total_bess_costs
                }

            except Exception as e:
                print(f"⚠️ Error reading {main_outputs_path}: {e}")

    # Load logo
    logo_path = 'graphics/INESCTEC_logo_secnd_COLOR.png'
    logo_img, logo_resized = None, None
    if os.path.exists(logo_path):
        logo_img = mpimg.imread(logo_path)
        scaling_factor = 0.04
        logo_resized = zoom(logo_img, (scaling_factor, scaling_factor, 1))

    for mode_x in sorted(mode_data.keys()):
        hacp_ys = sorted(mode_data[mode_x].keys())
        obj_values = [mode_data[mode_x][y]['obj_value'] for y in hacp_ys]

        fig, ax = plt.subplots(figsize=(10, 6))
        (line,) = ax.plot(
            hacp_ys, obj_values, marker='o', linestyle='-', color='royalblue', label='Cost (€)'
        )

        plt.suptitle(
            f"Mode {mode_x}: REC Cost vs HACP",
            fontsize=15,
            fontweight='bold',
            x=0.47,
            y=0.97,
        )

        if logo_resized is not None:
            fig.figimage(logo_resized, xo=int(fig.bbox.xmin) + 15, yo=int(fig.bbox.ymax) - 65, zorder=10, alpha=1.00)

        ax.set_xlabel("HACP (h)", fontsize=12)
        ax.set_ylabel("Euros (€)", fontsize=12)
        ax.grid(True)
        ax.set_xticks(hacp_ys)
        ax.tick_params(axis='x', rotation=0)

        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fancybox=True, shadow=True, fontsize=12)
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        save_path = os.path.join(base_path, f"mode_{mode_x}_obj_value_plot.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        logger.success(f"Saved plot: {save_path}")

        # Summary CSV
        summary_rows = []
        for y in hacp_ys:
            data = mode_data[mode_x][y]
            summary_rows.append({
                "Mode": mode_x,
                "HACP (h)": y,
                "Total REC Cost (€)": data['obj_value'],
                "Total PV Installed (kW)": data['total_pv'],
                "PV Investment (€)": data['pv_cost'],
                "Total BESS Installed (kW)": data['total_bess'],
                "BESS Investment (€)": data['bess_cost'],
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(base_path, f"mode_{mode_x}_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        logger.success(f"Summary CSV saved to: {summary_csv}")
