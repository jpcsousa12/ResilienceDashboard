import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
import re
import base64
from math import floor
from PIL import Image

# --------------------------------------------
# 1. Helpers
# --------------------------------------------

def load_setpoints(file_path):
    df = pd.read_csv(file_path)
    nested = {}
    for col in df.columns:
        if col.startswith("w_clustering"):
            continue
        if "_" in col:
            parts = col.rsplit("_", 1)
            if len(parts) == 2:
                var, member = parts
                nested.setdefault(var, {})[member] = df[col].tolist()
            else:
                nested[col] = df[col].tolist()
        else:
            nested[col] = df[col].tolist()
    return nested

def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def collect_mode_stats(base_path):
    """
    Extract obj_value, capacities, investments per hacp from all folders
    """
    data = []
    mode_folders = [
        f for f in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, f)) and "hacp_" in f
    ]
    for folder in mode_folders:
        match = re.search(r"mode_(\d+)_hacp_([\d.]+)", folder)
        if not match:
            continue
        mode = int(match.group(1))
        hacp = float(match.group(2))
        path = os.path.join(base_path, folder, "main_outputs.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            row = {
                "mode": mode,
                "hacp": hacp,
                "obj_value": df.get("obj_value", [None])[0]
            }
            # Sum capacities/investments if columns exist
            row["pv_kW"] = df.filter(like="p_gn_new_").sum(axis=1).iloc[0] if df.filter(like="p_gn_new_").shape[1] else 0
            row["bess_kWh"] = df.filter(like="e_bn_new_").sum(axis=1).iloc[0] if df.filter(like="e_bn_new_").shape[1] else 0
            row["pv_cost"] = sum(df[col].iloc[0] * df[f"l_gic_{col.split('_')[-1]}"].iloc[0] 
                                 for col in df.columns if col.startswith("p_gn_new_") and f"l_gic_{col.split('_')[-1]}" in df)
            row["bess_cost"] = sum(df[col].iloc[0] * df[f"l_bic_{col.split('_')[-1]}"].iloc[0] 
                                   for col in df.columns if col.startswith("e_bn_new_") and f"l_bic_{col.split('_')[-1]}" in df)
            data.append(row)
    return pd.DataFrame(data)

# --------------------------------------------
# 2. Layout and Sidebar
# --------------------------------------------

# Load your favicon
def get_favicon_base64(path):
    with open(path, "rb") as f:
        return f.read()

favicon = Image.open("graphics/INESCTEC.png")

st.set_page_config(
    page_title="Resilience Dashboard",
    page_icon=favicon,
    layout="wide"
)

with st.sidebar:
    # Logos
    logo1 = get_base64_image("graphics/INESCTEC_logo_secnd_mono_WHT.png")
    logo2 = get_base64_image("graphics/INESCTEC_PES_logo_mono_WHT.png")
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; height: 90px; margin-bottom: 1rem;">
            <img src="data:image/png;base64,{logo1}" style="max-height: 60px; max-width: 38%;"/>
            <img src="data:image/png;base64,{logo2}" style="max-height: 60px; max-width: 68%;"/>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🔋 Addressing Resilience in REC")
    view_mode1 = st.segmented_control("🧠 Select the approach", ["Average HACP", "HACP per hour", "Comparison"])
    st.markdown("---")


    base_dir = "./results/"

# --------------------------------------------
# 3A. Member Energy Viewer
# --------------------------------------------

if view_mode1 == "Average HACP":
    base_dir = base_dir + "average/"
if view_mode1 == "HACP per hour":
    base_dir = base_dir + "hourly/"

if view_mode1 in ["Average HACP", "HACP per hour"]:
    view_mode2 = st.sidebar.segmented_control("👀 Select what you want to see", ["Member Energy Viewer", "General View"])
    st.sidebar.markdown("---")


    if view_mode2 == "Member Energy Viewer":
        
        
        mode_folders = [f for f in os.listdir(base_dir)
                        if os.path.isdir(os.path.join(base_dir, f)) and re.match(r"mode_\d+_hacp_([\d.]+)", f)]

        hacp_values = []
        folder_map = {}
        for folder in mode_folders:
            match = re.search(r"hacp_([\d.]+)", folder)
            if match:
                hacp = float(match.group(1))
                hacp_values.append(hacp)
                folder_map[hacp] = folder

        if not hacp_values:
            st.error("No valid simulation folders found.")
            st.stop()

        
        st.sidebar.markdown("## 🔧 Configuration")

        hacp_values = sorted(set(hacp_values))
        selected_hacp = st.sidebar.selectbox("🎯 Select HACP", hacp_values)
        selected_folder = folder_map[selected_hacp]
        setpoint_path = os.path.join(base_dir, selected_folder, "setpoints.csv")

        if not os.path.exists(setpoint_path):
            st.error(f"No setpoints.csv found in {selected_folder}")
            st.stop()

        results = load_setpoints(setpoint_path)
        variables = [k for k, v in results.items() if isinstance(v, dict) and all(isinstance(val, list) for val in v.values())]
        members = sorted(results[variables[0]].keys())
        selected_member = st.sidebar.selectbox("👤 Select Member", members)

        horizon = len(results[variables[0]][selected_member])
        max_days = floor(horizon / 24)
        selected_days = st.sidebar.slider("📅 Select days", 1, max_days, value=1)
        time_range = (0, selected_days * 24 - 1)
        time = list(range(time_range[0], time_range[1]+1))


        st.markdown("## 📊 Member Energy Profile")

        
        # --------------------------------------------
        # Installed Assets and Energy Summary (Filtered)
        # --------------------------------------------

        # 1. Installed Assets (from main_outputs.csv)
        main_outputs_path = os.path.join(base_dir, selected_folder, "main_outputs.csv")
        installed_pv = 0.0
        installed_bess = 0.0

        if os.path.exists(main_outputs_path):
            df_main = pd.read_csv(main_outputs_path)
            if not df_main.empty:
                for col in df_main.columns:
                    if col.startswith("p_gn_new_") and selected_member.lower() in col.lower():
                        installed_pv += df_main[col].iloc[0]
                    if col.startswith("e_bn_new_") and selected_member.lower() in col.lower():
                        installed_bess += df_main[col].iloc[0]

        # 2. Filtered Energy Summary (from setpoints.csv)
        time_range = (0, selected_days * 24 - 1)  # already defined above
        def sum_over_range(key):
            return sum(results.get(key, {}).get(selected_member, [])[time_range[0]:time_range[1] + 1])

        total_sup = sum_over_range("e_sup")
        total_sur = sum_over_range("e_sur")
        total_gen = sum_over_range("e_g")
        total_con = sum_over_range("e_c")

        # 3. Display Results
        st.markdown("### 🧱 Installed Assets")

        col1, col2 = st.columns(2)
        col1.markdown(f"**🔆 PV Installed:** {installed_pv:.2f} kW")
        col2.markdown(f"**🔋 BESS Installed:** {installed_bess:.2f} kWh")

        st.markdown(f"### ⚡ Energy Summary (for {selected_days} day{'s' if selected_days > 1 else ''})")

        col3, col4 = st.columns(2)
        col3.markdown(f"**📦 Supplied Energy:** {total_sup:.2f} kWh")
        col4.markdown(f"**📤 Surplus Energy:** {total_sur:.2f} kWh")

        col5, col6 = st.columns(2)
        col5.markdown(f"**🔥 Consumption:** {total_con:.2f} kWh")
        col6.markdown(f"**🌞 PV Generation:** {total_gen:.2f} kWh")


        # Plot 1: PV vs Consumption
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=time, y=results["e_g"][selected_member][time_range[0]:time_range[1]+1], name="PV Generation", marker_color="darkturquoise"))
        fig1.add_trace(go.Bar(x=time, y=results["e_c"][selected_member][time_range[0]:time_range[1]+1], name="Consumption", marker_color="goldenrod"))
        fig1.add_trace(go.Scatter(x=time, y=results["l_buy"][selected_member][time_range[0]:time_range[1]+1],
                                name="Market Price", yaxis="y2", mode="lines", line=dict(color="orangered", width=2)))
        fig1.update_layout(
            title="PV Generation vs Consumption + Price",
            barmode="group",
            yaxis=dict(title="Energy [kWh]"),
            yaxis2=dict(title="Price [€/kWh]", overlaying="y", side="right"),
            margin=dict(l=60, r=60, t=40, b=40),
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Plot 2: Battery + SOC
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=time, y=results["e_bc"][selected_member][time_range[0]:time_range[1]+1], name="Charge", marker_color="limegreen"))
        fig2.add_trace(go.Bar(x=time, y=results["e_bd"][selected_member][time_range[0]:time_range[1]+1], name="Discharge", marker_color="firebrick"))
        fig2.add_trace(go.Scatter(x=time, y=results["soc"][selected_member][time_range[0]:time_range[1]+1],
                                name="SOC", mode="lines", line=dict(color="orangered", width=2), yaxis="y2"))
        fig2.add_shape(type="line", x0=time[0], x1=time[-1], y0=90, y1=90, line=dict(color="red", dash="dash"), yref="y2")
        fig2.add_shape(type="line", x0=time[0], x1=time[-1], y0=20, y1=20, line=dict(color="red", dash="dash"), yref="y2")
        fig2.update_layout(
            title="Battery Operation and SOC",
            barmode="group",
            yaxis=dict(title="Energy [kWh]"),
            yaxis2=dict(title="SOC [%]", overlaying="y", side="right", range=[0, 100]),
            margin=dict(l=60, r=60, t=40, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Plot 3: Supplied vs Surplus
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=time, y=results["e_sup"][selected_member][time_range[0]:time_range[1]+1], name="Supplied", marker_color="firebrick"))
        fig3.add_trace(go.Bar(x=time, y=results["e_sur"][selected_member][time_range[0]:time_range[1]+1], name="Surplus", marker_color="limegreen"))
        fig3.update_layout(title="Energy Supplied vs Surplus", barmode="group", yaxis_title="kWh", margin=dict(l=60, r=60, t=40, b=40))
        st.plotly_chart(fig3, use_container_width=True)

        # Plot 4: Purchased vs Sold
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=time, y=results["e_pur_pool"][selected_member][time_range[0]:time_range[1]+1], name="Purchased", marker_color="firebrick"))
        fig4.add_trace(go.Bar(x=time, y=results["e_sale_pool"][selected_member][time_range[0]:time_range[1]+1], name="Sold", marker_color="limegreen"))
        fig4.update_layout(title="Community Purchase vs Sale", barmode="group", yaxis_title="kWh", margin=dict(l=60, r=60, t=40, b=40))
        st.plotly_chart(fig4, use_container_width=True)

    # --------------------------------------------
    # 3B. General View
    # --------------------------------------------

    else:
        st.markdown("## 📊 General Overview of Simulations")

        stats_df = collect_mode_stats(base_dir)
        if stats_df.empty:
            st.warning("No valid main_outputs.csv files found.")
            st.stop()

        # Plot 1: Objective vs HACP
        fig = go.Figure()

        for mode in sorted(stats_df['mode'].unique()):
            df_mode = stats_df[(stats_df['mode'] == mode) & (~stats_df['obj_value'].isna())]
            df_mode = df_mode.sort_values(by="hacp")
            if not df_mode.empty:
                fig.add_trace(go.Scatter(
                    x=df_mode['hacp'],
                    y=df_mode['obj_value'],
                    mode='lines+markers',
                    name=f"Mode {mode}"
                ))

        fig.update_layout(
            title="Objective Value vs HACP (per Mode)",
            xaxis_title="HACP (h)",
            yaxis_title="Objective (€)",
        )
        st.plotly_chart(fig, use_container_width=True)


        # Plot 2: PV and BESS installed
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=stats_df['hacp'], y=stats_df['pv_kW'], name="PV Capacity (kW)", marker_color="darkturquoise"))
        fig2.add_trace(go.Bar(x=stats_df['hacp'], y=stats_df['bess_kWh'], name="BESS Capacity (kWh)", marker_color="goldenrod"))
        fig2.update_layout(title="Installed Capacity vs HACP", xaxis_title="HACP (h)", barmode="group")
        st.plotly_chart(fig2, use_container_width=True)

        # Plot 3: Investments
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=stats_df['hacp'], y=stats_df['pv_cost'], name="PV Investment (€)", marker_color="darkblue"))
        fig3.add_trace(go.Bar(x=stats_df['hacp'], y=stats_df['bess_cost'], name="BESS Investment (€)", marker_color="orange"))
        fig3.update_layout(title="Investment Costs vs HACP", xaxis_title="HACP (h)", barmode="group")
        st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------
# 4. Comparison Tab
# --------------------------------------------
if view_mode1 == "Comparison":
    st.markdown("## 📊 Comparison Between HACP per Hour and Average HACP")

    avg_dir = "./results/average/"
    hr_dir = "./results/hourly/"

    

    df_avg = collect_mode_stats(avg_dir)
    df_hr = collect_mode_stats(hr_dir)

    if df_avg.empty or df_hr.empty:
        st.warning("One or both simulation modes have no data.")
    else:
        # -----------------------------------------------------
        # Correct Mean per Simulation (total energy / num sims)
        # -----------------------------------------------------

        def compute_total_energy_per_simulation(results_dir):
            """
            For each simulation:
            - Sum all values (all members, all timesteps) for each energy key.
            At the end:
            - Return the total energy per key divided by number of simulations
            """
            keys = ["e_sup", "e_sur", "e_g", "e_c"]
            cumulative = {key: 0.0 for key in keys}
            sim_count = 0

            for folder in os.listdir(results_dir):
                full_path = os.path.join(results_dir, folder)
                if not os.path.isdir(full_path):
                    continue

                setpoint_path = os.path.join(full_path, "setpoints.csv")
                if not os.path.exists(setpoint_path):
                    continue

                sp = load_setpoints(setpoint_path)
                sim_count += 1

                for key in keys:
                    if key in sp:
                        total_vals = sum(sum(v) for v in sp[key].values())
                        cumulative[key] += total_vals

            if sim_count == 0:
                return {key: 0.0 for key in keys}

            return {key: cumulative[key] / sim_count for key in keys}


        def compute_avg_obj_value(results_dir):
            """
            Average obj_value across all main_outputs.csv files.
            """
            total = 0.0
            count = 0

            for folder in os.listdir(results_dir):
                full_path = os.path.join(results_dir, folder)
                csv_path = os.path.join(full_path, "main_outputs.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    if "obj_value" in df.columns and not df["obj_value"].isna().all():
                        total += df["obj_value"].iloc[0]
                        count += 1

            return total / count if count > 0 else 0.0


        # -----------------------------------------------
        # Compute final metrics
        # -----------------------------------------------
        avg_energy = compute_total_energy_per_simulation("./results/average/")
        hr_energy = compute_total_energy_per_simulation("./results/hourly/")
        avg_obj = compute_avg_obj_value("./results/average/")
        hr_obj = compute_avg_obj_value("./results/hourly/")

        # -----------------------------------------------
        # Display metrics
        # -----------------------------------------------

        labels = {
            "e_sup": "⚡ Supplied",
            "e_sur": "📤 Surplus",
            "e_g": "🔆 PV Generation",
            "e_c": "🔥 Consumption"
        }

        cols = st.columns(len(labels))

        for i, key in enumerate(labels):
            avg_val = avg_energy[key]
            hr_val = hr_energy[key]
            delta = hr_val - avg_val
            delta_pct = (delta / avg_val * 100) if avg_val else 0.0

            cols[i].metric(
                label=f"{labels[key]} (Hourly HACP)",
                value=f"{hr_val:.2f} kWh",
                delta=f"{delta:+.2f} kWh ({delta_pct:+.1f}%) vs Avg"
            )

        delta_obj = hr_obj - avg_obj
        delta_obj_pct = (delta_obj / avg_obj * 100) if avg_obj else 0.0

        st.metric(
            label="💸 Cost (Hourly HACP)",
            value=f"{hr_obj:,.2f} €",
            delta=f"{delta_obj:+,.2f} € ({delta_obj_pct:+.1f}%) vs Avg"
        )


        df_avg = df_avg.sort_values(by='hacp')
        df_hr = df_hr.sort_values(by='hacp')

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_avg['hacp'], y=df_avg['obj_value'],
            mode='lines+markers',
            name='Average HACP',
            line=dict(color='royalblue', width=2),
            marker=dict(size=4, opacity=0.6)
        ))
        fig.add_trace(go.Scatter(
            x=df_hr['hacp'], y=df_hr['obj_value'],
            mode='lines+markers',
            name='Hourly HACP',
            line=dict(color='lightblue', width=2),
            marker=dict(size=4, opacity=0.6)
        ))
        fig.update_layout(
            title="Objective Value Comparison",
            xaxis_title="HACP (h)",
            yaxis_title="Objective (€)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=df_avg['hacp'], y=df_avg['pv_kW'], name="PV (Average)", marker_color="darkblue"))
        fig2.add_trace(go.Bar(x=df_hr['hacp'], y=df_hr['pv_kW'], name="PV (Hourly)", marker_color="lightblue"))
        fig2.add_trace(go.Bar(x=df_avg['hacp'], y=df_avg['bess_kWh'], name="BESS (Average)", marker_color="darkorange"))
        fig2.add_trace(go.Bar(x=df_hr['hacp'], y=df_hr['bess_kWh'], name="BESS (Hourly)", marker_color="navajowhite"))
        fig2.update_layout(
            title="Installed Capacity Comparison",
            xaxis_title="HACP (h)",
            barmode="group"
        )
        st.plotly_chart(fig2, use_container_width=True)
