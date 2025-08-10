#DC-MJ Cell GitHub
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.constants import h, c, e, k
from scipy import integrate  
import re
file_path = r"D:\dowloads\AM1.5Global.xlsx" #Import the AM1.G spectrum here
df_logging = pd.read_excel(r"D:\dowloads\Data Logging #1.xlsx", sheet_name=0) #Import the optimized bandgap data sheet here
data = pd.read_excel(file_path)
wavelength = data.iloc[:, 0]  # in nm
irradiance = data.iloc[:, 1]  # in W/m²/nm
bandgap = np.linspace(0.1, 5, 1000)  # eV range
q = e     # Elementary charge in C
T = 300   #Room Temperatuer in K
pi = np.pi #pi 
n = 1      #Ideality Factor
CM = 1    #Carrier Multiplication 
# Convert to NumPy arrays for safe indexing
wavelength_np = wavelength.values
irradiance_np = irradiance.values
def compute_j0(Eg): 
    prefactor = 2*pi*q**4/(h**3*c**2)
    integrand = lambda E: E**2/(np.expm1(np.clip(q*E/(k*T), -700, 700)))
    integral,_ = integrate.quad(integrand, Eg, np.inf)
    return prefactor * integral 
def compute_thermal_efficiency(Eg, irradiance, wavelength):
    eV = h * c / (q * wavelength * 1e-9)
    p_in = np.trapezoid(irradiance, wavelength)
    mask = eV > Eg
    photon_flux = irradiance[mask]/(q*eV[mask])
    thermal_loss = np.trapezoid((eV[mask]-Eg)*photon_flux*CM*q, wavelength[mask])
    thermal_eff = 100*thermal_loss/p_in
    return thermal_eff
def compute_unabs_efficiency(Eg, irradiance, wavelength):
    eV = h * c / (q * wavelength * 1e-9)
    p_in = np.trapezoid(irradiance, wavelength)
    unmasked = (eV < Eg) 
    unabs_flux = np.trapezoid(irradiance[unmasked], wavelength[unmasked]) 
    unabs_eff = 100*unabs_flux/p_in
    return unabs_eff
def compute_recomb_efficiency(Eg, irradiance, wavelength):
    # Convert wavelength to photon energy (eV)
    eV = h * c / (q * wavelength * 1e-9)
    # Mask for absorbed photons (E >= Eg)
    mask = eV >= Eg
    if not np.any(mask):
        return 0.0
    # Photon flux and photocurrent
    photon_flux = np.trapezoid(irradiance[mask] / (q * eV[mask]), wavelength[mask])
    j_ph = CM * q * photon_flux
    # Input power
    p_in = np.trapezoid(irradiance, wavelength)
    # Reverse saturation current (J0)
    j0 = compute_j0(Eg)
    # Open-circuit voltage (Voc)
    voc = n * k * T * np.log((j_ph / j0) + 1) / q
    volt = np.linspace(0,voc,500)
    j_total = j_ph - j0 * np.expm1(volt * q/(n*k*T))  #modified ideal diode
    power = j_total * volt
    vmp = volt[np.argmax(power)]
    recomb_loss = vmp*j0 + q*np.trapezoid((Eg - (vmp)) * (irradiance[mask]/(q*eV[mask])), wavelength[mask])
    recomb_eff = 100*recomb_loss/p_in
    return recomb_eff 
def compute_efficiency(Eg, irradiance, wavelength):
    eV = h * c / (q * wavelength * 1e-9)
    mask = eV >= Eg
    if not np.any(mask):
        return 0.0
    photon_flux = np.trapezoid(irradiance[mask] / (q * eV[mask]), wavelength[mask])
    j_ph = CM*q*photon_flux
    p_in = np.trapezoid(irradiance, wavelength)
    unabs_eff = compute_unabs_efficiency(Eg, irradiance, wavelength)
    thermal_eff = compute_thermal_efficiency(Eg, irradiance, wavelength)
    recomb_eff = compute_recomb_efficiency(Eg, irradiance, wavelength)
    j0 = compute_j0(Eg) 
    voc = n*k*T*np.log((j_ph/j0) + 1)/q
    volt = np.linspace(0,voc,500)
    j_total = j_ph - j0 * np.expm1(volt * q/(n*k*T))  #modified ideal diode
    power = j_total * volt
    max_power = np.max(power)
    efficiency = 100*max_power/p_in
    return efficiency 
    #return efficiency + unabs_eff + thermal_eff + recomb_eff 
def compute_loss_efficiency(Eg, irradiance, wavelength):
    unabs_eff = compute_unabs_efficiency(Eg, irradiance, wavelength)
    thermal_eff = compute_thermal_efficiency(Eg, irradiance, wavelength)
    recomb_eff = compute_recomb_efficiency(Eg, irradiance, wavelength)
    return unabs_eff + thermal_eff + recomb_eff 
#Figure 1: SQ Limit
efficiencies = np.array([compute_efficiency(Eg, irradiance, wavelength) for Eg in bandgap])
unabs_losses = np.array([compute_unabs_efficiency(Eg, irradiance, wavelength) for Eg in bandgap])
thermal_losses = np.array([compute_thermal_efficiency(Eg, irradiance, wavelength) for Eg in bandgap])
recomb_losses = 100 - efficiencies - unabs_losses - thermal_losses
recomb_losses = np.maximum(recomb_losses, 0) 
plt.figure(figsize=(8, 6))
labels = ['SQ Efficiency', 'Recombination Loss', 'Thermalization Loss', 'Unabsorbed Loss']
colors = [
    '#1a53ff',  # SQ efficiency
    '#ff3333',  # Recombination loss
    '#ff9900',  # Thermalization loss
    '#339933'   # Transmission loss
]
plt.stackplot(bandgap,efficiencies,thermal_losses,unabs_losses,recomb_losses,labels=['SQ Efficiency', 'Recombination Loss', 'Thermalization Loss', 'Unabsorbed Loss'],colors=colors,edgecolor='white',)
plt.plot(bandgap, efficiencies, 'w-', linewidth=2, zorder=10)
plt.plot(bandgap, efficiencies, 'k--', linewidth=1.5, zorder=10)
plt.xlabel('Bandgap Energy (eV)', fontsize=12, fontweight='bold')
plt.ylabel('Efficiency / Loss Components (%)', fontsize=12, fontweight='bold')
plt.title('Solar Cell Efficiency Breakdown and Loss Mechanisms', fontsize=14, fontweight='bold', pad=15)
legend = plt.legend(loc='upper right', frameon=True, framealpha=0.95, facecolor='#f8f8f8', edgecolor='#404040')
legend.get_frame().set_boxstyle("round,pad=0.3,rounding_size=0.2")
plt.grid(True, linestyle='--', alpha=0.7, color='#dddddd')
plt.ylim(0, 100)
plt.xlim(bandgap.min(), bandgap.max())
plt.gca().set_facecolor('#f5f5f5')
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['top'].set_color('#d0d0d0')
plt.gca().spines['right'].set_color('#d0d0d0')
plt.tight_layout()
plt.show()
# Figure 3: Bandgap distribution vs MJ count (no DC)
plt.figure(figsize=(8,6))
colors = plt.cm.tab20(np.linspace(0, 1, 12))
list_of_bandgaps = []
for N in range(1, 13):
    row = df_logging[(df_logging['MJ Count'] == N) & (df_logging['DC Count'] == 0)]
    if not row.empty:
        bg_data = row.iloc[0, 4]
        if pd.isna(bg_data):
            bgs = []
        elif isinstance(bg_data, float):
            bgs = [bg_data]
        elif isinstance(bg_data, str):
            if '\n' in bg_data:
                bgs = [float(x) for x in bg_data.split('\n') if x.strip()]
            elif ',' in bg_data:
                bgs = [float(x) for x in bg_data.split(',') if x.strip()]
            else:
                try:
                    bgs = [float(bg_data)]
                except:
                    bgs = []
        else:
            bgs = []
    else:
        fallback_row = df_logging[(df_logging['MJ Count'] == N)].sort_values('DC Count').head(1)
        if not fallback_row.empty:
            bg_data = fallback_row.iloc[0, 4]
        else:
            bgs = []
    list_of_bandgaps.append(bgs)
for N, bgs in enumerate(list_of_bandgaps, start=1):
    if not bgs:  
        print(f"Warning: No bandgap data found for MJ={N}, DC=0")
        continue
        
    color = colors[N-1]
    plt.hlines(y=N, xmin=min(bgs), xmax=max(bgs), color=color, linewidth=1)
    for bg in bgs:
        plt.plot([bg, bg], [N - 0.2, N + 0.2], color=color, linewidth=1)
    plt.plot([], [], color=color, label=f'{N} MJ')
plt.xlim(0.1, 5)
major_ticks = [0.1, 5]
minor_ticks = np.arange(0.5, 5, 0.5)
plt.xticks(major_ticks + list(minor_ticks))
plt.tick_params(axis='x', which='major', length=8)
plt.tick_params(axis='x', which='minor', length=4)
plt.xlabel('Bandgap (eV)')
plt.ylabel('Number of MJ Cells')
plt.title('Optimal Bandgap Distribution vs MJ Count')
plt.yticks(range(1, 13))
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.legend(title='MJ Cell Count', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
# Figure 4 Efficiency heatmap (DC vs MJ)
fix_dc = [0,5,11]    # DC layers at which to draw vertical lines (x-axis)
fix_mj = [1,2, 12]    # MJ layers at which to draw horizontal lines (y-axis)
pivot = df_logging.pivot(index='MJ Count', columns='DC Count', values='Efficiency (%)')
eff_map = pivot.to_numpy()
plt.figure(figsize=(8,6))
plt.imshow(eff_map, origin='lower', aspect='auto', cmap='inferno')
plt.colorbar(label='Efficiency (%)')
plt.xlabel('DC Layer Count')
plt.ylabel('MJ Cell Count')
plt.title('Efficiency Heatmap: MJ vs DC')
plt.xticks(ticks=np.arange(pivot.columns.size), labels=pivot.columns)
plt.yticks(ticks=np.arange(pivot.index.size), labels=pivot.index)
for dc in fix_dc:
    if dc in pivot.columns:
        x = list(pivot.columns).index(dc)
        plt.vlines(x, -0.5, pivot.index.size-0.5, colors='black', linewidth=2)
for mj in fix_mj:
    if mj in pivot.index:
        y = list(pivot.index).index(mj)
        plt.hlines(y, -0.5, pivot.columns.size-0.5, colors='black', linewidth=2)

plt.tight_layout()
plt.show()
# Figure 4a, Efficiency vs MJ for fixed DC layers
plt.figure(figsize=(8,4))
for dc in fix_dc:
    if dc in pivot.columns:
        plt.plot(pivot.index, pivot[dc], marker='o', label=f'DC = {dc}')
plt.xlabel('MJ Cell Count')
plt.ylabel('Efficiency (%)')
plt.title('Efficiency vs MJ Cells @ Fixed DC Layers')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(8,4))
#Figure 4b, Irradiance vs Bandgap for fixed MJ=12, varying DC
for mj in fix_mj:
    if mj in pivot.index:
        plt.plot(pivot.columns, pivot.loc[mj], marker='s', label=f'MJ = {mj}')
plt.xlabel('DC Layer Count')
plt.ylabel('Efficiency (%)')
plt.title('Efficiency vs DC Layers @ Fixed MJ Cells')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#Figure 5
efficiencies = np.array([compute_efficiency(Eg, irradiance, wavelength) for Eg in bandgap])
dc_Eg = [4.9942, 4.7165, 4.6228, 4.3170, 4.2161, 4.0527, 4.0499, 3.9675, 3.9519, 3.8960, 3.8956, 3.8098, 3.7976, 3.7195, 3.6618, 3.5686, 3.5504, 3.5293, 3.4891, 3.4686, 3.4580, 3.3924, 3.2986, 3.2190, 3.2033, 3.1899, 3.1801, 3.1508, 3.1153, 3.0523, 3.0480, 3.0050, 2.9586, 2.9571, 2.9036, 2.8345, 2.8212, 2.7871, 2.7298, 2.6745, 2.6015, 2.6009, 2.5530, 2.5302, 2.5271, 2.4659, 2.4625, 2.4477, 2.3892, 2.3877, 2.3553, 2.3100, 2.2919, 2.2490, 2.1764, 2.1251, 2.0855, 2.0506, 2.0081, 1.9890, 1.9185, 1.9048, 1.8570, 1.8093, 1.6973, 1.6212, 1.6085, 1.5987, 1.5839, 1.5710, 1.5662, 1.5443, 1.5381, 1.4425, 1.3752, 1.3040, 1.2661, 1.1826, 1.1647, 1.1507, 1.1257, 1.0644, 1.0569, 0.9745, 0.9702, 0.9544, 0.9335, 0.9110, 0.8630, 0.8461, 0.8085, 0.7558, 0.7247, 0.6814, 0.6383, 0.6321, 0.5313, 0.5209, 0.5179, 0.3395]  # Down-converting layer energies (eV)
mj_Eg = [0.3098]  # Multijunction bandgaps (eV), must be decreasing
E_spectrum = h*c/ (q*wavelength_np*1e-9)  #converts wavelength to eV
I_dc = irradiance_np.copy() 
P_in = np.trapezoid(I_dc, wavelength_np)  #Total incident power (W/m²)
#Down conversion calculations
dλ = np.zeros_like(wavelength_np) #wavelength bin widths
dλ[0] = wavelength_np[1] - wavelength_np[0]
dλ[-1] = wavelength_np[-1] - wavelength_np[-2]
dλ[1:-1] = (wavelength_np[2:] - wavelength_np[:-2]) / 2.0
for Eg_target in dc_Eg:
    indices_src = np.where(np.abs(E_spectrum - Eg_target) <= 0.05)[0] #within 50mev of layer
    for i in indices_src:
        if I_dc[i] <= 0:
            continue
        E_tgt = E_spectrum[i] / 2.0  # Target energy for the two new photons
        indices_dst = np.where(np.abs(E_spectrum - E_tgt) <= 0.05)[0] #within 50mev of target
        if len(indices_dst) == 0:
            continue  #Skip if no destination bins
        I_dc[indices_dst] += I_dc[i] * dλ[i] / np.sum(dλ[indices_dst]) #distribute total energy over bin length to target
        I_dc[i] = 0  #Remove the original photon
#Multijunction cell calculation
P_max_list = []  #Max power for each subcell
eta_i_list = []  #Efficiency of each subcell
for i, Eg_i in enumerate(mj_Eg):
    if i == 0:
        # Top cell: absorbs photons from Eg_i to infinity
        λ_min = 0 
        λ_max = h*c / (q*1e-9*Eg_i) #top cell wavelength
        maskm = (wavelength <= λ_max)  
    else:
        Eg_above = mj_Eg[i-1]  # Bandgap of the cell above
        λ_min = h*c / (q*1e-9*Eg_above)  # Minimum wavelength for absorption
        λ_max = h*c / (q*1e-9*Eg_i)  # Maximum wavelength for absorption
        maskm = (wavelength >= λ_min) & (wavelength <= λ_max)
    E_cell = h * c / (q * wavelength[maskm] * 1e-9) 
    total_photon_flux = np.trapezoid(I_dc[maskm]/(q*E_cell), wavelength[maskm])
    J_ph = CM * q * total_photon_flux
    J0 = compute_j0(Eg_i)
    assert J_ph > 0 and J0 > 0
    Voc = n*k*T*np.log1p((J_ph/J0) + 1)/q 
    V = np.linspace(0, Voc, 500) if Voc > 0 else np.array([0])
    J_total = J_ph - J0 * np.expm1(V * e / (n * k * T))
    power = J_total * V 
    assert len(power) > 0
    max_power = np.max(power) 
    P_max_list.append(max_power)
    eta_i_list.append(max_power / P_in * 100.0)  # Subcell efficiency
    print(f"Power at {Eg_i} eV cell {max_power:.2f} watts")
P_collected = sum(P_max_list)
total_eff = P_collected / P_in * 100.0
print(f"Total power with down-conversion: {P_collected:.2f} watts")
print(f"Total efficiency with down-conversion: {total_eff:.2f}%")
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
irradiance_eV = irradiance * (q*1e-9*wavelength**2/(h*c))
I_dc_eV = I_dc * (q*1e-9*wavelength**2/(h*c))
E_orig = (h*c) / (q*1e-9*wavelength)
sort_idx = np.argsort(E_orig)
# Plot 1: Modified spectrum vs energy
axs[0].plot(E_orig[sort_idx], I_dc_eV[sort_idx])
axs[0].set_ylabel('Irradiance (W/m²/eV)')
axs[0].set_xlabel('Energy (eV)')
axs[0].set_title('Down-Converted Spectrum (vs. Energy)')
axs[0].grid(True)
# Plot 2: SQ limit with down-converted spectrum
efficiencies_dc = np.array([compute_efficiency(Eg, I_dc, wavelength) for Eg in bandgap])
axs[1].plot(bandgap, efficiencies_dc, 'r-')
axs[1].set_xlabel('Bandgap (eV)')
axs[1].set_ylabel('Efficiency (%)')
axs[1].set_title('SQ Limit (Down-Converted Spectrum)')
axs[1].grid(True)
plt.show()
#Figure 6: Modiifed Irradiance graphs
def _parse_cell_to_floats(cell):
    if pd.isna(cell):
        return []
    if isinstance(cell, (int, float, np.integer, np.floating)):
        return [float(cell)]
    s = str(cell).strip()
    if s.lower() == "none":
        return [0.0]
    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
    return [float(x) for x in nums] if nums else []
col0 = df_logging.iloc[:, 0].astype(str).str.strip().str.lower().replace({'none': '0'}).astype(float)
col1 = pd.to_numeric(df_logging.iloc[:, 1], errors='coerce')
mask = (col1 == 12) & (col0.isin([0, 9, 11])) # select rows where MJ == 12 and DC == {0,9,11}, the "optimal" cases
sel = df_logging[mask]
req_order = [0, 9, 11] 
dc_map = {}
for _, row in sel.iterrows():
    raw_key = row.iloc[0]
    if str(raw_key).strip().lower() == 'none':
        key = 0
    else:
        try:
            key = int(float(raw_key))
        except Exception:
            continue
    dc_map[key] = _parse_cell_to_floats(row.iloc[3])
list_of_dc_bandgaps = [dc_map.get(k, []) for k in req_order]
E_spectrum = h*c/(q*wavelength*1e-9) 
irradiance_eV = irradiance * (q*1e-9*wavelength**2/(h*c)) 
shift_x = 1.5 # per curve for visibility (tweak as needed)
colors = plt.cm.nipy_spectral(np.linspace(0,1,len(list_of_dc_bandgaps)))
plt.figure(figsize=(16,8))
colors = plt.cm.nipy_spectral(np.linspace(0,1,len(list_of_dc_bandgaps)))
for idx, dc_targets in enumerate(list_of_dc_bandgaps):
    label_key = req_order[idx] 
    I_mod = irradiance.copy()
    d_lambda = np.gradient(wavelength)
    for Eg_target in dc_targets:
        idx_src = np.where(np.abs(E_spectrum - Eg_target) <= 0.05)[0]
        for i in idx_src:
            if I_mod[i] <= 0:
                continue
            E_tgt = E_spectrum[i] / 2
            idx_dst = np.where(np.abs(E_spectrum - E_tgt) <= 0.05)[0]
            if not len(idx_dst):
                continue
            I_mod[idx_dst] += I_mod[i] * d_lambda[i] / np.sum(d_lambda[idx_dst])
            I_mod[i] = 0

    I_mod_eV = I_mod * (q*1e-9*wavelength**2/(h*c))
    x_shifted = E_spectrum + idx * shift_x
    plt.plot(x_shifted, I_mod_eV, color=colors[idx], linewidth=1.5,
             label=f"DC {label_key} (n={len(dc_targets)})")
ax = plt.gca()
ax.set_ylim(bottom=0)
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(axis='x', which='both', length=0)
ax.set_xticks([])         
ax.set_ylabel('Irradiance (W/m²/eV)')
ax.set_title('Shifted AM1.5G Spectra vs Energy for Varying DC Layers')
ymin, ymax = ax.get_ylim()
tick_len = 0.02 * (ymax - ymin)
x_start = E_spectrum.min()
for idx, color in enumerate(colors):
    label_key = req_order[idx]
    x0 = x_start + idx * shift_x
    ax.plot([x0, x0], [-1, tick_len], color=color, linewidth=1)
    ax.text(x0, -0.03*(ymax-ymin), f"{label_key}  DC", ha='center', va='top',
            color=color, fontsize=10)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# Figure 7: UDCMJ Model (Up/Down Converting Multijunction Solar Cell)
dc_Eg = [3.0448, 2.6460]
uc_Eg = [0.5748, 0.7760, 1.2683, 1.0870, 2.1491, 1.3815, 1.7867, 1.1715, 1.9393]
mj_Eg = [3.7504, 3.3853, 3.0706, 2.9152, 2.4568, 2.1922, 1.8306, 1.6264, 1.5013, 1.3906, 0.8321, 0.2483]
# Sort MJ bandgaps in descending order
mj_Eg.sort(reverse=True)
# Calculate bin widths
dλ = np.zeros_like(wavelength_np)
dλ[0] = wavelength_np[1] - wavelength_np[0]
dλ[-1] = wavelength_np[-1] - wavelength_np[-2]
dλ[1:-1] = (wavelength_np[2:] - wavelength_np[:-2]) / 2.0
# Process spectrum through DC and UC layers
I_processed = irradiance_np.copy()
P_in = np.trapezoid(I_processed, wavelength_np)
# Down-conversion processing
for Eg_target in dc_Eg:
    indices_src = np.where(np.abs(E_spectrum - Eg_target) <= 0.05)[0]
    for i in indices_src:
        if I_processed[i] <= 0:
            continue
        E_tgt = E_spectrum[i] / 2.0
        indices_dst = np.where(np.abs(E_spectrum - E_tgt) <= 0.05)[0]
        if len(indices_dst) == 0:
            continue
        I_processed[indices_dst] += I_processed[i] * dλ[i] / np.sum(dλ[indices_dst])
        I_processed[i] = 0
# Up-conversion processing
for Eg_target in uc_Eg:
    indices_src = np.where(np.abs(E_spectrum - Eg_target) <= 0.05)[0]
    for i in indices_src:
        if I_processed[i] <= 0:
            continue
        E_uc = 2 * E_spectrum[i]
        j = np.argmin(np.abs(E_spectrum - E_uc))
        if np.abs(E_spectrum[j] - E_uc) > 0.05:
            continue
        available_photons = I_processed[i] * dλ[i] / (E_spectrum[i] * q)
        conversions = available_photons / 2.0
        energy_removed = 2 * conversions * E_spectrum[i] * q
        energy_added = conversions * E_uc * q
        I_processed[i] -= energy_removed / dλ[i]
        I_processed[j] += energy_added / dλ[j]
# Multijunction cell calculation
P_max_list = []  # Max power for each subcell
eta_i_list = []  # Efficiency of each subcell
for i, Eg_i in enumerate(mj_Eg):
    if i == 0:
        λ_max = h*c/(q*1e-9*Eg_i)
        mask = wavelength_np <= λ_max
    else:
        λ_min = h*c/(q*1e-9*mj_Eg[i-1])
        λ_max = h*c/(q*1e-9*Eg_i)
        mask = (wavelength_np >= λ_min) & (wavelength_np <= λ_max)
    E_cell = h * c / (q * wavelength_np[mask] * 1e-9) 
    photon_flux = np.trapezoid(I_processed[mask]/(q*E_cell), wavelength_np[mask])
    J_ph = CM * q * photon_flux
    J0 = compute_j0(Eg_i)
    if J_ph > 0 and J0 > 0:
        Voc = n*k*T*np.log1p(J_ph/J0)/q 
        V = np.linspace(0, Voc, 500)
        J_total = J_ph - J0 * np.expm1(V * q/(n*k*T))
        power = J_total * V
        max_power = np.max(power) 
    else:
        max_power = 0.0
    P_max_list.append(max_power)
    eta_i_list.append(max_power / P_in * 100.0)
P_collected = sum(P_max_list)
total_eff = P_collected / P_in * 100.0
print(f"Total efficiency with DC+UC: {total_eff:.2f}%")
I_processed_eV = I_processed * (q*1e-9*wavelength**2/(h*c))
E_orig = (h*c) / (q*1e-9*wavelength)
sort_idx = np.argsort(E_orig)
# Figure 7a: Modified spectrum (original simple style)
plt.figure(figsize=(12, 6))
plt.plot(E_orig[sort_idx], I_processed_eV[sort_idx])
plt.ylabel('Irradiance (W/m²/eV)')
plt.xlabel('Energy (eV)')
plt.title('Modified Spectrum After DC and UC Processing')
plt.grid(True)
plt.xlim(0, 4)
plt.ylim(0, None)
plt.tight_layout()
plt.show()
# Figure 7a: Subcell power generation with detailed table (fixed layout)
fig = plt.figure(figsize=(14, 12))
gs = fig.add_gridspec(10, 1)
ax1 = fig.add_subplot(gs[0:7, 0])
x = np.arange(len(mj_Eg)) + 1
bars = ax1.bar(x, P_max_list, color=plt.cm.viridis(np.linspace(0, 1, len(mj_Eg))))
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5, 
             f'Eg={mj_Eg[i]:.2f}eV', 
             ha='center', va='bottom', rotation=0, fontsize=10)
ax1.set_ylabel('Power Density (W/m²)', fontsize=12)
ax1.set_xlabel('Subcell Index', fontsize=12)
ax1.set_title('Power Generation by Subcell', fontsize=14)
ax1.set_xticks(x)
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
ax1.set_axisbelow(True)
ax1.set_ylim(0, max(P_max_list)*1.3)
ax2 = fig.add_subplot(gs[7:10, 0])
cell_text = []
for i in range(len(mj_Eg)):
    cell_text.append([
        f"Subcell {i+1}", 
        f"{mj_Eg[i]:.4f} eV", 
        f"{P_max_list[i]:.4f} W/m²",
        f"{eta_i_list[i]:.2f}%"
    ])
table = ax2.table(
    cellText=cell_text,
    colLabels=['Subcell', 'Bandgap', 'Power Density', 'Efficiency'],
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
ax2.axis('off')
plt.subplots_adjust(hspace=0.4)
plt.tight_layout()
plt.savefig('Figure 7a', dpi = 1200)
plt.show()