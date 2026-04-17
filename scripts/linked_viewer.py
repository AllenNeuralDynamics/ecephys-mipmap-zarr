import panel as pn
from ecephys_mipmap_zarr.read import open_mipmap
from ecephys_mipmap_zarr.viewer import TraceViewer, load_phy_spikes

phy_dir = "/mnt/Data/ephys/phy/ecephys_786867_2025-09-25_12-43-56_sorted_2025-12-24_08-46-14/experiment1_Record Node 111#Neuropix-PXI-100.50213-1_recording1"
store = open_mipmap("/mnt/Data/ephys/test_mipmap_lsb_rounded_v2.zarr")

sd_a, tc_a = load_phy_spikes(phy_dir, store.sampling_frequency, unit_ids=[225], t_start=store.t_start)
sd_b, tc_b = load_phy_spikes(phy_dir, store.sampling_frequency, unit_ids=[309, 210], t_start=store.t_start)

viewer_a = TraceViewer(store, spike_data=sd_a, initial_channels=sorted(set().union(*tc_a.values())))
viewer_b = TraceViewer(store, spike_data=sd_b, initial_channels=sorted(set().union(*tc_b.values())))

# Link time axes
viewer_b._fig.x_range = viewer_a._fig.x_range

pn.Column(viewer_a.panel(), viewer_b.panel()).show()  # opens browser tab
