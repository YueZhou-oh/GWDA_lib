import h5py

source = "EMRI"
# source = 'MBHB'
# source = 'SGWB'
# source = 'VGB'

f = h5py.File(f"test_{source}.hdf5", "r")
all_keys = [key for key in f.keys()]
d = f[all_keys[0]]

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig = plt.figure(figsize=(15, 3), facecolor="white", dpi=100)

ax = fig.add_subplot(111)

ax.plot(d[0, :6000], d[1, :6000], alpha=0.8, label="X channel")
ax.plot(d[0, :6000], d[2, :6000], alpha=0.8, label="Y channel")
ax.plot(d[0, :6000], d[3, :6000], alpha=0.8, label="Z channel")

ax.legend(loc="lower right")
ax.set_xlabel("Time(s)")
ax.set_ylabel("Amplitude")
ax.set_title(f"{source} waveform")

axins = ax.inset_axes((0.2, 0.2, 0.4, 0.3))
pt = 500
axins.plot(d[0, :pt], d[1, :pt])
axins.plot(d[0, :pt], d[2, :pt])
axins.plot(d[0, :pt], d[3, :pt])
ax.indicate_inset_zoom(axins, alpha=0.8)

# plt.savefig(f'{source}_wave.png', dpi=100)
plt.show()
