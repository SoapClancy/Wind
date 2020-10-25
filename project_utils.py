from pathlib import Path

cwd = Path().cwd()
project_path_ = Path('/'.join(cwd.parts[:cwd.parts.index('MyProject') + 1]))

WS_POUT_2D_PLOT_KWARGS = {
    'x_lim': (-0.5, 29.5),
    'y_lim': (-0.05, 1.05),
    'x_label': 'Wind Speed [m/s]',
    'y_label': 'Active Power Output [p.u.]'
}

WS_POUT_SCATTER_ALPHA = 0.5

WS_POUT_SCATTER_SIZE = 1

MFR_KWARGS = ({'ws': range(30), 'marker': 's', 's': 12, 'zorder': 300},
              {'ws': range(30), 'marker': '*', 'color': 'black', 's': 16, 'zorder': 300},
              {'ws': range(30), 'marker': '+', 'color': 'lime', 's': 16, 'zorder': 300})
