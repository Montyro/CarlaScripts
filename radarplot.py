import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

theta = radar_factory(19, frame='circle')
fig, ax = plt.subplots(figsize=(9, 9), nrows=1, ncols=1,
                        subplot_kw=dict(projection='radar'))
ax.set_varlabels(['car ','bicycle ','motorcycle ','truck ','other-vehicle','person ','bicyclist ','motorcyclist ','road','parking ','sidewalk ','other-ground ','building ','fence ','vegetation ','trunk ','terrain ','pole ','traffic-sign'])

#baseline = [95.7,47.7,49.5,47.2,48.3,64.1,66.7,48.2,88.5,57.7,70.7,23.2,90.1,63.9,84.5,67.7,69.0,53.1,62.1,63.0]
#baseline2= [95.7,45.6,44.5,48.0,47.6,62.6,68.6,59.1,88.8,58.3,71.1,26.8,90.3,64.7,84.2,66.6,68.1,53.2,62.4,63.5]
#baseline = [95.7,47.7,49.5,47.2,48.3,64.1,66.7,48.2,88.5,58.4]
#baseline2= [95.7,45.6,44.5,48.0,47.6,62.6,68.6,59.1,88.8,59.0]

baseline = [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
lt= [100,95.6,89.9,101.7,98.6,97.7,102.8,122.6,100.3,101.2,100.6,115.5,100.2,101.3,99.6,98.4,98.7,100.2,100.5]
full = [100.1,100.0,95.4,102.5,101.4,98.6,104.5,101.7,100.5,101.4,101.0,103.4,99.8,99.5,99.9,99.3,99.6,101.7,100.8]

ax.set_rlabel_position(theta[0] * 180/np.pi)
ax.get_yaxis().set_tick_params(labelsize=8)
ax.get_xaxis().set_tick_params(pad=20)

ax.set_ylim(70,125)

ax.plot(theta,baseline , color='C0',label='SPVCNN')
ax.fill(theta,baseline , facecolor='C0', alpha=0.1, label='_nolegend_')
ax.plot(theta,full , color='C2',label='SPVCNN-F',linestyle='-.')
ax.fill(theta,full , facecolor='C2', alpha=0.05, label='_nolegend_')
ax.plot(theta,lt , color='C1',label='SPVCNN-LT',linestyle='--')
ax.fill(theta,lt , facecolor='C1', alpha=0.05, label='_nolegend_')

plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fancybox=True, shadow=True)
plt.savefig("radarplot_spvcnn.png", dpi=600, bbox_inches='tight')




theta = radar_factory(19, frame='circle')
fig, ax = plt.subplots(figsize=(9, 9), nrows=1, ncols=1,
                        subplot_kw=dict(projection='radar'))
ax.set_varlabels(['car ','bicycle ','motorcycle ','truck','other-vehicle','person ','bicyclist ','motorcyclist ','road','parking ','sidewalk ','other-ground ','building ','fence ','vegetation ','trunk ','terrain ','pole ','traffic-sign'])


baseline = [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
lt= [103.7,129.4,105.9,166.6,128.6,104.4,97.9,145.9,99.0,98.9,100.3,90.9,101.8,104.4,100.7,95.2,104.5,104.6,110.5]
full = [103.4,142.5,113.8,113.5,117.3,112.4,100.9,66.7,99.2,97.0,100.7,99.5,100.8,103.0,101.3,101.8,103.3,102.3,115.4]


ax.set_rlabel_position(theta[0] * 180/np.pi)
ax.get_yaxis().set_tick_params(labelsize=8)
ax.get_xaxis().set_tick_params(pad=20)

ax.set_ylim(60,170)

ax.plot(theta,baseline , color='C0',label='SSV3')
ax.fill(theta,baseline , facecolor='C0', alpha=0.1, label='_nolegend_')
ax.plot(theta,full , color='C2',label='SSV3-F',linestyle='-.')
ax.fill(theta,full , facecolor='C2', alpha=0.05, label='_nolegend_')
ax.plot(theta,lt , color='C1',label='SSV3-LT',linestyle='--')
ax.fill(theta,lt , facecolor='C1', alpha=0.05, label='_nolegend_')

plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fancybox=True, shadow=True)
plt.savefig("radarplot_ssqv3.png", dpi=600, bbox_inches='tight')

theta = radar_factory(18, frame='circle')
fig, ax = plt.subplots(figsize=(9, 9), nrows=1, ncols=1,
                        subplot_kw=dict(projection='radar'))
ax.set_varlabels(['car ','bicycle ','motorcycle ','other-vehicle','person ','bicyclist ','motorcyclist ','road','parking ','sidewalk ','other-ground ','building ','fence ','vegetation ','trunk ','terrain ','pole ','traffic-sign'])


baseline = [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]
lt= [103.7,129.4,105.9,128.6,104.4,97.9,145.9,99.0,98.9,100.3,90.9,101.8,104.4,100.7,95.2,104.5,104.6,110.5]
full = [103.4,142.5,113.8,117.3,112.4,100.9,66.7,99.2,97.0,100.7,99.5,100.8,103.0,101.3,101.8,103.3,102.3,115.4]


ax.set_rlabel_position(theta[0] * 180/np.pi)
ax.get_yaxis().set_tick_params(labelsize=8)
ax.get_xaxis().set_tick_params(pad=20)

ax.set_ylim(60,150)

ax.plot(theta,baseline , color='C0',label='SSV3')
ax.fill(theta,baseline , facecolor='C0', alpha=0.1, label='_nolegend_')
ax.plot(theta,full , color='C2',label='SSV3-F',linestyle='-.')
ax.fill(theta,full , facecolor='C2', alpha=0.05, label='_nolegend_')
ax.plot(theta,lt , color='C1',label='SSV3-LT',linestyle='--')
ax.fill(theta,lt , facecolor='C1', alpha=0.05, label='_nolegend_')

plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fancybox=True, shadow=True)
plt.savefig("radarplot_ssqv3_notrcuk.png", dpi=600, bbox_inches='tight')

