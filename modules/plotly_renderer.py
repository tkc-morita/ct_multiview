# coding: utf-8

import numpy as np
from plotly.figure_factory import create_trisurf
import trimesh
import io
from PIL import Image

scene = {'{}axis'.format(ax):
					dict(
						title='',
						showgrid=False,
						zeroline=False,
						showticklabels=False,
						linecolor='rgb(0, 0, 0)',
					)
					for ax in 'xyz'}
scene['bgcolor'] = 'rgb(0, 0, 0)'
scene['aspectmode'] = 'data'
def load_stl(path):
	mesh = trimesh.load_mesh(path)
	x,y,z = zip(*mesh.vertices)
	fig = create_trisurf(x=x,
						y=y, 
						z=z, 
						plot_edges=False,
						colormap=['rgb(255,255,255)','rgb(255,255,255)'],
						simplices=mesh.faces,
						backgroundcolor='rgb(0, 0, 0)',
						title='',
						show_colorbar=False,
						)
	fig.update_layout(
			scene=scene,
			margin=dict(r=0, l=0, b=0, t=0),
			)
	fig.update_traces(lighting=dict(ambient=0.5))
	return fig

def project2D(fig, elev, azim, dist=1.8, width=512, height=512, **kwargs):
	elev=np.deg2rad(elev)
	azim=np.deg2rad(azim)
	z=dist*np.sin(elev)
	r_xy=dist*np.cos(elev)
	x=r_xy*np.cos(azim)
	y=r_xy*np.sin(azim)

	fig.update_layout(
			scene_camera_eye=dict(x=x, y=y, z=z),
			)
	fig.update_traces(lightposition=dict(x=x*1.5, y=y*1.5, z=z*1.5))

	img_bytes = fig.to_image(format="png", width=width, height=height)
	img = Image.open(io.BytesIO(img_bytes))
	return img