# coding: utf-8

import stl.mesh
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

def load_stl(path):
	return stl.mesh.Mesh.from_file(path)

def project2D(stl_mesh, elev, azim, linewidths=0.1, alpha=1.0, **kwargs):
	"""
	Project 3D mesh to a 2D image.
	"""
	fig = plt.figure(figsize=(8, 8), tight_layout={"pad":0.0, "w_pad":0.0, "h_pad":0.0, "rect":None})
	ax = fig.add_subplot(111, projection='3d')

	mesh = Poly3DCollection(stl_mesh.vectors, linewidths=linewidths, alpha=alpha)
	mesh.set_facecolor((1, 1, 1))
	ax.add_collection3d(mesh)
	xmax,ymax,zmax = stl_mesh.vectors.max(axis=(0,1))
	ax.set_xlim(0,xmax)
	ax.set_ylim(0,ymax)
	ax.set_zlim(0,zmax)
	ax.set_facecolor((0, 0, 0))
	ax.view_init(elev=elev, azim=azim)
	ax.axis('off')
	fig.tight_layout(pad=0)
	ax.grid(b=None)
	ax.margins(0)

	canvas = FigureCanvasAgg(fig)
	canvas.draw()
	s, (width, height) = canvas.print_to_buffer()
	img = Image.frombytes("RGBA", (width, height), s)

	plt.close(fig)
	return img