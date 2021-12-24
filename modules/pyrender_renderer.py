# coding: utf-8

import numpy as np
import trimesh
import pyrender
from scipy.spatial.transform import Rotation
from PIL import Image

def load_stl(path):
	mesh = trimesh.load_mesh(path)
	mesh.vertices /= np.max(mesh.extents)
	mesh.vertices -= mesh.center_mass
	mesh.visual.face_colors = [255,255,255]
	mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
	return mesh

def project2D(mesh, elev, azim, yfov=np.pi/4.5, light_intensity=5.0, img_size=512, **kwargs):
	# mesh = mesh.copy()
	rot = Rotation.from_euler('zx', [-90-azim, elev], degrees=True)
	trans = np.eye(4)
	try:
		trans[:3,:3] = rot.as_matrix()
	except:
		trans[:3,:3] = rot.as_dcm() # as_matrix is too new.
	scene = pyrender.Scene(bg_color=[0.0]*3+[1.0])#, ambient_light=[1., 1., 1.])
	scene.add(mesh, pose=trans)
	camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
	scene.add(camera, pose=camera_pose)
	light = pyrender.DirectionalLight(color=np.ones(3), intensity=light_intensity)
	scene.add(light, pose=camera_pose)
	renderer = pyrender.OffscreenRenderer(img_size, img_size)
	color, depth = renderer.render(scene)
	img = Image.fromarray(color, 'RGB')
	return img

camera_pose = np.eye(4)
try:
	camera_pose[:3,:3] = Rotation.from_euler('x', 90, degrees=True).as_matrix()
except:
	camera_pose[:3,:3] = Rotation.from_euler('x', 90, degrees=True).as_dcm()
dist = 1.8
camera_pose[:3,3] = dist * np.array([0.0, -1.0, 0.0])