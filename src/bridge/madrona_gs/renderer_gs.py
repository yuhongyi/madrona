import numpy as np
import torch
import taichi as ti

from madrona_gs._madrona_gs_batch_renderer import MadronaBatchRenderer
from madrona_gs._madrona_gs_batch_renderer.madrona import ExecMode
from trimesh.visual.texture import TextureVisuals
from PIL import Image

class BatchRendererGS:
  """Wraps Genesis Model around MadronaBatchRenderer."""

  def __init__(
      self,
      rigid,
      gpu_id,
      num_worlds,
      cameras,
      num_lights,
      lights_pos,
      lights_dir,
      lights_intensity,
      lights_directional,
      lights_castshadow,
      lights_cutoff,
      batch_render_view_width=128,
      batch_render_view_height=128,
      add_cam_debug_geo=False,
      use_rasterizer=False,
  ):
    assert rigid is not None, "Rigid body model is required for BatchRendererGS"
    assert gpu_id >= 0, "GPU ID must be greater than or equal to 0"
    assert num_worlds > 0, "Number of worlds must be greater than 0"
    assert batch_render_view_width > 0, "Batch render view width must be greater than 0"
    assert batch_render_view_height > 0, "Batch render view height must be greater than 0"

    # defaults
    default_geom_group = 2
    default_enabled_geom_groups=np.array([default_geom_group])

    self.rigid = rigid
    self.num_worlds = num_worlds
    self.cameras = cameras
    self.lights_pos = lights_pos
    self.lights_dir = lights_dir
    self.lights_intensity = lights_intensity
    self.lights_directional = lights_directional
    self.lights_castshadow = lights_castshadow
    self.lights_cutoff = lights_cutoff

    mesh_vertices = rigid.vverts_info.init_pos.to_numpy()
    mesh_faces = rigid.vfaces_info.vverts_idx.to_numpy()
    mesh_vertex_offsets = rigid.vgeoms_info.vvert_start.to_numpy()
    mesh_face_offsets = rigid.vgeoms_info.vface_start.to_numpy()
    n_vgeom = rigid.n_vgeoms

    face_start = rigid.vgeoms_info.vface_start.to_numpy()
    face_end = rigid.vgeoms_info.vface_end.to_numpy()
    for i in range(n_vgeom):
      mesh_faces[face_start[i] : face_end[i]] -= mesh_vertex_offsets[i]

    geom_types = np.full((n_vgeom,), 7, dtype=np.int32)
    geom_groups = np.full((n_vgeom,), default_geom_group, dtype=np.int32)
    geom_data_ids = np.arange(n_vgeom, dtype=np.int32)
    geom_sizes = np.ones((n_vgeom, 3), dtype=np.float32)
    num_cams = len(cameras) if cameras is not None else 0
    assert num_cams > 0, "Must have at least one camera for Madrona to work!"
    geom_rgba = rigid.vgeoms_info.color.to_numpy()

    geom_mat_ids, mesh_texcoord_num, mesh_texcoord_offsets, mesh_texcoord_data, texture_widths, texture_heights, texture_nchans, texture_data, texture_offsets, material_texture_ids, material_rgba = self.get_texture_data()

    # TODO: Support mutable camera fov
    cam_fovy = np.array([cam.fov for cam in cameras], dtype=np.float32)

    self.madrona = MadronaBatchRenderer(
        gpu_id=gpu_id,
        mesh_vertices=mesh_vertices,
        mesh_faces=mesh_faces,
        mesh_vertex_offsets=mesh_vertex_offsets,
        mesh_face_offsets=mesh_face_offsets,
        mesh_texcoords=mesh_texcoord_data,
        mesh_texcoord_offsets=mesh_texcoord_offsets,
        mesh_texcoord_num=mesh_texcoord_num,
        geom_types=geom_types,
        geom_groups=geom_groups,
        geom_data_ids=geom_data_ids,
        geom_sizes=geom_sizes,
        geom_mat_ids=geom_mat_ids,
        geom_rgba=geom_rgba,
        mat_rgba=material_rgba,
        mat_tex_ids=material_texture_ids,

        tex_data=texture_data,
        tex_offsets=texture_offsets,
        tex_widths=texture_widths,
        tex_heights=texture_heights,
        tex_nchans=texture_nchans,
        num_lights=num_lights,
        num_cams=num_cams,
        num_worlds=num_worlds,

        batch_render_view_width=batch_render_view_width,
        batch_render_view_height=batch_render_view_height,
        cam_fovy=cam_fovy,
        enabled_geom_groups=default_enabled_geom_groups,
        add_cam_debug_geo=add_cam_debug_geo,
        use_rt=not use_rasterizer,
    )
    

  def init(self):
    cam_pos, cam_rot = self.get_camera_pos_rot_torch()
    geom_pos, geom_rot = self.get_geom_pos_rot_torch()
    geom_mat_ids, geom_rgb, geom_sizes = self.get_geom_properties_torch()
    light_pos, light_dir, light_directional, light_castshadow, light_cutoff, light_intensity = self.get_lights_properties_torch()

    # Make a copy to actually shuffle the memory layout before passing to C++
    self.madrona.init(
      geom_pos,
      geom_rot,
      cam_pos,
      cam_rot,
      geom_mat_ids,
      geom_rgb,
      geom_sizes,
      light_pos,
      light_dir,
      light_directional,
      light_castshadow,
      light_cutoff,
      light_intensity,
    )


  def render(self):
    # Assume execution on GPU
    # TODO: Need to check if the device is GPU or CPU, or assert if not GPU
    cam_pos, cam_rot = self.get_camera_pos_rot_torch()
    geom_pos, geom_rot = self.get_geom_pos_rot_torch()

    #self.madrona.render_dummy()
    self.madrona.render_torch(
        geom_pos,
        geom_rot,
        cam_pos,
        cam_rot,
    )
    rgb_torch = self.madrona.rgb_tensor().to_torch()
    depth_torch = self.madrona.depth_tensor().to_torch()    
    return rgb_torch, depth_torch
  
  def destroy(self):
    self.rigid = None
    self.cameras = None
    self.lights_pos = None
    self.lights_dir = None
    self.lights_intensity = None
    self.lights_directional = None
    self.lights_castshadow = None
    self.lights_cutoff = None

  def get_texture_data(self):
    n_vgeom = self.rigid.n_vgeoms
    vgeoms = self.rigid.vgeoms

    # get number of textures, total texcoord and texture data size
    num_textures = 0
    total_texcoord_data_size = 0
    total_texture_data_size = 0
    for geomIdx, vgeom in enumerate(vgeoms):
      visual = vgeom.get_trimesh().visual
      if(isinstance(visual, TextureVisuals)):
        total_texcoord_data_size += visual.uv.shape[0]
        texture_width = visual.material.image.width
        texture_height = visual.material.image.height
        texture_nchans = 4 if visual.material.image.mode == "RGBA" else 3
        texture_data_size = texture_width * texture_height * texture_nchans
        total_texture_data_size += texture_data_size
        num_textures += 1

    # allocate memory for texcoord data
    geom_mat_ids = np.full(n_vgeom, -1, dtype=np.int32)
    mesh_texcoord_num = np.full(n_vgeom, 0, dtype=np.int32)
    mesh_texcoord_offsets = np.full(n_vgeom, -1, dtype=np.int32)
    texcoord_data = np.empty((total_texcoord_data_size, 2), dtype=np.float32)
    texture_widths = np.empty(num_textures, dtype=np.int32)
    texture_heights = np.empty(num_textures, dtype=np.int32)
    texture_nchans = np.empty(num_textures, dtype=np.int32)
    texture_data = np.empty(total_texture_data_size, dtype=np.uint8)
    texture_offsets = np.empty(num_textures, dtype=np.int32)
    num_textures_per_material = 10 # Madrona allows up to 10 textures per material
    material_texture_ids = np.full((num_textures, num_textures_per_material), -1, dtype=np.int32)
    material_rgba = np.empty((num_textures, 4), dtype=np.float32)

    num_textures = 0 # reset index
    total_texcoord_data_size = 0 # reset size
    total_texture_data_size = 0 # reset size
    for geomIdx, vgeom in enumerate(vgeoms):
      visual = vgeom.get_trimesh().visual
      if(isinstance(visual, TextureVisuals)):
        # Copy texcoord data
        uv_size = visual.uv.shape
        mesh_texcoord_num[geomIdx] = uv_size[0]
        mesh_texcoord_offsets[geomIdx] = total_texcoord_data_size
        total_texcoord_data_size += mesh_texcoord_num[geomIdx]
        texcoord_data[mesh_texcoord_offsets[geomIdx] : mesh_texcoord_offsets[geomIdx] + mesh_texcoord_num[geomIdx]] = visual.uv.astype(np.float32)

        # Copy texture data
        texture_widths[num_textures] = visual.material.image.width
        texture_heights[num_textures] = visual.material.image.height
        texture_nchans[num_textures] = 4 if visual.material.image.mode == "RGBA" else 3
        texture_data_size = texture_widths[num_textures] * texture_heights[num_textures] * texture_nchans[num_textures]
        texture_offsets[num_textures] = total_texture_data_size
        total_texture_data_size += texture_data_size
        texture_data[texture_offsets[num_textures] : texture_offsets[num_textures] + texture_data_size] = np.array(list(visual.material.image.transpose(method=Image.Transpose.FLIP_TOP_BOTTOM).getdata()), dtype=np.uint8).flatten()

        # Set material id
        geom_mat_ids[geomIdx] = num_textures
        material_texture_ids[num_textures, 0] = num_textures # Use first texture as diffuse
        material_rgba[num_textures] = visual.material.diffuse.astype(np.float32) / 255.0

        # Bump texture index
        num_textures += 1

    return geom_mat_ids, mesh_texcoord_num, mesh_texcoord_offsets, texcoord_data, texture_widths, texture_heights, texture_nchans, texture_data, texture_offsets, material_texture_ids, material_rgba

########################## Utils ##########################  
  def get_camera_pos_rot_torch(self):
    # TODO: Consider making cam.pos and cam.quat_for_madrona as torch tensors,
    # but it would be tricky to concatenate them into camera_pos and camera_rot
    # might need a ti.kernel to do this, but not sure how fast that will be,
    # since ti.kernel functions are blocking
    cam_pos = ti.ndarray(dtype=ti.f32, shape=(len(self.cameras), 3))
    cam_rot = ti.ndarray(dtype=ti.f32, shape=(len(self.cameras), 4))
    for i, cam in enumerate(self.cameras):
        cam_pos[i] = np.array(cam.pos)
        cam_rot[i] = np.array(cam.quat_for_madrona)

    cam_pos = torch.tensor(cam_pos).to("cuda")
    cam_rot = torch.tensor(cam_rot).to("cuda")
    return cam_pos, cam_rot
  
  def get_geom_pos_rot_torch(self):
    geom_pos = self.rigid.vgeoms_state.pos.to_torch()
    geom_rot = self.rigid.vgeoms_state.quat.to_torch()
    geom_pos = geom_pos.transpose(0, 1).contiguous().to("cuda")
    geom_rot = geom_rot.transpose(0, 1).contiguous().to("cuda")
    return geom_pos, geom_rot
  
  def get_geom_properties_torch(self):
    geom_mat_ids = ti.zeros((self.rigid.n_vgeoms,), dtype=ti.i32) - 1
    geom_sizes = ti.ones((self.rigid.n_vgeoms, 3), dtype=ti.f32)

    geom_rgb_ti = ti.ndarray(dtype=ti.u32, shape=(self.rigid.n_vgeoms,))
    for i in range(self.rigid.n_vgeoms):
      rgba_uint = ti.cast(self.rigid.vgeoms_info.color[i] * 255, ti.u32)
      geom_rgb_ti[i] = rgba_uint[0] * (1 << 16) + rgba_uint[1] * (1 << 8) + rgba_uint[2]
    geom_rgb = geom_rgb_ti.to_torch()

    geom_mat_ids = geom_mat_ids.repeat(self.num_worlds)
    geom_rgb = geom_rgb.repeat(self.num_worlds)
    geom_sizes = geom_sizes.repeat(self.num_worlds, 1)
    return geom_mat_ids, geom_rgb, geom_sizes
  
  def get_lights_properties_torch(self):
    light_pos = self.lights_pos.to_torch().reshape(-1, 3).repeat(self.num_worlds, 1)
    light_dir = self.lights_dir.to_torch().reshape(-1, 3).repeat(self.num_worlds, 1)
    light_directional = self.lights_directional.to_torch().reshape(-1).repeat(self.num_worlds)
    light_castshadow = self.lights_castshadow.to_torch().reshape(-1).repeat(self.num_worlds)
    light_cutoff = self.lights_cutoff.to_torch().reshape(-1).repeat(self.num_worlds)
    light_intensity = self.lights_intensity.to_torch().reshape(-1).repeat(self.num_worlds)
    return light_pos, light_dir, light_directional, light_castshadow, light_cutoff, light_intensity
