#![allow(unsafe_code)]

use egui::{
    emath::Rect,
    epaint::{Color32, Mesh, Vertex},
};

use glow::HasContext;

// TODO proper error checking, memory leaks, etc
fn srgbtexture2d(
    gl: &glow::Context,
    data: &[u8],
    w: usize,
    h: usize,
) -> Option<glow::NativeTexture> {
    assert_eq!(data.len(), w * h * 4);
    assert!(w >= 1);
    assert!(h >= 1);
    unsafe {
        let tex = gl.create_texture().ok()?;
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));

        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MAG_FILTER,
            glow::LINEAR as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_S,
            glow::CLAMP_TO_EDGE as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_WRAP_T,
            glow::CLAMP_TO_EDGE as i32,
        );

        gl.tex_storage_2d(glow::TEXTURE_2D, 1, glow::SRGB8_ALPHA8, w as i32, h as i32);
        gl.tex_sub_image_2d(
            glow::TEXTURE_2D,
            0,
            0,
            0,
            w as i32,
            h as i32,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(data),
        );
        gl.bind_texture(glow::TEXTURE_2D, None);
        assert_eq!(gl.get_error(), glow::NO_ERROR);
        Some(tex)
    }
}

unsafe fn as_u8_slice<T>(s: &[T]) -> &[u8] {
    std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * std::mem::size_of::<T>())
}

pub struct Painter<'a> {
    gl: &'a glow::Context,
    program: glow::NativeProgram,
    u_screen_size: glow::UniformLocation,
    u_sampler: glow::UniformLocation,
    egui_texture: Option<glow::NativeTexture>,
    egui_texture_version: Option<u64>,

    /// `None` means unallocated (freed) slot.
    user_textures: Vec<Option<UserTexture>>,

    va: glow::NativeVertexArray,
    vb: glow::NativeBuffer,
    eb: glow::NativeBuffer,
}

#[derive(Default)]
struct UserTexture {
    /// Pending upload (will be emptied later).
    /// This is the format glow likes.
    pixels: Vec<u8>,
    pixels_res: (usize, usize),

    /// Lazily uploaded
    gl_texture: Option<glow::NativeTexture>,
}

#[derive(Copy, Clone, Debug)]
enum ShaderVersion {
    Gl120,
    Gl140,
    Es100,
    Es300,
}

//impl ShaderVersion {
//    fn get(&glow::Context) -> Self {
//        unsafe {
//            let glsl_ver = glow::get_parameter_string(glow::SHADING_LANGUAGE_VERSION);
//            let maj = glsl_ver[0].parse::<u8>();
//            let min = glsl_ver[2].parse::<u8>();
//            if maj < 3 {
//                if min < 4 {
//                    ShaderVersion::Gl120
//                } else {
//                    ShaderVersion::Gl140
//                }
//            } else {
//                ShaderVersion::Es300
//            }
//        }
//    }
//}

impl<'a> Painter<'a> {
    pub fn new(gl: &'a glow::Context) -> Painter<'a> {
        use ShaderVersion::*;
        // TODO detect the version
        let (v_src, f_src) = match Gl140 {
            Gl120 => (
                include_str!("shader/vertex_120.glsl"),
                include_str!("shader/fragment_120.glsl"),
            ),
            Gl140 => (
                include_str!("shader/vertex_140.glsl"),
                include_str!("shader/fragment_140.glsl"),
            ),
            Es100 => (
                include_str!("shader/vertex_100es.glsl"),
                include_str!("shader/fragment_100es.glsl"),
            ),
            Es300 => (
                include_str!("shader/vertex_300es.glsl"),
                include_str!("shader/fragment_300es.glsl"),
            ),
        };

        // TODO error handling
        unsafe {
            let v = gl.create_shader(glow::VERTEX_SHADER).unwrap();
            let f = gl.create_shader(glow::FRAGMENT_SHADER).unwrap();
            gl.shader_source(v, v_src);
            gl.shader_source(f, f_src);
            gl.compile_shader(v);
            gl.compile_shader(f);
            let program = gl.create_program().unwrap();
            gl.attach_shader(program, v);
            gl.attach_shader(program, f);
            gl.link_program(program);
            gl.delete_shader(v);
            gl.delete_shader(f);

            let u_screen_size = gl.get_uniform_location(program, "u_screen_size").unwrap();
            let u_sampler = gl.get_uniform_location(program, "u_sampler").unwrap();

            let va = gl.create_vertex_array().unwrap();
            let vb = gl.create_buffer().unwrap();
            let eb = gl.create_buffer().unwrap();

            gl.bind_vertex_array(Some(va));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vb));

            gl.vertex_attrib_pointer_f32(
                0,
                2,
                glow::FLOAT,
                false,
                std::mem::size_of::<Vertex>() as i32,
                0,
            );
            gl.enable_vertex_attrib_array(0);

            gl.vertex_attrib_pointer_f32(
                2,
                2,
                glow::FLOAT,
                false,
                std::mem::size_of::<Vertex>() as i32,
                2 * std::mem::size_of::<f32>() as i32,
            );
            gl.enable_vertex_attrib_array(2);

            gl.vertex_attrib_pointer_f32(
                1,
                4,
                glow::UNSIGNED_BYTE,
                false,
                std::mem::size_of::<Vertex>() as i32,
                4 * std::mem::size_of::<f32>() as i32,
            );
            gl.enable_vertex_attrib_array(1);

            Painter {
                gl,
                program,
                u_screen_size,
                u_sampler,
                egui_texture: None,
                egui_texture_version: None,
                user_textures: Default::default(),
                va,
                vb,
                eb,
            }
        }
    }

    pub fn upload_egui_texture(&mut self, texture: &egui::Texture) {
        if self.egui_texture_version == Some(texture.version) {
            return; // No change
        }

        let pixels: Vec<u8> = texture
            .pixels
            .iter()
            .map(|a| Vec::from(Color32::from_white_alpha(*a).to_array()))
            .flatten()
            .collect();

        if let Some(old_tex) = std::mem::replace(
            &mut self.egui_texture,
            Some(srgbtexture2d(self.gl, &pixels, texture.width, texture.height).unwrap()),
        ) {
            unsafe {
                self.gl.delete_texture(old_tex);
            }
        }
        self.egui_texture_version = Some(texture.version);
    }

    unsafe fn prepare_painting(
        &mut self,
        gl_window: &glutin::WindowedContext<glutin::PossiblyCurrent>,
        pixels_per_point: f32,
    ) -> (u32, u32) {
        self.gl.enable(glow::FRAMEBUFFER_SRGB);
        self.gl.enable(glow::SCISSOR_TEST);
        // egui outputs mesh in both winding orders:
        self.gl.disable(glow::CULL_FACE);

        self.gl.enable(glow::BLEND);
        self.gl.blend_equation(glow::FUNC_ADD);
        self.gl.blend_func_separate(
            // egui outputs colors with premultiplied alpha:
            glow::ONE,
            glow::ONE_MINUS_SRC_ALPHA,
            // Less important, but this is technically the correct alpha blend function
            // when you want to make use of the framebuffer alpha (for screenshots, compositing, etc).
            glow::ONE_MINUS_DST_ALPHA,
            glow::ONE,
        );

        let glutin::dpi::PhysicalSize {
            width: width_in_pixels,
            height: height_in_pixels,
        } = gl_window.window().inner_size();
        let width_in_points = width_in_pixels as f32 / pixels_per_point;
        let height_in_points = height_in_pixels as f32 / pixels_per_point;

        //gl.viewport(0, 0, width_in_pixels as i32, height_in_pixels as i32);
        self.gl.use_program(Some(self.program));

        // The texture coordinates for text are so that both nearest and linear should work with the egui font texture.
        // For user textures linear sampling is more likely to be the right choice.
        self.gl
            .uniform_2_f32(Some(&self.u_screen_size), width_in_points, height_in_points);
        self.gl.uniform_1_i32(Some(&self.u_sampler), 0);
        self.gl.active_texture(glow::TEXTURE0);

        self.gl.bind_vertex_array(Some(self.va));
        self.gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vb));
        self.gl
            .bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(self.eb));
        (width_in_pixels, height_in_pixels)
    }

    /// Main entry-point for painting a frame.
    /// You should call `target.clear_color(..)` before
    /// and `target.finish()` after this.
    pub fn paint_meshes(
        &mut self,
        gl_window: &glutin::WindowedContext<glutin::PossiblyCurrent>,
        pixels_per_point: f32,
        cipped_meshes: Vec<egui::ClippedMesh>,
        egui_texture: &egui::Texture,
    ) {
        self.upload_egui_texture(egui_texture);
        self.upload_pending_user_textures();

        let (w, h) = unsafe { self.prepare_painting(gl_window, pixels_per_point) };
        for egui::ClippedMesh(clip_rect, mesh) in cipped_meshes {
            self.paint_mesh_no_setup(w, h, pixels_per_point, clip_rect, &mesh)
        }
    }

    // I don't think this needs to exist as pub, it's just not very useful when it can be included
    // in paint_meshes
    // No need for additional complexity - especially when egui_web does not have this pub
    #[deprecated]
    pub fn paint_mesh(
        &mut self,
        gl_window: &glutin::WindowedContext<glutin::PossiblyCurrent>,
        pixels_per_point: f32,
        clip_rect: Rect,
        mesh: &Mesh,
    ) {
        let (w, h) = unsafe { self.prepare_painting(gl_window, pixels_per_point) };
        self.paint_mesh_no_setup(w, h, pixels_per_point, clip_rect, &mesh)
    }

    #[inline(never)] // Easier profiling
    fn paint_mesh_no_setup(
        &mut self,
        width_in_pixels: u32,
        height_in_pixels: u32,
        pixels_per_point: f32,
        clip_rect: Rect,
        mesh: &Mesh,
    ) {
        debug_assert!(mesh.is_valid());

        if let Some(texture) = self.get_texture(mesh.texture_id) {
            unsafe {
                self.gl.buffer_data_u8_slice(
                    glow::ARRAY_BUFFER,
                    as_u8_slice(mesh.vertices.as_slice()),
                    glow::STREAM_DRAW,
                );

                self.gl.buffer_data_u8_slice(
                    glow::ELEMENT_ARRAY_BUFFER,
                    as_u8_slice(mesh.indices.as_slice()),
                    glow::STREAM_DRAW,
                );

                self.gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            }
            // Transform clip rect to physical pixels:
            let clip_min_x = pixels_per_point * clip_rect.min.x;
            let clip_min_y = pixels_per_point * clip_rect.min.y;
            let clip_max_x = pixels_per_point * clip_rect.max.x;
            let clip_max_y = pixels_per_point * clip_rect.max.y;

            // Make sure clip rect can fit within a `u32`:
            let clip_min_x = clip_min_x.clamp(0.0, width_in_pixels as f32);
            let clip_min_y = clip_min_y.clamp(0.0, height_in_pixels as f32);
            let clip_max_x = clip_max_x.clamp(clip_min_x, width_in_pixels as f32);
            let clip_max_y = clip_max_y.clamp(clip_min_y, height_in_pixels as f32);

            let clip_min_x = clip_min_x.round() as i32;
            let clip_min_y = clip_min_y.round() as i32;
            let clip_max_x = clip_max_x.round() as i32;
            let clip_max_y = clip_max_y.round() as i32;

            unsafe {
                self.gl.scissor(
                    clip_min_x,
                    height_in_pixels as i32 - clip_max_y,
                    clip_max_x - clip_min_x,
                    clip_max_y - clip_min_y,
                );
                self.gl.draw_elements(
                    glow::TRIANGLES,
                    mesh.indices.len() as i32,
                    glow::UNSIGNED_INT,
                    0,
                );
            }
        }
    }

    // ------------------------------------------------------------------------
    // user textures: this is an experimental feature.
    // No need to implement this in your egui integration!

    pub fn alloc_user_texture(&mut self) -> egui::TextureId {
        for (i, tex) in self.user_textures.iter_mut().enumerate() {
            if tex.is_none() {
                *tex = Some(Default::default());
                return egui::TextureId::User(i as u64);
            }
        }
        let id = egui::TextureId::User(self.user_textures.len() as u64);
        self.user_textures.push(Some(Default::default()));
        id
    }

    /// register glow texture as egui texture
    /// Usable for render to image rectangle
    pub unsafe fn register_glow_texture(
        &mut self,
        texture: glow::NativeTexture,
    ) -> egui::TextureId {
        let id = self.alloc_user_texture();
        if let egui::TextureId::User(id) = id {
            if let Some(Some(user_texture)) = self.user_textures.get_mut(id as usize) {
                if let Some(old_tex) = std::mem::replace(
                    user_texture,
                    UserTexture {
                        pixels: vec![],
                        pixels_res: (0, 0),
                        gl_texture: Some(texture),
                    },
                )
                .gl_texture
                {
                    self.gl.delete_texture(old_tex);
                }
            }
        }
        id
    }

    pub fn set_user_texture(
        &mut self,
        id: egui::TextureId,
        size: (usize, usize),
        pixels: &[Color32],
    ) {
        assert_eq!(
            size.0 * size.1,
            pixels.len(),
            "Mismatch between texture size and texel count"
        );
        assert!(size.0 >= 1 && size.1 >= 1, "Empty texture");

        if let egui::TextureId::User(id) = id {
            if let Some(Some(user_texture)) = self.user_textures.get_mut(id as usize) {
                let pixels: Vec<u8> = pixels
                    .iter()
                    .map(|srgba| Vec::from(srgba.to_array()))
                    .flatten()
                    .collect();

                if let Some(old_tex) = std::mem::replace(
                    user_texture,
                    UserTexture {
                        pixels,
                        pixels_res: size,
                        gl_texture: None,
                    },
                )
                .gl_texture
                {
                    unsafe {
                        self.gl.delete_texture(old_tex);
                    }
                }
            }
        }
    }

    pub fn free_user_texture(&mut self, id: egui::TextureId) {
        if let egui::TextureId::User(id) = id {
            let index = id as usize;
            if index < self.user_textures.len() {
                self.user_textures[index] = None;
            }
        }
    }

    pub fn get_texture(&self, texture_id: egui::TextureId) -> Option<glow::NativeTexture> {
        match texture_id {
            egui::TextureId::Egui => self.egui_texture,
            egui::TextureId::User(id) => self.user_textures.get(id as usize)?.as_ref()?.gl_texture,
        }
    }

    pub fn upload_pending_user_textures(&mut self) {
        for user_texture in self.user_textures.iter_mut().flatten() {
            if user_texture.gl_texture.is_none() {
                let pixels = std::mem::take(&mut user_texture.pixels);
                user_texture.gl_texture = Some(
                    srgbtexture2d(
                        self.gl,
                        &pixels,
                        user_texture.pixels_res.0,
                        user_texture.pixels_res.1,
                    )
                    .unwrap(),
                );
                user_texture.pixels_res = (0, 0);
            }
        }
    }
}

impl Drop for Painter<'_> {
    fn drop(&mut self) {
        todo!()
    }
}
