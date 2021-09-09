#![allow(unsafe_code)]

use egui::{
    emath::Rect,
    epaint::{Color32, Mesh, Vertex},
};

use std::convert::TryInto;

use glow::HasContext;

const VERT_SRC: &str = include_str!("shader/vertex.glsl");
const FRAG_SRC: &str = include_str!("shader/fragment.glsl");

fn srgbtexture2d(gl: &glow::Context, data: &[u8], w: usize, h: usize) -> glow::NativeTexture {
    assert_eq!(data.len(), w * h * 4);
    assert!(w >= 1);
    assert!(h >= 1);
    unsafe {
        let tex = gl.create_texture().unwrap();
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

        assert_eq!(gl.get_error(), glow::NO_ERROR);
        tex
    }
}

unsafe fn as_u8_slice<T>(s: &[T]) -> &[u8] {
    std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * std::mem::size_of::<T>())
}

pub struct Painter {
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

    old_textures: Vec<glow::NativeTexture>,
    // Only in debug builds, to make sure egui is used correctly
    #[cfg(debug_assertions)]
    destructed: bool,
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
#[allow(dead_code)]
pub enum ShaderVersion {
    Gl120,
    Gl140,
    Es100,
    Es300,
}

impl ShaderVersion {
    fn get(gl: &glow::Context) -> Self {
        let glsl_ver = unsafe { gl.get_parameter_string(glow::SHADING_LANGUAGE_VERSION) };
        let start = glsl_ver.find(|c| char::is_ascii_digit(&c)).unwrap();
        let es = glsl_ver[..start].contains("ES");
        let ver = glsl_ver[start..].splitn(1, ' ').next().unwrap();
        let [maj, min]: [u8; 2] = ver
            .splitn(3, '.')
            .take(2)
            .map(|x| x.parse().unwrap_or_default())
            .collect::<Vec<u8>>()
            .try_into()
            .unwrap();
        // TODO Kept in while PR is WIP, so people can make sure it's detecting the correct version
        eprintln!(
            "{}\n{}.{}{}",
            glsl_ver,
            maj,
            min,
            if es { " ES" } else { "" }
        );
        if es {
            if maj >= 3 {
                Self::Es300
            } else {
                Self::Es100
            }
        } else if maj > 1 || (maj == 1 && min >= 40) {
            Self::Gl140
        } else {
            Self::Gl120
        }
    }

    fn version(&self) -> &'static str {
        match self {
            Self::Gl120 => "#version 120\n",
            Self::Gl140 => "#version 140\n",
            Self::Es100 => "#version 100\n",
            Self::Es300 => "#version 300 es\n",
        }
    }
}

impl Painter {
    pub fn new(gl: &glow::Context) -> Painter {
        let header = ShaderVersion::get(gl).version();
        let v_src = header.to_owned() + VERT_SRC;
        let f_src = header.to_owned() + FRAG_SRC;
        // TODO error handling
        unsafe {
            let v = gl.create_shader(glow::VERTEX_SHADER).unwrap();
            gl.shader_source(v, &v_src);
            gl.compile_shader(v);
            if !gl.get_shader_compile_status(v) {
                panic!(
                    "Failed to compile vertex shader: {}",
                    gl.get_shader_info_log(v)
                );
            }

            let f = gl.create_shader(glow::FRAGMENT_SHADER).unwrap();
            gl.shader_source(f, &f_src);
            gl.compile_shader(f);
            if !gl.get_shader_compile_status(f) {
                panic!(
                    "Failed to compile fragment shader: {}",
                    gl.get_shader_info_log(f)
                );
            }

            let program = gl.create_program().unwrap();
            gl.attach_shader(program, v);
            gl.attach_shader(program, f);
            gl.link_program(program);
            if !gl.get_program_link_status(program) {
                panic!("{}", gl.get_program_info_log(program));
            }
            gl.detach_shader(program, v);
            gl.detach_shader(program, f);
            gl.delete_shader(v);
            gl.delete_shader(f);

            let u_screen_size = gl.get_uniform_location(program, "u_screen_size").unwrap();
            let u_sampler = gl.get_uniform_location(program, "u_sampler").unwrap();

            let va = gl.create_vertex_array().unwrap();
            let vb = gl.create_buffer().unwrap();
            let eb = gl.create_buffer().unwrap();

            debug_assert_eq!(std::mem::size_of::<Vertex>(), 20);

            gl.bind_vertex_array(Some(va));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vb));

            let a_pos_loc = gl.get_attrib_location(program, "a_pos").unwrap();
            let a_tc_loc = gl.get_attrib_location(program, "a_tc").unwrap();
            let a_srgba_loc = gl.get_attrib_location(program, "a_srgba").unwrap();

            gl.vertex_attrib_pointer_f32(
                a_pos_loc,
                2,
                glow::FLOAT,
                false,
                std::mem::size_of::<Vertex>() as i32,
                0,
            );
            gl.enable_vertex_attrib_array(a_pos_loc);

            gl.vertex_attrib_pointer_f32(
                a_tc_loc,
                2,
                glow::FLOAT,
                false,
                std::mem::size_of::<Vertex>() as i32,
                2 * std::mem::size_of::<f32>() as i32,
            );
            gl.enable_vertex_attrib_array(a_tc_loc);

            gl.vertex_attrib_pointer_f32(
                a_srgba_loc,
                4,
                glow::UNSIGNED_BYTE,
                false,
                std::mem::size_of::<Vertex>() as i32,
                4 * std::mem::size_of::<f32>() as i32,
            );
            gl.enable_vertex_attrib_array(a_srgba_loc);
            assert_eq!(gl.get_error(), glow::NO_ERROR);

            Painter {
                program,
                u_screen_size,
                u_sampler,
                egui_texture: None,
                egui_texture_version: None,
                user_textures: Default::default(),
                va,
                vb,
                eb,
                old_textures: Vec::new(),
                #[cfg(debug_assertions)]
                destructed: false,
            }
        }
    }

    pub fn upload_egui_texture(&mut self, gl: &glow::Context, texture: &egui::Texture) {
        #[cfg(debug_assertions)]
        if self.destructed {
            unreachable!();
        }

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
            Some(srgbtexture2d(gl, &pixels, texture.width, texture.height)),
        ) {
            unsafe {
                gl.delete_texture(old_tex);
            }
        }
        self.egui_texture_version = Some(texture.version);
    }

    unsafe fn prepare_painting(
        &mut self,
        display: &glutin::WindowedContext<glutin::PossiblyCurrent>,
        gl: &glow::Context,
        pixels_per_point: f32,
    ) -> (u32, u32) {
        gl.enable(glow::SCISSOR_TEST);
        // egui outputs mesh in both winding orders:
        gl.disable(glow::CULL_FACE);

        gl.enable(glow::BLEND);
        gl.blend_equation(glow::FUNC_ADD);
        gl.blend_func_separate(
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
        } = display.window().inner_size();
        let width_in_points = width_in_pixels as f32 / pixels_per_point;
        let height_in_points = height_in_pixels as f32 / pixels_per_point;

        // TODO maybe don't set here?
        //gl.viewport(0, 0, width_in_pixels as i32, height_in_pixels as i32);
        gl.use_program(Some(self.program));

        // The texture coordinates for text are so that both nearest and linear should work with the egui font texture.
        // For user textures linear sampling is more likely to be the right choice.
        gl.uniform_2_f32(Some(&self.u_screen_size), width_in_points, height_in_points);
        gl.uniform_1_i32(Some(&self.u_sampler), 0);
        gl.active_texture(glow::TEXTURE0);

        gl.bind_vertex_array(Some(self.va));
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vb));
        gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(self.eb));
        (width_in_pixels, height_in_pixels)
    }

    /// Main entry-point for painting a frame.
    /// You should call `target.clear_color(..)` before
    /// and `target.finish()` after this.
    ///
    /// The following OpenGL features will be set:
    /// - Scissor test will be enabled
    /// - Cull face will be disabled
    /// - Blend will be enabled
    ///
    /// The scissor area and blend parameters will be changed.
    ///
    /// As well as this, the following objects will be rebound:
    /// - Vertex Array
    /// - Vertex Buffer
    /// - Element Buffer
    /// - Texture (and active texture will be set to 0)
    /// - Program
    ///
    /// Please be mindful of these effects when integrating into your program, and also be mindful
    /// of the effects your program might have on this code. Look at the source if in doubt.
    pub fn paint_meshes(
        &mut self,
        display: &glutin::WindowedContext<glutin::PossiblyCurrent>,
        gl: &glow::Context,
        pixels_per_point: f32,
        cipped_meshes: Vec<egui::ClippedMesh>,
        egui_texture: &egui::Texture,
    ) {
        #[cfg(debug_assertions)]
        if self.destructed {
            unreachable!();
        }

        self.upload_egui_texture(gl, egui_texture);
        self.upload_pending_user_textures(gl);

        let (w, h) = unsafe { self.prepare_painting(display, gl, pixels_per_point) };
        for egui::ClippedMesh(clip_rect, mesh) in cipped_meshes {
            self.paint_mesh(gl, w, h, pixels_per_point, clip_rect, &mesh)
        }

        assert_eq!(unsafe { gl.get_error() }, glow::NO_ERROR);
    }

    #[inline(never)] // Easier profiling
    fn paint_mesh(
        &mut self,
        gl: &glow::Context,
        width_in_pixels: u32,
        height_in_pixels: u32,
        pixels_per_point: f32,
        clip_rect: Rect,
        mesh: &Mesh,
    ) {
        debug_assert!(mesh.is_valid());

        if let Some(texture) = self.get_texture(mesh.texture_id) {
            unsafe {
                gl.buffer_data_u8_slice(
                    glow::ARRAY_BUFFER,
                    as_u8_slice(mesh.vertices.as_slice()),
                    glow::STREAM_DRAW,
                );

                gl.buffer_data_u8_slice(
                    glow::ELEMENT_ARRAY_BUFFER,
                    as_u8_slice(mesh.indices.as_slice()),
                    glow::STREAM_DRAW,
                );

                gl.bind_texture(glow::TEXTURE_2D, Some(texture));
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
                gl.scissor(
                    clip_min_x,
                    height_in_pixels as i32 - clip_max_y,
                    clip_max_x - clip_min_x,
                    clip_max_y - clip_min_y,
                );
                gl.draw_elements(
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
        #[cfg(debug_assertions)]
        if self.destructed {
            unreachable!();
        }

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
    pub fn register_glow_texture(&mut self, texture: glow::NativeTexture) -> egui::TextureId {
        #[cfg(debug_assertions)]
        if self.destructed {
            unreachable!();
        }

        let id = self.alloc_user_texture();
        if let egui::TextureId::User(id) = id {
            if let Some(Some(user_texture)) = self.user_textures.get_mut(id as usize) {
                if let UserTexture {
                    gl_texture: Some(old_tex),
                    ..
                } = std::mem::replace(
                    user_texture,
                    UserTexture {
                        pixels: vec![],
                        pixels_res: (0, 0),
                        gl_texture: Some(texture),
                    },
                ) {
                    self.old_textures.push(old_tex);
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
        #[cfg(debug_assertions)]
        if self.destructed {
            unreachable!();
        }

        assert_eq!(
            size.0 * size.1,
            pixels.len(),
            "Mismatch between texture size and texel count"
        );

        if let egui::TextureId::User(id) = id {
            if let Some(Some(user_texture)) = self.user_textures.get_mut(id as usize) {
                let pixels: Vec<u8> = pixels
                    .iter()
                    .map(|srgba| Vec::from(srgba.to_array()))
                    .flatten()
                    .collect();

                if let UserTexture {
                    gl_texture: Some(old_tex),
                    ..
                } = std::mem::replace(
                    user_texture,
                    UserTexture {
                        pixels,
                        pixels_res: size,
                        gl_texture: None,
                    },
                ) {
                    self.old_textures.push(old_tex);
                }
            }
        }
    }

    pub fn free_user_texture(&mut self, id: egui::TextureId) {
        #[cfg(debug_assertions)]
        if self.destructed {
            unreachable!();
        }

        if let egui::TextureId::User(id) = id {
            let index = id as usize;
            if index < self.user_textures.len() {
                self.user_textures[index] = None;
            }
        }
    }

    pub fn get_texture(&self, texture_id: egui::TextureId) -> Option<glow::NativeTexture> {
        #[cfg(debug_assertions)]
        if self.destructed {
            unreachable!();
        }

        match texture_id {
            egui::TextureId::Egui => self.egui_texture,
            egui::TextureId::User(id) => self.user_textures.get(id as usize)?.as_ref()?.gl_texture,
        }
    }

    pub fn upload_pending_user_textures(&mut self, gl: &glow::Context) {
        #[cfg(debug_assertions)]
        if self.destructed {
            unreachable!();
        }

        for user_texture in self.user_textures.iter_mut().flatten() {
            if user_texture.gl_texture.is_none() {
                let pixels = std::mem::take(&mut user_texture.pixels);
                user_texture.gl_texture = Some(srgbtexture2d(
                    gl,
                    &pixels,
                    user_texture.pixels_res.0,
                    user_texture.pixels_res.1,
                ));
                user_texture.pixels_res = (0, 0);
            }
        }
        for t in self.old_textures.drain(..) {
            unsafe {
                gl.delete_texture(t);
            }
        }
    }

    unsafe fn destruct_gl(&self, gl: &glow::Context) {
        gl.delete_program(self.program);
        if let Some(tex) = self.egui_texture {
            gl.delete_texture(tex);
        }
        for tex in &self.user_textures {
            if let Some(UserTexture {
                gl_texture: Some(t),
                ..
            }) = tex
            {
                gl.delete_texture(*t);
            }
        }
        gl.delete_vertex_array(self.va);
        gl.delete_buffer(self.vb);
        gl.delete_buffer(self.eb);
        for t in &self.old_textures {
            gl.delete_texture(*t);
        }
    }

    /// This function must be called before Painter is dropped, as Painter has some OpenGL objects
    /// that should be deleted.

    #[cfg(debug_assertions)]
    pub fn destruct(&mut self, gl: &glow::Context) {
        unsafe {
            self.destruct_gl(gl);
        }
        #[cfg(debug_assertions)]
        {
            self.destructed = true;
        }
    }

    #[cfg(not(debug_assertions))]
    pub fn destruct(&self, gl: &glow::Context) {
        unsafe {
            self.destruct_gl(gl);
        }
    }
}

impl Drop for Painter {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        debug_assert!(
            self.destructed,
            "Make sure to destruct() rather than dropping, to avoid leaking OpenGL objects!"
        );
    }
}
