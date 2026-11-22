//! wgpu resources and render passes for the viewport.

use super::geom::{GpuVertex, LineVertex};
use bytemuck::{Pod, Zeroable};

/// Format egui samples (gamma-space bytes, no sRGB decode). We render through an sRGB view of it.
pub(crate) const TEX_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
pub(crate) const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
pub(crate) const SHADOW_RES: u32 = 512;
pub(crate) const MSAA: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct Uniforms {
    pub view_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub base_color: [f32; 4],
    pub back_color: [f32; 4],
    pub ghost_color: [f32; 4],
    pub plate_color: [f32; 4],
    pub line_color: [f32; 4],
    pub wire_color: [f32; 4],
    pub slice_point: [f32; 4],
    pub slice_normal: [f32; 4],
    pub scalar: [f32; 4],
    pub flags: [f32; 4],
    pub ground_origin: [f32; 4],
    pub ground_u: [f32; 4],
    pub ground_v: [f32; 4],
    pub up: [f32; 4],
    pub viewport: [f32; 4],
    pub bg_top: [f32; 4],
    pub bg_bottom: [f32; 4],
    pub ground_color: [f32; 4],
}

/// Colour (MSAA + resolve) and depth attachments at one size.
pub(crate) struct Target {
    pub width: u32,
    pub height: u32,
    msaa_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    pub resolve: wgpu::Texture,
    resolve_srgb_view: wgpu::TextureView,
    /// Non-sRGB view of the resolved texture, for egui.
    pub egui_view: wgpu::TextureView,
}

impl Target {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let msaa = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport msaa color"),
            size,
            mip_level_count: 1,
            sample_count: MSAA,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport depth"),
            size,
            mip_level_count: 1,
            sample_count: MSAA,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let resolve = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport resolve"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TEX_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[COLOR_FORMAT],
        });
        let resolve_srgb_view = resolve.create_view(&wgpu::TextureViewDescriptor {
            label: Some("viewport resolve srgb view"),
            format: Some(COLOR_FORMAT),
            ..Default::default()
        });
        let egui_view = resolve.create_view(&wgpu::TextureViewDescriptor {
            label: Some("viewport resolve egui view"),
            format: Some(TEX_FORMAT),
            ..Default::default()
        });
        Self {
            width,
            height,
            msaa_view: msaa.create_view(&Default::default()),
            depth_view: depth.create_view(&Default::default()),
            resolve,
            resolve_srgb_view,
            egui_view,
        }
    }
}

/// A vertex buffer that grows (never shrinks) and tracks how many vertices are valid.
pub(crate) struct DynBuffer {
    buf: Option<wgpu::Buffer>,
    capacity: u64,
    pub count: u32,
    label: &'static str,
}

impl DynBuffer {
    fn new(label: &'static str) -> Self {
        Self {
            buf: None,
            capacity: 0,
            count: 0,
            label,
        }
    }

    /// Upload `data` (vertices); reallocates only if it does not fit.
    pub fn upload<T: Pod>(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[T]) {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let needed = (bytes.len() as u64).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
        if needed > self.capacity || self.buf.is_none() {
            let cap = needed.max(self.capacity + self.capacity / 2).max(256);
            self.buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: cap,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.capacity = cap;
        }
        // all vertex types are multiples of 4 bytes, as write_buffer requires
        debug_assert_eq!(bytes.len() as u64 % wgpu::COPY_BUFFER_ALIGNMENT, 0);
        if let Some(buf) = &self.buf
            && !bytes.is_empty()
        {
            queue.write_buffer(buf, 0, bytes);
        }
        self.count = data.len() as u32;
    }

    fn slice(&self) -> Option<wgpu::BufferSlice<'_>> {
        let buf = self.buf.as_ref()?;
        if self.count == 0 {
            return None;
        }
        Some(buf.slice(..))
    }
}

pub(crate) struct Gpu {
    uniform_buf: wgpu::Buffer,
    main_bind_group: wgpu::BindGroup,
    blur_h_bind_group: wgpu::BindGroup,
    blur_v_bind_group: wgpu::BindGroup,
    shadow_a: wgpu::TextureView,
    shadow_b: wgpu::TextureView,
    shadow_c: wgpu::TextureView,

    p_background: wgpu::RenderPipeline,
    p_ground: wgpu::RenderPipeline,
    p_mesh: wgpu::RenderPipeline,
    p_ghost: wgpu::RenderPipeline,
    p_plate: wgpu::RenderPipeline,
    p_lines: wgpu::RenderPipeline,
    p_shadow: wgpu::RenderPipeline,
    p_blur_h: wgpu::RenderPipeline,
    p_blur_v: wgpu::RenderPipeline,

    pub mesh: DynBuffer,
    pub ghost: DynBuffer,
    pub plate: DynBuffer,
    pub lines: DynBuffer,
}

const MESH_ATTRS: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32];
const LINE_ATTRS: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];

fn mesh_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<GpuVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &MESH_ATTRS,
    }
}

fn line_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<LineVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &LINE_ATTRS,
    }
}

struct PipeDesc<'a> {
    label: &'a str,
    module: &'a wgpu::ShaderModule,
    layout: &'a wgpu::PipelineLayout,
    vs: &'a str,
    fs: &'a str,
    buffers: &'a [Option<wgpu::VertexBufferLayout<'a>>],
    format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
    /// (depth write, compare); None = no depth attachment.
    depth: Option<(bool, wgpu::CompareFunction)>,
    samples: u32,
}

fn pipeline(device: &wgpu::Device, d: PipeDesc<'_>) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(d.label),
        layout: Some(d.layout),
        vertex: wgpu::VertexState {
            module: d.module,
            entry_point: Some(d.vs),
            compilation_options: Default::default(),
            buffers: d.buffers,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: d.depth.map(|(write, cmp)| wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(write),
            depth_compare: Some(cmp),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: d.samples,
            ..Default::default()
        },
        fragment: Some(wgpu::FragmentState {
            module: d.module,
            entry_point: Some(d.fs),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: d.format,
                blend: d.blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn shadow_texture(device: &wgpu::Device, label: &str) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: SHADOW_RES,
                height: SHADOW_RES,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SHADOW_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

impl Gpu {
    pub fn new(device: &wgpu::Device) -> Self {
        let scene = device.create_shader_module(wgpu::include_wgsl!("scene.wgsl"));
        let blur = device.create_shader_module(wgpu::include_wgsl!("blur.wgsl"));

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("viewport shadow sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let tex_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let smp_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let main_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewport main bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                tex_entry(1),
                smp_entry(2),
            ],
        });
        let blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewport blur bgl"),
            entries: &[tex_entry(0), smp_entry(1)],
        });

        let shadow_a = shadow_texture(device, "viewport shadow coverage");
        let shadow_b = shadow_texture(device, "viewport shadow blur h");
        let shadow_c = shadow_texture(device, "viewport shadow blur v");

        let main_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport main bg"),
            layout: &main_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadow_c),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let blur_bg = |view: &wgpu::TextureView, label| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &blur_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            })
        };
        let blur_h_bind_group = blur_bg(&shadow_a, "viewport blur h bg");
        let blur_v_bind_group = blur_bg(&shadow_b, "viewport blur v bg");

        let main_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewport main layout"),
            bind_group_layouts: &[Some(&main_bgl)],
            immediate_size: 0,
        });
        let blur_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewport blur layout"),
            bind_group_layouts: &[Some(&blur_bgl)],
            immediate_size: 0,
        });

        use wgpu::CompareFunction as Cmp;
        let alpha = Some(wgpu::BlendState::ALPHA_BLENDING);
        let mesh_buffers = [Some(mesh_layout())];
        let line_buffers = [Some(line_layout())];
        let main = |label, vs, fs, buffers, blend, depth| {
            pipeline(
                device,
                PipeDesc {
                    label,
                    module: &scene,
                    layout: &main_layout,
                    vs,
                    fs,
                    buffers,
                    format: COLOR_FORMAT,
                    blend,
                    depth: Some(depth),
                    samples: MSAA,
                },
            )
        };
        let p_background = main(
            "viewport background",
            "vs_background",
            "fs_background",
            &[],
            None,
            (false, Cmp::Always),
        );
        let p_ground = main(
            "viewport ground",
            "vs_ground",
            "fs_ground",
            &[],
            alpha,
            (true, Cmp::Less),
        );
        let p_mesh = main(
            "viewport mesh",
            "vs_mesh",
            "fs_mesh",
            &mesh_buffers,
            None,
            (true, Cmp::Less),
        );
        let p_ghost = main(
            "viewport ghost",
            "vs_mesh",
            "fs_ghost",
            &mesh_buffers,
            alpha,
            (false, Cmp::Less),
        );
        let p_plate = main(
            "viewport plate",
            "vs_mesh",
            "fs_plate",
            &mesh_buffers,
            alpha,
            (false, Cmp::Less),
        );
        let p_lines = main(
            "viewport lines",
            "vs_line",
            "fs_line",
            &line_buffers,
            alpha,
            (false, Cmp::LessEqual),
        );
        let max_blend = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Max,
        };
        let p_shadow = pipeline(
            device,
            PipeDesc {
                label: "viewport shadow coverage",
                module: &scene,
                layout: &main_layout,
                vs: "vs_shadow",
                fs: "fs_shadow",
                buffers: &mesh_buffers,
                format: SHADOW_FORMAT,
                blend: Some(wgpu::BlendState {
                    color: max_blend,
                    alpha: max_blend,
                }),
                depth: None,
                samples: 1,
            },
        );
        let blur_pipe = |label, fs| {
            pipeline(
                device,
                PipeDesc {
                    label,
                    module: &blur,
                    layout: &blur_layout,
                    vs: "vs_full",
                    fs,
                    buffers: &[],
                    format: SHADOW_FORMAT,
                    blend: None,
                    depth: None,
                    samples: 1,
                },
            )
        };
        let p_blur_h = blur_pipe("viewport blur h", "fs_blur_h");
        let p_blur_v = blur_pipe("viewport blur v", "fs_blur_v");

        Self {
            uniform_buf,
            main_bind_group,
            blur_h_bind_group,
            blur_v_bind_group,
            shadow_a,
            shadow_b,
            shadow_c,
            p_background,
            p_ground,
            p_mesh,
            p_ghost,
            p_plate,
            p_lines,
            p_shadow,
            p_blur_h,
            p_blur_v,
            mesh: DynBuffer::new("viewport mesh vertices"),
            ghost: DynBuffer::new("viewport ghost vertices"),
            plate: DynBuffer::new("viewport plate vertices"),
            lines: DynBuffer::new("viewport line vertices"),
        }
    }

    /// Record the shadow coverage + blur passes (call when the mesh or ground changed).
    fn encode_shadow(&self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewport shadow coverage"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.shadow_a,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            if let Some(slice) = self.mesh.slice() {
                pass.set_pipeline(&self.p_shadow);
                pass.set_bind_group(0, &self.main_bind_group, &[]);
                pass.set_vertex_buffer(0, slice);
                pass.draw(0..self.mesh.count, 0..1);
            }
        }
        for (pipe, bg, dst, label) in [
            (&self.p_blur_h, &self.blur_h_bind_group, &self.shadow_b, "viewport blur h"),
            (&self.p_blur_v, &self.blur_v_bind_group, &self.shadow_c, "viewport blur v"),
        ] {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: dst,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Record a full frame into `target`, rebuilding the ground shadow map first if requested.
    pub fn encode_frame(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &Target,
        uniforms: &Uniforms,
        rebuild_shadow: bool,
        draw_ground: bool,
    ) {
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(uniforms));
        if rebuild_shadow {
            self.encode_shadow(encoder);
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("viewport main"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.msaa_view,
                depth_slice: None,
                resolve_target: Some(&target.resolve_srgb_view),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Discard,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &target.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_bind_group(0, &self.main_bind_group, &[]);

        pass.set_pipeline(&self.p_background);
        pass.draw(0..3, 0..1);

        if let Some(slice) = self.mesh.slice() {
            pass.set_pipeline(&self.p_mesh);
            pass.set_vertex_buffer(0, slice);
            pass.draw(0..self.mesh.count, 0..1);
        }
        // ground after the opaque mesh: it is alpha-faded at its rim
        if draw_ground {
            pass.set_pipeline(&self.p_ground);
            pass.draw(0..6, 0..1);
        }
        if let Some(slice) = self.lines.slice() {
            pass.set_pipeline(&self.p_lines);
            pass.set_vertex_buffer(0, slice);
            pass.draw(0..self.lines.count, 0..1);
        }
        if let Some(slice) = self.plate.slice() {
            pass.set_pipeline(&self.p_plate);
            pass.set_vertex_buffer(0, slice);
            pass.draw(0..self.plate.count, 0..1);
        }
        if let Some(slice) = self.ghost.slice() {
            pass.set_pipeline(&self.p_ghost);
            pass.set_vertex_buffer(0, slice);
            pass.draw(0..self.ghost.count, 0..1);
        }
    }
}
