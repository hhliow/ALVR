//! Hand passthrough mask.
//!
//! The tracked hands are approximated with capsules (one per bone, plus a few to fill the palm)
//! and rendered into a per-view mask texture, using the same view and projection as the stream
//! layer. The stream shader then cuts the hands out of the streamed image (alpha = 0), so that
//! the passthrough layer behind it shows the real hands. Since the hand pose is sampled locally at
//! display time, the cut-out does not suffer from the streaming latency.
//!
//! Mask texture channels:
//! - R: hand coverage, 0..1 with soft edges.
//! - G: nearness of the nearest hand surface, `1 - depth / HAND_DEPTH_MAX` (view axis depth).
//!
//! When the scene depth of the streamed frame is available, `hand_visibility()` in stream.wgsl
//! compares it with G to hide the hands behind virtual objects (see scene_depth.rs). Otherwise the
//! hands are always visible.

use crate::{GraphicsContext, MAX_PUSH_CONSTANTS_SIZE, scene_depth::SceneDepth};
use alvr_common::{
    ViewParams,
    glam::{Mat4, UVec2, Vec3},
};
use alvr_session::HandPassthroughConfig;
use std::mem;
use wgpu::{
    BlendComponent, BlendFactor, BlendOperation, BlendState, Buffer, BufferAddress,
    BufferDescriptor, BufferUsages, Color, ColorTargetState, ColorWrites, CommandEncoder, Extent3d,
    FragmentState, LoadOp, Operations, PipelineCompilationOptions, PipelineLayoutDescriptor,
    PrimitiveState, PrimitiveTopology, PushConstantRange, RenderPassColorAttachment,
    RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, ShaderStages, StoreOp,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    VertexAttribute, VertexBufferLayout, VertexFormat, VertexState, VertexStepMode, include_wgsl,
};

pub const HAND_JOINT_COUNT: usize = 26;

/// Hand depth corresponding to nearness 0. 8 bit nearness gives a 5 mm resolution.
pub(crate) const HAND_DEPTH_MAX: f32 = 1.275;
const PROJECTION_NEAR: f32 = 0.01;

// The mask is soft-edged, a lower resolution is enough.
const MASK_RESOLUTION_DIVISOR: u32 = 2;
const MASK_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;

const PUSH_CONSTANTS_SIZE: u32 = 80; // mat4x4f + f32, padded to 16 bytes
const _: () = assert!(PUSH_CONSTANTS_SIZE <= MAX_PUSH_CONSTANTS_SIZE);

// Joint indices follow XR_EXT_hand_tracking
const WRIST: usize = 1;
const THUMB_METACARPAL: usize = 2;
const THUMB_PROXIMAL: usize = 3;
const THUMB_DISTAL: usize = 4;
const THUMB_TIP: usize = 5;
const INDEX_METACARPAL: usize = 6;
const MIDDLE_METACARPAL: usize = 11;
const RING_METACARPAL: usize = 16;
const LITTLE_METACARPAL: usize = 21;
// Offsets from each finger's metacarpal joint
const PROXIMAL: usize = 1;
const INTERMEDIATE: usize = 2;
const DISTAL: usize = 3;
const TIP: usize = 4;

const FINGERS: [usize; 4] = [
    INDEX_METACARPAL,
    MIDDLE_METACARPAL,
    RING_METACARPAL,
    LITTLE_METACARPAL,
];

const HAND_CAPSULES: [(usize, usize); 29] = {
    let mut capsules = [(0, 0); 29];

    // Thumb
    capsules[0] = (WRIST, THUMB_METACARPAL);
    capsules[1] = (THUMB_METACARPAL, THUMB_PROXIMAL);
    capsules[2] = (THUMB_PROXIMAL, THUMB_DISTAL);
    capsules[3] = (THUMB_DISTAL, THUMB_TIP);

    // Fingers, and palm along the metacarpals
    let mut i = 0;
    while i < 4 {
        let m = FINGERS[i];
        capsules[4 + i * 4] = (m, m + PROXIMAL);
        capsules[5 + i * 4] = (m + PROXIMAL, m + INTERMEDIATE);
        capsules[6 + i * 4] = (m + INTERMEDIATE, m + DISTAL);
        capsules[7 + i * 4] = (m + DISTAL, m + TIP);
        i += 1;
    }

    // Palm, across the knuckles and across the base
    let mut i = 0;
    while i < 3 {
        capsules[20 + i] = (FINGERS[i] + PROXIMAL, FINGERS[i + 1] + PROXIMAL);
        capsules[23 + i] = (FINGERS[i], FINGERS[i + 1]);
        i += 1;
    }

    // Wrist and thenar area
    capsules[26] = (WRIST, INDEX_METACARPAL);
    capsules[27] = (WRIST, LITTLE_METACARPAL);
    capsules[28] = (THUMB_METACARPAL, INDEX_METACARPAL + PROXIMAL);

    capsules
};

const MAX_HANDS: usize = 2;
const MAX_CAPSULES_PER_VIEW: usize = HAND_CAPSULES.len() * MAX_HANDS;

#[derive(Clone, Copy, Debug)]
pub struct HandJoint {
    /// Position in the same space as the stream view poses
    pub position: Vec3,
    pub radius: f32,
}

pub struct HandPassthroughFrame<'a> {
    pub config: &'a HandPassthroughConfig,
    pub hands: &'a [[HandJoint; HAND_JOINT_COUNT]],
    /// Used for occlusion, if enabled in the config
    pub scene_depth: Option<SceneDepth<'a>>,
}

// Matches VertexInput in hand_mask.wgsl
#[repr(C)]
#[derive(Clone, Copy)]
struct CapsuleInstance {
    start: [f32; 3],
    radius: f32,
    end: [f32; 3],
    _pad: f32,
}

const INSTANCE_SIZE: BufferAddress = mem::size_of::<CapsuleInstance>() as BufferAddress;
const VIEW_INSTANCES_SIZE: BufferAddress = INSTANCE_SIZE * MAX_CAPSULES_PER_VIEW as BufferAddress;

pub(crate) struct HandMaskRenderer {
    pipeline: RenderPipeline,
    instance_buffer: Buffer,
    mask_views: [TextureView; 2],
}

impl HandMaskRenderer {
    pub fn new(context: &GraphicsContext, target_view_resolution: UVec2) -> Self {
        let device = &context.device;

        let shader_module =
            device.create_shader_module(include_wgsl!("../resources/hand_mask.wgsl"));
        let constants = [("HAND_DEPTH_MAX", HAND_DEPTH_MAX as f64)];

        // Coverage and nearness are both merged with max: the nearest hand surface wins.
        let max_blend = BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::One,
            operation: BlendOperation::Max,
        };

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("hand mask"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[],
                push_constant_ranges: &[PushConstantRange {
                    stages: ShaderStages::VERTEX_FRAGMENT,
                    range: 0..PUSH_CONSTANTS_SIZE,
                }],
            })),
            vertex: VertexState {
                module: &shader_module,
                entry_point: None,
                compilation_options: PipelineCompilationOptions {
                    constants: &constants,
                    zero_initialize_workgroup_memory: false,
                },
                buffers: &[VertexBufferLayout {
                    array_stride: INSTANCE_SIZE,
                    step_mode: VertexStepMode::Instance,
                    attributes: &[
                        VertexAttribute {
                            format: VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 0,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32x4,
                            offset: 16,
                            shader_location: 1,
                        },
                    ],
                }],
            },
            // No culling: the eye can be inside a capsule bounding box
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(FragmentState {
                module: &shader_module,
                entry_point: None,
                compilation_options: PipelineCompilationOptions {
                    constants: &constants,
                    zero_initialize_workgroup_memory: false,
                },
                targets: &[Some(ColorTargetState {
                    format: MASK_FORMAT,
                    blend: Some(BlendState {
                        color: max_blend,
                        alpha: max_blend,
                    }),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        let instance_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("hand mask capsules"),
            size: VIEW_INSTANCES_SIZE * 2,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mask_resolution = (target_view_resolution / MASK_RESOLUTION_DIVISOR).max(UVec2::ONE);
        let mask_views = [(); 2].map(|_| {
            device
                .create_texture(&TextureDescriptor {
                    label: Some("hand mask"),
                    size: Extent3d {
                        width: mask_resolution.x,
                        height: mask_resolution.y,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: MASK_FORMAT,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        });

        Self {
            pipeline,
            instance_buffer,
            mask_views,
        }
    }

    pub fn mask_view(&self, view_idx: usize) -> &TextureView {
        &self.mask_views[view_idx]
    }

    /// Render the mask of each view. Returns false if there was nothing to draw, in which case the
    /// mask textures are left untouched and must not be used.
    pub fn render(
        &self,
        context: &GraphicsContext,
        encoder: &mut CommandEncoder,
        frame: &HandPassthroughFrame,
        view_params: [&ViewParams; 2],
    ) -> bool {
        if frame.hands.is_empty() {
            return false;
        }

        for (view_idx, view_params) in view_params.into_iter().enumerate() {
            let world_to_view = (Mat4::from_translation(view_params.pose.position)
                * Mat4::from_quat(view_params.pose.orientation))
            .inverse();

            let instances = frame
                .hands
                .iter()
                .take(MAX_HANDS)
                .flat_map(|joints| {
                    HAND_CAPSULES.iter().map(move |&(a, b)| {
                        let start = world_to_view.transform_point3(joints[a].position);
                        let end = world_to_view.transform_point3(joints[b].position);
                        let radius =
                            (joints[a].radius + joints[b].radius) * 0.5 + frame.config.dilation;

                        CapsuleInstance {
                            start: start.to_array(),
                            radius,
                            end: end.to_array(),
                            _pad: 0.0,
                        }
                    })
                })
                .collect::<Vec<_>>();

            let offset = VIEW_INSTANCES_SIZE * view_idx as BufferAddress;
            context
                .queue
                .write_buffer(&self.instance_buffer, offset, &instance_bytes(&instances));

            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("hand mask"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &self.mask_views[view_idx],
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::TRANSPARENT),
                        store: StoreOp::Store,
                    },
                })],
                ..Default::default()
            });

            let projection = super::projection_from_fov_with_near(view_params.fov, PROJECTION_NEAR);
            let mut push_constants = [0_u8; PUSH_CONSTANTS_SIZE as usize];
            for (chunk, value) in push_constants.chunks_exact_mut(4).zip(
                projection
                    .to_cols_array()
                    .into_iter()
                    .chain([frame.config.edge_feathering]),
            ) {
                chunk.copy_from_slice(&value.to_le_bytes());
            }

            pass.set_pipeline(&self.pipeline);
            pass.set_push_constants(ShaderStages::VERTEX_FRAGMENT, 0, &push_constants);
            pass.set_vertex_buffer(
                0,
                self.instance_buffer
                    .slice(offset..offset + INSTANCE_SIZE * instances.len() as BufferAddress),
            );
            pass.draw(0..14, 0..instances.len() as u32);
        }

        true
    }
}

fn instance_bytes(instances: &[CapsuleInstance]) -> Vec<u8> {
    instances
        .iter()
        .flat_map(|i| {
            i.start
                .into_iter()
                .chain([i.radius])
                .chain(i.end)
                .chain([i._pad])
        })
        .flat_map(f32::to_le_bytes)
        .collect()
}
