//! Scene depth for hand occlusion.
//!
//! The streamer sends a low resolution depth image of each streamed frame, quantized to 8 bit
//! nearness (see `SceneDepthHeader` in alvr_packets). It is bound to the stream shader, which hides
//! the passthrough hands where the scene is nearer than them.
//!
//! The depth image usually arrives a few frames after its video frame, so the newest one available
//! belongs to an older head pose. The stream shader rotates the lookup from the view of the frame
//! being displayed into the view of the depth image, which keeps the occlusion edges in place when
//! turning the head. Head translation between the two is neglected.

use crate::GraphicsContext;
use alvr_common::{
    Fov, ViewParams,
    glam::{Mat3, UVec2, Vec4},
};
use std::{mem, time::Duration};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType,
    BufferDescriptor, BufferUsages, Extent3d, Origin3d, ShaderStages, TexelCopyBufferLayout,
    TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor, TextureDimension,
    TextureFormat, TextureSampleType, TextureUsages, TextureViewDimension,
};

pub struct SceneDepth<'a> {
    /// Timestamp of the video frame this depth belongs to
    pub timestamp: Duration,
    pub eye_resolution: UVec2,
    /// Distance corresponding to nearness 0
    pub max_distance: f32,
    /// Left and right eye side by side, top row first
    pub nearness: &'a [u8],
    /// View poses and FoVs the depth was rendered with
    pub view_params: [ViewParams; 2],
}

// Matches SceneDepthUniforms in stream.wgsl
const VIEW_UNIFORM_VEC4_COUNT: usize = 5;
const UNIFORM_VEC4_COUNT: usize = VIEW_UNIFORM_VEC4_COUNT * 2 + 1;
const UNIFORM_SIZE: usize = UNIFORM_VEC4_COUNT * mem::size_of::<Vec4>();

pub(crate) struct SceneDepthResources {
    bind_group_layout: BindGroupLayout,
    uniform_buffer: Buffer,
    texture: Texture,
    bind_group: BindGroup,
    uploaded_timestamp: Option<Duration>,
}

impl SceneDepthResources {
    pub fn new(context: &GraphicsContext) -> Self {
        let device = &context.device;

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("scene depth"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("scene depth uniforms"),
            size: UNIFORM_SIZE as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Placeholder until the first depth image arrives
        let texture = create_texture(context, UVec2::ONE);
        let bind_group = create_bind_group(context, &bind_group_layout, &texture, &uniform_buffer);

        Self {
            bind_group_layout,
            uniform_buffer,
            texture,
            bind_group,
            uploaded_timestamp: None,
        }
    }

    pub fn bind_group_layout(&self) -> &BindGroupLayout {
        &self.bind_group_layout
    }

    pub fn bind_group(&self) -> &BindGroup {
        &self.bind_group
    }

    /// `current_view_params` are the input view params of the frame being displayed. Returns false
    /// if the depth image is malformed and must not be used.
    pub fn update(
        &mut self,
        context: &GraphicsContext,
        depth: &SceneDepth,
        current_view_params: [&ViewParams; 2],
    ) -> bool {
        let size = UVec2::new(depth.eye_resolution.x * 2, depth.eye_resolution.y);
        if size.x == 0 || size.y == 0 || depth.nearness.len() != (size.x * size.y) as usize {
            return false;
        }

        if self.uploaded_timestamp != Some(depth.timestamp) {
            if self.texture.width() != size.x || self.texture.height() != size.y {
                self.texture = create_texture(context, size);
                self.bind_group = create_bind_group(
                    context,
                    &self.bind_group_layout,
                    &self.texture,
                    &self.uniform_buffer,
                );
            }

            context.queue.write_texture(
                TexelCopyTextureInfo {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                depth.nearness,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size.x),
                    rows_per_image: Some(size.y),
                },
                Extent3d {
                    width: size.x,
                    height: size.y,
                    depth_or_array_layers: 1,
                },
            );
            self.uploaded_timestamp = Some(depth.timestamp);
        }

        // The rotation changes with the displayed frame even when the depth image does not
        let mut uniforms = [Vec4::ZERO; UNIFORM_VEC4_COUNT];
        for (view_idx, chunk) in uniforms
            .chunks_exact_mut(VIEW_UNIFORM_VEC4_COUNT)
            .enumerate()
        {
            let current = current_view_params[view_idx];
            let depth_view = &depth.view_params[view_idx];

            // From the current view space to the depth view space
            let rotation =
                Mat3::from_quat(depth_view.pose.orientation.inverse() * current.pose.orientation);

            chunk[0] = rotation.x_axis.extend(0.0);
            chunk[1] = rotation.y_axis.extend(0.0);
            chunk[2] = rotation.z_axis.extend(0.0);
            chunk[3] = fov_tangents(current.fov);
            chunk[4] = fov_tangents(depth_view.fov);
        }
        uniforms[UNIFORM_VEC4_COUNT - 1] = Vec4::new(depth.max_distance, 0.0, 0.0, 0.0);

        let bytes = uniforms
            .iter()
            .flat_map(|v| v.to_array())
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        context.queue.write_buffer(&self.uniform_buffer, 0, &bytes);

        true
    }
}

// left, right, up, down
fn fov_tangents(fov: Fov) -> Vec4 {
    Vec4::new(
        fov.left.tan(),
        fov.right.tan(),
        fov.up.tan(),
        fov.down.tan(),
    )
}

fn create_texture(context: &GraphicsContext, size: UVec2) -> Texture {
    context.device.create_texture(&TextureDescriptor {
        label: Some("scene depth"),
        size: Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::R8Unorm,
        usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

fn create_bind_group(
    context: &GraphicsContext,
    layout: &BindGroupLayout,
    texture: &Texture,
    uniform_buffer: &Buffer,
) -> BindGroup {
    context.device.create_bind_group(&BindGroupDescriptor {
        label: Some("scene depth"),
        layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&texture.create_view(&Default::default())),
            },
            BindGroupEntry {
                binding: 1,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    })
}
