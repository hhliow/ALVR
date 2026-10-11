// Renders the tracked hands as capsules into the hand mask texture (see hand_mask.rs).
// Output: R = coverage, G = normalized inverse view depth of the nearest hand surface.

override HAND_DEPTH_NEAR: f32;

struct PushConstant {
    projection: mat4x4f,
    edge_feathering: f32,
}
var<push_constant> pc: PushConstant;

struct VertexInput {
    // View space, xyz: start point, w: radius
    @location(0) start_radius: vec4f,
    // View space, xyz: end point
    @location(1) end: vec4f,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) view_position: vec3f,
    @location(1) @interpolate(flat) start: vec3f,
    @location(2) @interpolate(flat) end: vec3f,
    @location(3) @interpolate(flat) radius: f32,
}

// Corners of a cube as a 14 vertex triangle strip
fn cube_corner(vertex_index: u32) -> vec3f {
    let x = (0x287au >> vertex_index) & 1u;
    let y = (0x02afu >> vertex_index) & 1u;
    let z = (0x31e3u >> vertex_index) & 1u;

    return vec3f(f32(x), f32(y), f32(z)) * 2.0 - 1.0;
}

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32, input: VertexInput) -> VertexOutput {
    let start = input.start_radius.xyz;
    let end = input.end.xyz;
    let radius = input.start_radius.w;

    // Box bounding the capsule (including the feathered edge), aligned to the capsule axis
    let axis = end - start;
    let half_length = length(axis) * 0.5;
    var u = vec3f(0.0, 1.0, 0.0);
    if half_length > 1e-5 {
        u = axis / (half_length * 2.0);
    }
    var helper = vec3f(0.0, 1.0, 0.0);
    if abs(u.y) > 0.99 {
        helper = vec3f(1.0, 0.0, 0.0);
    }
    let v = normalize(cross(u, helper));
    let w = cross(u, v);

    let extent = radius + pc.edge_feathering;
    let corner = cube_corner(vertex_index);
    let view_position = (start + end) * 0.5
        + u * (half_length + extent) * corner.x
        + v * extent * corner.y
        + w * extent * corner.z;

    var result: VertexOutput;
    result.position = pc.projection * vec4f(view_position, 1.0);
    result.view_position = view_position;
    result.start = start;
    result.end = end;
    result.radius = radius;

    return result;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4f {
    // View ray from the eye (origin) through this pixel
    let dir = normalize(input.view_position);

    // Closest points between the ray and the capsule segment
    let ba = input.end - input.start;
    let e = dot(ba, ba);
    let f = dot(ba, dir);
    let g = dot(input.start, dir);
    let h = dot(input.start, ba);
    let denom = e - f * f;
    var s = 0.0;
    if denom > 1e-8 {
        s = clamp((g * f - h) / denom, 0.0, 1.0);
    }
    let t = max(g + s * f, 0.0);
    let distance = length(input.start + s * ba - t * dir);

    let feather = max(pc.edge_feathering, 1e-4);
    let coverage = 1.0 - smoothstep(input.radius - feather * 0.5, input.radius + feather * 0.5, distance);
    if coverage <= 0.0 {
        discard;
    }

    // Front surface of the capsule along the ray
    let half_chord = sqrt(max(input.radius * input.radius - distance * distance, 0.0));
    let view_depth = max((t - half_chord) * -dir.z, HAND_DEPTH_NEAR);

    return vec4f(coverage, HAND_DEPTH_NEAR / view_depth, 0.0, 0.0);
}
