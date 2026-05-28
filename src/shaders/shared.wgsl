#define_import_path boimp::shared

#import bevy_pbr::{
    pbr_types::{PbrInput, STANDARD_MATERIAL_FLAGS_UNLIT_BIT, pbr_input_new},
    view_transformations::{position_ndc_to_world, frag_coord_to_ndc, position_world_to_clip},
    pbr_functions::calculate_view,
    mesh_view_bindings::view,
};

const IMPOSTER_MATERIAL_UNLIT: u32 = 1;
const IMPOSTER_MATERIAL_EMISSIVE: u32 = 2;

struct ImposterData {
    center_and_scale: vec4<f32>,
    packed_offset: vec2<u32>,
    packed_size: vec2<u32>,
    grid_size: u32,
    base_tile_size: u32,
    flags: u32,
    alpha: f32,
    multisample_amount: f32,
}

struct ImposterVertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) base_world_position: vec3<f32>,
    @location(2) inverse_rotation_0c: vec3<f32>,
    @location(3) inverse_rotation_1c: vec3<f32>,
    @location(4) inverse_rotation_2c: vec3<f32>,
}

struct UnpackedMaterialProps {
    rgba: vec4<f32>,
    normal: vec3<f32>,
    roughness: f32,
    metallic: f32,
    flags: u32,
    depth: f32, // [-1..1]
}

fn spherical_uv_from_normal(dir: vec3<f32>) -> vec2<f32> {
    let octant: vec3<f32> = sign(dir);
    let sum: f32 = dot(dir, octant);
    let octahedron: vec3<f32> = dir / sum;
    let absolute: vec3<f32> = abs(octahedron);
    return (select(octahedron.xz, octant.xz * vec2(1.0 - absolute.z, 1.0 - absolute.x), octahedron.y < 0.0) + 1.0) * 0.5;
}

fn spherical_normal_from_uv(uv: vec2<f32>) -> vec3<f32> {
    let x = uv.x * 2.0 - 1.0;
    let z = uv.y * 2.0 - 1.0;
    let y = 1.0 - abs(x) - abs(z);

    let n = select(
        vec3(x, y, z),
        vec3(sign(x) * (1.0 - abs(z)), y, sign(z) * (1.0 - abs(x))),
        y < 0.0
    );
    return normalize(n);
}

fn normalize_or_zero(in: vec3<f32>) -> vec3<f32> {
    let len = length(in);
    return select(in / len, vec3(0.0), len < 0.00001);
}

// rg32uint
// r: [0-4] r, [5-9] g, [10-14] b, [15-19] a, [20-23] roughness, [24-27] metallic, [28-31] flags
// g: [0-23] normal, [24-31] depth (linear, 0 => -1r, 128 => 0, 255 => +1r)

// pack
fn pack_bits(input: f32, offset: u32, count: u32) -> u32 {
    let mask = (1u << count) - 1u;
    return u32(saturate(input) * f32(mask) + 0.5) << offset;
}

fn pack_normal_and_depth(normal: vec3<f32>, depth: f32) -> u32 {
    let octahedral_normal = spherical_uv_from_normal(normal);
    // Pre-quantise normals to 4 effective bits per axis (16 levels), even
    // though we store them in 12-bit slots for layout compatibility. With
    // 16 levels per axis the median-cut's 1-bucket spread on normals
    // (~1/15 in normalised space) is coarser than RGB's, matching RGB and
    // depth's per-bucket cost in the cluster decisions. Stops the high-
    // frequency normal noise that previously amplified into "firefly"
    // specular speckle on shiny mip bakes.
    let nx_q = floor(octahedral_normal.x * 15.0 + 0.5) / 15.0;
    let ny_q = floor(octahedral_normal.y * 15.0 + 0.5) / 15.0;
    // Depth pre-quant to 4 bits (16 levels). For the idx10s format the
    // per-tile depth palette holds the *averaged* depth per (tile, slot),
    // so per-pixel parallax precision is ultimately bounded by what the
    // palette stores (8 bits) and how many distinct slots a (mat, norm)
    // earns. Coarser pre-quant collapses near-equal depth pixels into the
    // same bitmap bucket, reducing per-group slot cost and freeing palette
    // budget for more distinct (mat, norm). 16 levels = ~6% of imposter
    // radius per step — visibly coarser than 6-bit but the depth-palette
    // refinement closes most of the gap.
    let depth_q = floor(depth * 15.0 + 0.5) / 15.0;
    return
        pack_bits(nx_q, 0u, 12u) +
        pack_bits(ny_q, 12u, 12u) +
        pack_bits(depth_q, 24u, 8u);
}

fn pack_rgba_roughness_metallic_flags(albedo: vec4<f32>, roughness: f32, metallic: f32, flags: u32) -> u32 {
    // Pre-quantise all material channels at bake time to collapse "similar
    // but not identical" pixels onto the *same* pack. The median-cut then
    // sees them as one weighted point instead of many near-neighbours,
    // which fights the colour-banding-at-cut-boundary problem at its
    // source: rather than the cut slicing through a smooth gradient (and
    // putting adjacent pixels in opposite buckets), the gradient is
    // pre-discretised into a small number of steps and adjacent pixels
    // straddling a step share a pack — and thus a palette entry.
    //
    // Effective bit budget after pre-quant:
    //   RGB        4 bits each (16 levels) — primary visual channel
    //   Alpha      3 bits      (8 levels)  — silhouette steps are visible
    //   Roughness  2 bits      (4 levels)  — barely-visible at distance
    //   Metallic   2 bits      (4 levels)  — mostly 0 or 1 in practice
    //   Flags      stays 4 bits — discrete enum, not averaged
    //   (normal 5 bits each, depth 6 bits are pre-quantised in
    //    pack_normal_and_depth below.)
    let r_q = floor(albedo.r * 15.0 + 0.5) / 15.0;
    let g_q = floor(albedo.g * 15.0 + 0.5) / 15.0;
    let b_q = floor(albedo.b * 15.0 + 0.5) / 15.0;
    let a_q = floor(albedo.a * 7.0 + 0.5) / 7.0;
    let rough_q = floor(roughness * 3.0 + 0.5) / 3.0;
    let metal_q = floor(metallic * 3.0 + 0.5) / 3.0;
    return
        pack_bits(r_q, 0u, 5u) +
        pack_bits(g_q, 5u, 5u) +
        pack_bits(b_q, 10u, 5u) +
        pack_bits(a_q, 15u, 5u) +
        pack_bits(rough_q, 20u, 4u) +
        pack_bits(metal_q, 24u, 4u) +
        (flags << 28u);
}

fn pack_props(input: UnpackedMaterialProps) -> vec2<u32> {
    return vec2<u32>(
        pack_rgba_roughness_metallic_flags(input.rgba, input.roughness, input.metallic, input.flags),
        pack_normal_and_depth(input.normal, input.depth * 0.5 + 0.5)
    );
}

fn pack_pbrinput(input: PbrInput) -> vec2<u32> {
    let use_emissive = length(input.material.base_color.rgb) < length(input.material.emissive.rgb);
    let color = select(input.material.base_color, input.material.emissive, use_emissive);
    let flags = 
        u32((input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) != 0u) * IMPOSTER_MATERIAL_UNLIT + 
        u32(use_emissive) * IMPOSTER_MATERIAL_EMISSIVE;
    return vec2<u32>(
        pack_rgba_roughness_metallic_flags(vec4(color.rgb, input.material.base_color.a), input.material.perceptual_roughness, input.material.metallic, flags),
        pack_normal_and_depth(input.world_normal, input.frag_coord.z)
    );
}

// unpack
fn unpack_bits(input: u32, offset: u32, count: u32) -> f32 {
    let mask = (1u << count) - 1u;
    return f32(((input >> offset) & mask)) / f32(mask);
}

fn unpack_normal(input: u32) -> vec3<f32> {
    return spherical_normal_from_uv(vec2<f32>(
        unpack_bits(input, 0u, 12u),
        unpack_bits(input, 12u, 12u),
    ));
}

fn unpack_depth(input: u32) -> f32 {
    return unpack_bits(input, 24u, 8u) * 2.0 - 1.0;
}

fn unpack_flags(input: u32) -> u32 {
    return (input >> 28u);
}

fn unpack_rgba(input: u32) -> vec4<f32> {
    return vec4<f32>(
        unpack_bits(input, 0u, 5u),
        unpack_bits(input, 5u, 5u),
        unpack_bits(input, 10u, 5u),
        unpack_bits(input, 15u, 5u),
    );
}

// roughness and metallic both read back as the raw 4-bit value with no
// remapping. bevy_pbr clamps `perceptual_roughness` itself before evaluating
// the GGX microfacet BRDF, so we don't need an extra floor here; the
// previous `[0.1, 0.9]` double clamp compounded across mips (each higher
// mip rebakes from the lower mip, applies the clamp on input, and stores
// it back) and made fully-metallic surfaces never look fully metallic.
fn unpack_roughness(input: u32) -> f32 {
    return unpack_bits(input, 20u, 4u);
}

fn unpack_metallic(input: u32) -> f32 {
    return unpack_bits(input, 24u, 4u);
}

fn unpack_props(packed: vec2<u32>) -> UnpackedMaterialProps {
    var props: UnpackedMaterialProps;
    props.rgba = unpack_rgba(packed.r);
    props.roughness = unpack_roughness(packed.r);
    props.metallic = unpack_metallic(packed.r);
    props.flags = unpack_flags(packed.r);
    props.normal = unpack_normal(packed.g);
    props.depth = unpack_depth(packed.g);
    return props;
}

/// Unpack the tight 32-bit idx10s palette entry.
///   bits 0-3:   r (4 bits)
///   bits 4-7:   g
///   bits 8-11:  b
///   bits 12-14: alpha (3 bits)
///   bits 15-16: roughness (2 bits)
///   bits 17-18: metallic (2 bits)
///   bits 19-22: flags (4 bits)
///   bits 23-26: normal_x (4 bits)
///   bits 27-30: normal_y (4 bits)
///   bit  31:    unused
/// Depth is *not* in the palette for idx10s — the caller overrides
/// `props.depth` after this unpack using a per-tile depth-palette lookup.
fn unpack_props_10s(pack: u32) -> UnpackedMaterialProps {
    var props: UnpackedMaterialProps;
    let r = f32(pack & 0xFu) / 15.0;
    let g = f32((pack >> 4u) & 0xFu) / 15.0;
    let b = f32((pack >> 8u) & 0xFu) / 15.0;
    let a = f32((pack >> 12u) & 0x7u) / 7.0;
    let rough = f32((pack >> 15u) & 0x3u) / 3.0;
    let metal = f32((pack >> 17u) & 0x3u) / 3.0;
    let flags = (pack >> 19u) & 0xFu;
    let nx = f32((pack >> 23u) & 0xFu) / 15.0;
    let ny = f32((pack >> 27u) & 0xFu) / 15.0;
    props.rgba = vec4<f32>(r, g, b, a);
    props.roughness = rough;
    props.metallic = metal;
    props.flags = flags;
    props.normal = spherical_normal_from_uv(vec2<f32>(nx, ny));
    props.depth = 0.0;
    return props;
}

fn weighted_props(a: UnpackedMaterialProps, b: UnpackedMaterialProps, weight_a: f32) -> UnpackedMaterialProps {
    var out: UnpackedMaterialProps;

    let raw_wa = a.rgba.a * weight_a;
    let raw_wb = b.rgba.a * (1.0 - weight_a);
    let total_weight = raw_wa + raw_wb;
    if total_weight == 0.0 {
        return out;
    }

    let wa = raw_wa / total_weight;
    let wb = raw_wb / total_weight;

    // Alpha uses a sub-linear power mean (p=0.5) instead of a linear mix.
    // Idempotent for equal inputs — (x,x)->x — so uniformly-translucent
    // surfaces still blend to their own value, but mixed cases like (1,0)
    // collapse to 0.25 instead of 0.5. This compresses fine-grid patterns
    // (an isolated opaque sub-pixel surrounded by empties) toward the
    // "see-through" perception the eye expects, rather than the
    // mathematically-correct-but-too-opaque linear average.
    let alpha = pow(sqrt(a.rgba.a) * weight_a + sqrt(b.rgba.a) * (1.0 - weight_a), 2.0);

    out.rgba = vec4(a.rgba.rgb * wa + b.rgba.rgb * wb, alpha);
    out.roughness = a.roughness * wa + b.roughness * wb;
    out.metallic = a.metallic * wa + b.metallic * wb;
    out.normal = normalize_or_zero(a.normal * wa + b.normal * wb);
    out.depth = a.depth * wa + b.depth * wb;
    out.flags = select(a.flags, b.flags, wa < wb);
    return out;
}

// Convert a sample's parallax depth (where the surface sits inside the
// *source* imposter's volume) into the current bake camera's NDC z (the
// raw `frag_coord.z`-style [0, 1] value that `pack_pbrinput` stores).
// Mirrors the inverse operation used in fragment.wgsl when rendering an
// imposter — the level-N bake camera projects the recovered world-space
// surface point back into its own clip space so the stored depth has
// the right meaning at level N+1.
fn parallax_depth_to_bake_ndc(
    quad_world_position: vec3<f32>,
    back: vec3<f32>,
    parallax_depth: f32,
    parallax_scale: f32,
) -> f32 {
    let surface_world = quad_world_position + back * parallax_depth * parallax_scale;
    let clip = position_world_to_clip(surface_world);
    return clip.z / clip.w;
}

// Manual depth check. Storage-buffer writes from a fragment shader bypass
// the hardware depth/stencil test (per WebGPU spec), so multiple fragments
// at the same pixel would otherwise all write into bake_buffer and the
// last-rendered wins instead of the closest. Skip the new fragment if
// there is already an opaque pixel in front of us; for translucent or
// empty existing we accept order-dependence and composite anyway.
// Bevy reverse-z: higher depth = closer.
fn passes_depth_check(new_depth: f32, existing: UnpackedMaterialProps) -> bool {
    return new_depth >= existing.depth || existing.rgba.a < 1.0;
}

// Porter-duff "over": composite `incoming` on top of `existing`. Hardware
// depth-test arbitrates which fragments reach the shader (opaque: closest
// wins; transparent: all passing fragments execute), and we rely on the
// rendering pipeline's order for transparent overlap. Existing samples
// with `a == 0` are the storage-buffer sentinel for "nothing here yet".
//
// `incoming`'s normal / depth / roughness / metallic / flags become the
// result's — for opaque this is the closest surface (correct), for
// transparent over opaque this is the transparent's parallax/orientation
// (a known limitation of single-surface imposters but no worse than before).
fn compose_over(existing: UnpackedMaterialProps, incoming: UnpackedMaterialProps) -> UnpackedMaterialProps {
    if existing.rgba.a <= 0.0 {
        return incoming;
    }
    let a_out = incoming.rgba.a + existing.rgba.a * (1.0 - incoming.rgba.a);
    if a_out <= 0.0 {
        return incoming;
    }
    let rgb_out = (incoming.rgba.rgb * incoming.rgba.a + existing.rgba.rgb * existing.rgba.a * (1.0 - incoming.rgba.a)) / a_out;
    var out = incoming;
    out.rgba = vec4<f32>(rgb_out, a_out);
    return out;
}

fn unpack_pbrinput(props: UnpackedMaterialProps, frag_coord: vec4<f32>) -> PbrInput {
    var input = pbr_input_new();


    if (props.flags & IMPOSTER_MATERIAL_UNLIT) != 0u {
        input.material.flags |= STANDARD_MATERIAL_FLAGS_UNLIT_BIT;
    }
    let use_emissive = (props.flags & IMPOSTER_MATERIAL_EMISSIVE) != 0u;

    input.material.base_color = props.rgba;
    if use_emissive {
        input.material.emissive = props.rgba;
    }
    input.material.perceptual_roughness = props.roughness;
    input.material.metallic = props.metallic;

    input.N = props.normal;
    input.world_normal = input.N;
    input.frag_coord = frag_coord;
    input.world_position = vec4(position_ndc_to_world(frag_coord_to_ndc(frag_coord)), 1.0);
    input.is_orthographic = view.clip_from_view[3].w == 1.0;
    input.V = calculate_view(input.world_position, input.is_orthographic);

    return input;
}
