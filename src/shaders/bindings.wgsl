#define_import_path boimp::bindings

#import bevy_pbr::{
    view_transformations::position_view_to_world,
}

#import boimp::shared::{
    ImposterData,
    UnpackedMaterialProps,
    spherical_normal_from_uv,
    spherical_uv_from_normal,
    unpack_props,
    unpack_props_10s,
    weighted_props,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(0)
var<uniform> imposter_data: ImposterData;

@group(#{MATERIAL_BIND_GROUP}) @binding(1)
var imposter_pixels: texture_2d<u32>;

#ifdef INDEXED_PIXELS
@group(#{MATERIAL_BIND_GROUP}) @binding(2)
var imposter_indices: texture_2d<u32>;
#endif

#ifdef INDEXED_V2
// V2 reuses binding 2 for the (low-byte / single-byte) index texture, and
// adds binding 3 for the idx12 high-nibble texture. idx10s reuses 3 for the
// quarter-width high-2-bits texture and adds binding 4 for the per-tile
// depth palette. Bindings are always present (dummy fallback for variants
// that don't use them) — the shader only reads the relevant ones under
// matching `INDEXED_V2_12` / `INDEXED_V2_14` / `INDEXED_V2_10S` defs.
@group(#{MATERIAL_BIND_GROUP}) @binding(2)
var imposter_indices: texture_2d<u32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(3)
var imposter_indices_hi: texture_2d<u32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4)
var imposter_depth_palette: texture_2d<u32>;
#endif

struct SamplePositions {
    tile_indices: array<vec2<u32>, 3>,
    tile_weights: vec3<f32>,
}

fn oct_sample_weights(tile_uv: vec2<f32>) -> vec3<f32> {
    let res = vec3<f32>(
        1.0 - max(tile_uv.x, tile_uv.y),
        abs(tile_uv.x - tile_uv.y),
        min(tile_uv.x, tile_uv.y),
    );
    return res / (res.x + res.y + res.z);
}

fn oct_sample_positions(uv: vec2<f32>) -> SamplePositions {
    var sample_positions: SamplePositions;

    let grid_pos = uv * (f32(imposter_data.grid_size) - 1.0);
    sample_positions.tile_indices[0] = clamp(vec2<u32>(grid_pos), vec2(0u), vec2(imposter_data.grid_size - 2));

    let frac = clamp(grid_pos - vec2<f32>(sample_positions.tile_indices[0]), vec2(0.0), vec2(1.0));

    sample_positions.tile_weights = oct_sample_weights(frac);
    sample_positions.tile_indices[1] = sample_positions.tile_indices[0] + select(vec2(0u,1u), vec2(1u,0u), frac.x >= frac.y);
    sample_positions.tile_indices[2] = sample_positions.tile_indices[0] + vec2(1u,1u);

    return sample_positions;
}

fn sample_positions_from_camera_dir(dir: vec3<f32>) -> SamplePositions {
    let grid_size = f32(imposter_data.grid_size);

#ifdef GRID_HEMISPHERICAL
        // map direction to uv
        let dir2 = normalize(max(dir, vec3(-1.0, 0.0, -1.0)));
        let octant: vec3<f32> = sign(dir2);
        let sum: f32 = dot(dir2, octant);
        let octahedron: vec3<f32> = dir2 / sum;
        let uv = (vec2<f32>(octahedron.x + octahedron.z, octahedron.z - octahedron.x) + 1.0) * 0.5;
        
        return oct_sample_positions(uv);
#endif

#ifdef GRID_HORIZONTAL
        let dir2 = normalize(vec2(dir.x, dir.z));
        let angle = 0.5 - atan2(dir2.x, -dir2.y) / 6.283185307;
        let index = angle * f32(imposter_data.grid_size * imposter_data.grid_size);
        let l_index = u32(index);
        let r_index = l_index + 1u;
        var sample_positions: SamplePositions;
        sample_positions.tile_indices[0] = vec2(l_index % imposter_data.grid_size, (l_index / imposter_data.grid_size) % imposter_data.grid_size);
        sample_positions.tile_indices[1] = vec2(r_index % imposter_data.grid_size, (r_index / imposter_data.grid_size) % imposter_data.grid_size);
        sample_positions.tile_weights[1] = fract(index);
        sample_positions.tile_weights[0] = 1.0 - sample_positions.tile_weights[1];
        return sample_positions;
#endif

#ifdef GRID_SPHERICAL
        let uv = spherical_uv_from_normal(dir);
        return oct_sample_positions(uv);
#endif
}

struct Basis {
    normal: vec3<f32>,
    up: vec3<f32>,
}

fn oct_mode_normal_from_uv(grid_index: vec2<u32>, inv_rot: mat3x3<f32>) -> Basis {
    var n: vec3<f32>;

#ifdef GRID_HEMISPHERICAL
        let grid_count = f32(imposter_data.grid_size);
        let tile_origin = vec2<f32>(grid_index) / grid_count;
        let tile_size = 1.0 / grid_count;
        let uv = tile_origin * grid_count / (grid_count - 1.0);
        var x = uv.x - uv.y;
        var z = -1.0 + uv.x + uv.y;
        var y = 1.0 - abs(x) - abs(z);
        n = normalize(vec3(x, y, z));
#endif

#ifdef GRID_HORIZONTAL
        let index = grid_index.y * imposter_data.grid_size + grid_index.x;
        let angle: f32 = 6.283185307 * f32(index) / f32(imposter_data.grid_size * imposter_data.grid_size);
        let x: f32 = sin(angle);
        let z: f32 = cos(angle);
        n = vec3<f32>(x, 0.0, z);
#endif

#ifdef GRID_SPHERICAL
        let grid_count = f32(imposter_data.grid_size);
        let tile_origin = vec2<f32>(grid_index) / grid_count;
        let tile_size = 1.0 / grid_count;
        let uv = tile_origin * grid_count / (grid_count - 1.0);
        let uv2 = uv * (f32(imposter_data.grid_size) - 1.0) * f32(imposter_data.grid_size);
        n = spherical_normal_from_uv(uv);
#endif

    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(n.y) > 0.99);

    var basis: Basis;
    basis.normal = inv_rot * n;
    basis.up = inv_rot * up;
    return basis;
}

// Initial UV/depth + perspective slope for one sample of an imposter tile.
//   initial_uv    - texture UV at which to read the depth-and-material data
//   initial_depth - depth (relative to mid plane; +1 = near, -1 = far) that
//                   initial_uv corresponds to. The consumer must subtract this
//                   from the texture-read depth before applying dduddv.
//   dduddv        - shift in UV per +1 unit of depth (toward the near plane).
struct UVSample {
    initial_uv: vec2<f32>,
    initial_depth: f32,
    dduddv: vec2<f32>,
}

// Returns the anchor UV at which to read the depth/material texture, the depth
// of that anchor, and the per-unit-depth perspective slope (`dduddv`). For the
// common case where the perspective ray's mid-plane projection lands inside the
// content's V range, the anchor is just that mid-plane UV at depth 0.
//
// For look-up views the mid-plane projection can land *above* the content
// silhouette (empty texture). Rather than smear, anchor at a depth between the
// fragment and the depth at which the perspective ray leaves the content
// bounds — specifically the midpoint of those two depths — so the depth read
// lands inside the content where there's real data to drive the parallax.
fn sample_uvs_unbounded(base_world_position: vec3<f32>, world_position: vec3<f32>, inv_rot: mat3x3<f32>, grid_index: vec2<u32>, content_half_extent: vec3<f32>) -> UVSample {
    let basis = oct_mode_normal_from_uv(grid_index, inv_rot);
    let sample_r_vec = cross(basis.normal, -basis.up);
    let sample_u_vec = cross(sample_r_vec, basis.normal);
    let sample_r = normalize(sample_r_vec);
    let sample_u = normalize(sample_u_vec);
    // `+basis.normal` points from the imposter toward the bake camera, so the
    // plane at `base + n * scale` is the *near* plane (depth +1), not the back.
    let near_base_world_position = base_world_position + basis.normal * imposter_data.center_and_scale.w;

#ifdef VIEW_PROJECTION_ORTHOGRAPHIC
    let v = world_position - base_world_position;
    let x = dot(v, sample_r / (imposter_data.center_and_scale.w * 2.0));
    let y = dot(v, sample_u / (imposter_data.center_and_scale.w * 2.0));

    let near_v = world_position - near_base_world_position;
    let near_x = dot(near_v, sample_r / (imposter_data.center_and_scale.w * 2.0));
    let near_y = dot(near_v, sample_u / (imposter_data.center_and_scale.w * 2.0));
#else
    let camera_world_position = position_view_to_world(vec3<f32>(0.0));
    let cam_to_fragment = normalize(world_position - camera_world_position);

    let distance = dot(base_world_position - camera_world_position, basis.normal) / dot(cam_to_fragment, basis.normal);
    let intersect = distance * cam_to_fragment + camera_world_position;
    let v = intersect - base_world_position;
    let x = dot(v, sample_r / (imposter_data.center_and_scale.w * 2.0));
    let y = dot(v, sample_u / (imposter_data.center_and_scale.w * 2.0));

    let near_distance = dot(near_base_world_position - camera_world_position, basis.normal) / dot(cam_to_fragment, basis.normal);
    let near_intersect = near_distance * cam_to_fragment + camera_world_position;
    let near_v = near_intersect - near_base_world_position;
    let near_x = dot(near_v, sample_r / (imposter_data.center_and_scale.w * 2.0));
    let near_y = dot(near_v, sample_u / (imposter_data.center_and_scale.w * 2.0));
#endif

    let mid_uv = vec2<f32>(x, y) + 0.5;
    let near_uv = vec2<f32>(near_x, near_y) + 0.5;
    let dduddv = near_uv - mid_uv;

    var out: UVSample;
    out.dduddv = dduddv;
    out.initial_uv = mid_uv;
    out.initial_depth = 0.0;

#ifndef VIEW_PROJECTION_ORTHOGRAPHIC
    // Anchor the sample halfway between the fragment and where the perspective
    // ray exits content's V band, clamped to lie between the fragment and the
    // midplane so we never go past either endpoint. This smooths the
    // discontinuity at mid_uv == content_v_max — the previous "if mid OOB"
    // branch produced a visible horizontal seam there.
    //
    // Only `content_v_max` is honoured: the bake's content aabb is clipped
    // from below (we discard points below y=1 so very flat scenes stay
    // floor-only), so `content_v_min` is artificially raised and the actual
    // texture often holds content below it. Treating that bogus floor as a
    // real boundary mis-anchors look-down fragments. `content_v_max`, by
    // contrast, is genuine — nothing is clipped from above — so we use that
    // as the boundary and let the clamp handle the look-down direction (where
    // d_use collapses to `fragment_depth` and the anchor lands at the
    // fragment's own UV).
    let h_u = dot(content_half_extent, abs(sample_u));
    let content_half_v = h_u / (imposter_data.center_and_scale.w * 2.0);
    let content_v_max = clamp(0.5 + content_half_v, 0.0, 1.0);

    let fragment_depth = dot(world_position - base_world_position, basis.normal) / imposter_data.center_and_scale.w;
    let denom = select(dduddv.y, sign(dduddv.y) * 1e-6 + 1e-12, abs(dduddv.y) < 1e-6);
    let d_exit = (content_v_max - mid_uv.y) / denom;
    let d_use_raw = (fragment_depth + d_exit) * 0.5;
    let d_use = clamp(d_use_raw, min(fragment_depth, 0.0), max(fragment_depth, 0.0));

    out.initial_uv = mid_uv + d_use * dduddv;
    out.initial_depth = d_use;
#endif

    return out;
}

fn single_sample(coords: vec2<f32>, bounds_min: vec2<f32>, bounds_max: vec2<f32>, tile_idx_in_grid: u32) -> UnpackedMaterialProps {
    let oob_mask = vec2<u32>(select(1u, 0u, any(coords < bounds_min) || any(coords >= bounds_max)));

#ifdef INDEXED_PIXELS
    let pixel_dims = textureDimensions(imposter_pixels);
    var index: u32;

    if pixel_dims.x * pixel_dims.y < 65536 {
        // using u16 pairs
        let index_pair = textureLoad(imposter_indices, vec2<u32>(coords * vec2(0.5, 1.0)), 0).r;
        index = select(index_pair & 0xFFFF, index_pair >> 16, (u32(coords.x) & 1u) == 1u);
    } else {
        index = textureLoad(imposter_indices, vec2<u32>(coords), 0).r;
    }

    let index_x = index % pixel_dims.x;
    let index_y = index / pixel_dims.x;

    let props = textureLoad(imposter_pixels, vec2(index_x, index_y), 0).rg * oob_mask;
    return unpack_props(props);
#else
#ifdef INDEXED_V2
    // idx8: index is the byte read from imposter_indices.
    // idx12 (INDEXED_V2_12 set): also read high nibble from imposter_indices_hi
    //   — one byte per pair of source pixels, low nibble = even-x pixel,
    //   high nibble = odd-x pixel. Combine into a 12-bit palette index.
    // idx14 (INDEXED_V2_14 set): read full high byte from imposter_indices_hi
    //   (same width as imposter_indices), low 6 bits hold the high 6 bits
    //   of a 14-bit palette index.
    // idx10s (INDEXED_V2_10S set): read 2 high bits from imposter_indices_hi
    //   packed 4 pixels per byte (quarter-width). Mat+norm comes from the
    //   palette (R32Uint, tight 32-bit pack — no depth). Depth comes from
    //   imposter_depth_palette at (index, tile_idx_in_grid).
    let coords_u = vec2<u32>(coords);
    let pixel_dims = textureDimensions(imposter_pixels);
    let lo = textureLoad(imposter_indices, coords_u, 0).r;
#ifdef INDEXED_V3
    // v3: 8-bit index (lo) or 12-bit (lo + half-width high nibble). Mat+norm
    // from the R32Uint palette; depth is a per-pixel 4-bit *direct* value read
    // from the depth plane (half-width, two nibbles per byte, low=even-x) — no
    // per-tile palette, so depth is stable across viewing angles.
#ifdef INDEXED_V3_12
    let hi_byte_v3 = textureLoad(imposter_indices_hi, vec2<u32>(coords_u.x >> 1u, coords_u.y), 0).r;
    let hi_nib_v3 = (hi_byte_v3 >> ((coords_u.x & 1u) * 4u)) & 0xFu;
    let index = lo | (hi_nib_v3 << 8u);
#else
    let index = lo;
#endif
    let index_x = index % pixel_dims.x;
    let index_y = index / pixel_dims.x;
    let pack_v3 = textureLoad(imposter_pixels, vec2(index_x, index_y), 0).r * oob_mask.x;

    let d_byte_v3 = textureLoad(imposter_depth_palette, vec2<u32>(coords_u.x >> 1u, coords_u.y), 0).r;
    let d_nib_v3 = (d_byte_v3 >> ((coords_u.x & 1u) * 4u)) & 0xFu;
    let depth_f_v3 = f32(d_nib_v3) / 15.0 * 2.0 - 1.0;

    var props = unpack_props_10s(pack_v3);
    props.depth = depth_f_v3;
    return props;
#else
#ifdef INDEXED_V2_10S
    let hi_byte_10s = textureLoad(imposter_indices_hi, vec2<u32>(coords_u.x >> 2u, coords_u.y), 0).r;
    let hi_bits_10s = (hi_byte_10s >> ((coords_u.x & 3u) * 2u)) & 0x3u;
    let index = lo | (hi_bits_10s << 8u);

    let index_x = index % pixel_dims.x;
    let index_y = index / pixel_dims.x;
    // Tight 32-bit pack — read u32 (.r of the R32Uint texture).
    let pack_10s = textureLoad(imposter_pixels, vec2(index_x, index_y), 0).r * oob_mask.x;

    // Per-tile depth lookup: row = tile_idx_in_grid, column = palette
    // index >> 1 (two 4-bit depths packed per byte; low nibble = even
    // slot, high nibble = odd slot).
    let dp_byte = textureLoad(imposter_depth_palette, vec2(index >> 1u, tile_idx_in_grid), 0).r;
    let dp_nibble = (dp_byte >> ((index & 1u) * 4u)) & 0xFu;
    let depth_f = f32(dp_nibble) / 15.0 * 2.0 - 1.0;

    var props = unpack_props_10s(pack_10s);
    props.depth = depth_f;
    return props;
#else
#ifdef INDEXED_V2_14
    let hi_byte = textureLoad(imposter_indices_hi, coords_u, 0).r;
    let index = lo | ((hi_byte & 0x3Fu) << 8u);
#else
#ifdef INDEXED_V2_12
    let hi_byte = textureLoad(imposter_indices_hi, vec2<u32>(coords_u.x >> 1u, coords_u.y), 0).r;
    let hi_nibble = (hi_byte >> ((coords_u.x & 1u) * 4u)) & 0xFu;
    let index = lo | (hi_nibble << 8u);
#else
    let index = lo;
#endif
#endif

    let index_x = index % pixel_dims.x;
    let index_y = index / pixel_dims.x;

    let props = textureLoad(imposter_pixels, vec2(index_x, index_y), 0).rg * oob_mask;
    return unpack_props(props);
#endif
#endif
#else
    let props = textureLoad(imposter_pixels, vec2<u32>(coords), 0).rg * oob_mask;
    return unpack_props(props);
#endif
#endif
}

// Read at `coords` clamped to the tile bounds. Returns whatever's at the
// nearest in-bounds texel — used for the parallax-anchor read in
// `sample_tile_material` when the perspective intersection lands outside
// the tile (close-range views looking past the imposter silhouette). The
// final material read still uses `single_sample` so genuinely-empty regions
// continue to discard via alpha-zero.
fn single_sample_clamped(coords: vec2<f32>, bounds_min: vec2<f32>, bounds_max: vec2<f32>, tile_idx_in_grid: u32) -> UnpackedMaterialProps {
    return single_sample(
        clamp(coords, bounds_min, bounds_max - vec2<f32>(1.0)),
        bounds_min,
        bounds_max,
        tile_idx_in_grid,
    );
}

fn sample_tile_material(sample: UVSample, grid_index: vec2<u32>, coord_offset: vec2<f32>) -> UnpackedMaterialProps {
    let bounds_min = vec2<f32>(grid_index * imposter_data.packed_size);
    let bounds_max = bounds_min + vec2<f32>(imposter_data.packed_size);
    let coords_unadjusted = bounds_min - vec2<f32>(imposter_data.packed_offset) + sample.initial_uv * vec2<f32>(imposter_data.base_tile_size) + coord_offset;
    // Linear tile index for the idx10s per-tile depth-palette lookup.
    let tile_idx = grid_index.y * imposter_data.grid_size + grid_index.x;

    // The texture's `depth` field is the surface depth relative to the mid
    // plane. When the anchor UV is itself at `initial_depth != 0`, the
    // perspective shift from the anchor to the surface uses the *delta*.
    let depth_shift = sample.dduddv * vec2<f32>(imposter_data.base_tile_size);

#ifdef MATERIAL_MULTISAMPLE
        // multisample for depth
        let pixel_tl_depth = single_sample(coords_unadjusted, bounds_min, bounds_max, tile_idx);
        let pixel_tr_depth = single_sample(coords_unadjusted + vec2(1.0, 0.0), bounds_min, bounds_max, tile_idx);
        let pixel_bl_depth = single_sample(coords_unadjusted + vec2(0.0, 1.0), bounds_min, bounds_max, tile_idx);
        let pixel_br_depth = single_sample(coords_unadjusted + vec2(1.0, 1.0), bounds_min, bounds_max, tile_idx);

        let frac = clamp((fract(coords_unadjusted) - (imposter_data.multisample_amount / 2.0)) / (1.0 - imposter_data.multisample_amount), vec2(0.0), vec2(1.0));
        let pixel_top_depth = weighted_props(pixel_tl_depth, pixel_tr_depth, 1.0 - frac.x);
        let pixel_bottom_depth = weighted_props(pixel_bl_depth, pixel_br_depth, 1.0 - frac.x);
        let pixel_depth = weighted_props(pixel_top_depth, pixel_bottom_depth, 1.0 - frac.y);
        let depth = pixel_depth.depth;

        let coords = coords_unadjusted + (depth - sample.initial_depth) * depth_shift;

        // multisample final material
        let pixel_tl = single_sample(coords, bounds_min, bounds_max, tile_idx);
        let pixel_tr = single_sample(coords + vec2(1.0, 0.0), bounds_min, bounds_max, tile_idx);
        let pixel_bl = single_sample(coords + vec2(0.0, 1.0), bounds_min, bounds_max, tile_idx);
        let pixel_br = single_sample(coords + vec2(1.0, 1.0), bounds_min, bounds_max, tile_idx);

        let frac2 = clamp((fract(coords) - (imposter_data.multisample_amount / 2.0)) / (1.0 - imposter_data.multisample_amount), vec2(0.0), vec2(1.0));
        let pixel_top = weighted_props(pixel_tl, pixel_tr, 1.0 - frac2.x);
        let pixel_bottom = weighted_props(pixel_bl, pixel_br, 1.0 - frac2.x);
        var pixel = weighted_props(pixel_top, pixel_bottom, 1.0 - frac2.y);

        return pixel;
#else
        // Anchor read uses clamped coords so close-range views (where the
        // perspective intersection lands outside the silhouette) still recover
        // a depth value to shift with. The OOB case was the only thing
        // preventing the read from firing. After clamping, the read still
        // returns alpha=0 if the *clamped* coords are themselves on an empty
        // texel, in which case we leave coords unshifted and let the final
        // sample discard naturally.
        let pixel_depth = single_sample_clamped(coords_unadjusted, bounds_min, bounds_max, tile_idx);
        var coords = coords_unadjusted;
        if pixel_depth.rgba.a > 0.0 {
            coords = coords_unadjusted + (pixel_depth.depth - sample.initial_depth) * depth_shift;
        }
        let pixel = single_sample(coords, bounds_min, bounds_max, tile_idx);

        return pixel;
#endif
}
