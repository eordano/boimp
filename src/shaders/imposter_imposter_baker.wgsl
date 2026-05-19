#import boimp::shared::{ImposterVertexOut, compose_over, pack_props, unpack_props, weighted_props};
#import boimp::bindings::{sample_positions_from_camera_dir, sample_tile_material, sample_uvs_unbounded};

#import bevy_pbr::view_transformations::{direction_view_to_world, position_view_to_world};

@group(3) @binding(0) var bake_target: texture_storage_2d<rg32uint, read_write>;

@fragment
fn fragment(in: ImposterVertexOut) {
    let inv_rot = mat3x3(
        in.inverse_rotation_0c,
        in.inverse_rotation_1c,
        in.inverse_rotation_2c,
    );

    let camera_world_position = position_view_to_world(vec3<f32>(0.0));
#ifdef VIEW_PROJECTION_ORTHOGRAPHIC
    let back_vec = direction_view_to_world(vec3<f32>(0.0, 0.0, 1.0));
#else
    let back_vec = camera_world_position - in.base_world_position;
#endif

    let back = normalize(back_vec);

    let samples = sample_positions_from_camera_dir(back * inv_rot);

    let uv_a = sample_uvs_unbounded(in.base_world_position, in.world_position, inv_rot, samples.tile_indices[0]);
    let uv_b = sample_uvs_unbounded(in.base_world_position, in.world_position, inv_rot, samples.tile_indices[1]);

    let props_a = sample_tile_material(uv_a, samples.tile_indices[0], vec2(0.0));
    let props_b = sample_tile_material(uv_b, samples.tile_indices[1], vec2(0.0));

#ifndef GRID_HORIZONTAL
    let uv_c = sample_uvs_unbounded(in.base_world_position, in.world_position, inv_rot, samples.tile_indices[2]);
    let props_c = sample_tile_material(uv_c, samples.tile_indices[2], vec2(0.0));
#endif

    let weights = samples.tile_weights;
    let props_ab = weighted_props(props_a, props_b, weights.x / max(weights.x + weights.y, 0.0001));
#ifndef GRID_HORIZONTAL
    let props_final = weighted_props(props_ab, props_c, (weights.x + weights.y) / (weights.x + weights.y + weights.z));
#else
    let props_final = props_ab;
#endif

    if props_final.rgba.a <= 0.0 {
        discard;
    }

    // rotate the sampled normal into world space. depth stays as the source
    // imposter's parallax-correction depth — it represents where the surface
    // sits inside the imposter volume, not bake-camera screen depth.
    var new_props = props_final;
    new_props.normal = inv_rot * normalize(new_props.normal);

    let pixel = vec2<i32>(in.position.xy);
    let existing = unpack_props(textureLoad(bake_target, pixel));
    let composed = compose_over(existing, new_props);
    textureStore(bake_target, pixel, pack_props(composed));
}
