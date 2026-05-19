#import bevy_pbr::{
    prepass_io::VertexOutput,
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
}

#import boimp::shared::{compose_over, pack_pbrinput, pack_props, unpack_props};

struct BakeDims {
    width: u32,
}

@group(3) @binding(0) var<storage, read_write> bake_buffer: array<vec2<u32>>;
@group(3) @binding(1) var<uniform> bake_dims: BakeDims;

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) {
    // generate a PbrInput struct from the StandardMaterial bindings
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // material-specific alpha handling (mask cutoff / opaque snap / blend preserve)
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

    // skip fully transparent fragments (no contribution to the composite)
    if pbr_input.material.base_color.a <= 0.0 {
        discard;
    }

    // composite the new fragment over whatever's already at this pixel in the bake buffer.
    let new_packed = pack_pbrinput(pbr_input);
    let new_props = unpack_props(new_packed);

    let pixel = vec2<u32>(in.position.xy);
    let idx = pixel.y * bake_dims.width + pixel.x;
    let existing = unpack_props(bake_buffer[idx]);
    let composed = compose_over(existing, new_props);
    bake_buffer[idx] = pack_props(composed);
}
