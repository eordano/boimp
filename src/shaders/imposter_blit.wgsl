#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;
#import boimp::shared::{pack_props, unpack_props, weighted_props, UnpackedMaterialProps};

struct BakeDims {
    width: u32,
}

struct BlitData {
    samples: u32,
    pad_0: u32,
    pad_1: u32,
    pad_2: u32,
}

// Read the bake buffer that the bake fragment shaders wrote to. `bake_dims.width`
// is the high-res (intermediate) bake-target width; downsample by `data.samples`
// (1 when MSAA is off) and return as a color-attachment write to the output texture.
@group(0) @binding(0) var<storage, read> bake_buffer: array<vec2<u32>>;
@group(0) @binding(1) var<uniform> bake_dims: BakeDims;
@group(0) @binding(2) var<uniform> data: BlitData;

@fragment
fn blend_materials(in: FullscreenVertexOutput) -> @location(0) vec2<u32> {
    // bake_buffer is sized for one tile at high-res (tile_size * samples).
    // `bake_dims.width` is that high-res width; the tile's output size is
    // therefore `bake_dims.width / samples`. uv runs [0,1] over the viewport
    // (one tile of the output), so a tile-local target pixel is uv * tile_size.
    let tile_target = bake_dims.width / data.samples;
    let target_pixel = vec2<u32>(in.uv * vec2<f32>(f32(tile_target), f32(tile_target)));
    // top-left of this output pixel's region in the source bake buffer.
    let src_origin = target_pixel * data.samples;

    var y_samples: array<UnpackedMaterialProps,8>;
    var y_end = data.samples;

    for (var y = 0u; y < data.samples; y ++) {
        var x_end = data.samples;
        var x_samples: array<UnpackedMaterialProps,8>;
        for (var x = 0u; x < data.samples; x++) {
            let src_pixel = src_origin + vec2(x, y);
            let idx = src_pixel.y * bake_dims.width + src_pixel.x;
            x_samples[x] = unpack_props(bake_buffer[idx]);
        }

        while x_end > 1u {
            x_end /= 2u;

            for (var x = 0u; x < x_end; x++) {
                x_samples[x] = weighted_props(x_samples[x], x_samples[x + x_end], 0.5);
            }
        }

        y_samples[y] = x_samples[0];
    }

    while y_end > 1u {
        y_end /= 2u;

        for (var y = 0u; y < y_end; y++) {
            y_samples[y] = weighted_props(y_samples[y], y_samples[y + y_end], 0.5);
        }
    }

    return pack_props(y_samples[0]);
}
