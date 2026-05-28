#import bevy_pbr::{
    mesh_functions,
    view_transformations::{position_world_to_clip, position_view_to_world, direction_view_to_world, perspective_camera_near},
}

#ifdef PREPASS_PIPELINE
    #import bevy_pbr::prepass_io::Vertex;
#else
    #import bevy_pbr::forward_io::Vertex;
#endif

#import boimp::shared::ImposterVertexOut;
#import boimp::bindings::{imposter_data, sample_uvs_unbounded, grid_weights, sample_positions_from_camera_dir};

@vertex
fn vertex(vertex: Vertex) -> ImposterVertexOut {
    var out: ImposterVertexOut;

    var model = mesh_functions::get_world_from_local(vertex.instance_index);

    let center = imposter_data.center_and_scale.xyz;
    let scale = imposter_data.center_and_scale.w;

    let imposter_world_position = mesh_functions::mesh_position_local_to_world(model, vec4<f32>(center, 1.0)).xyz;
    let camera_world_position = position_view_to_world(vec3<f32>(0.0));

    // extract inverse rotation
    let inv_rot = transpose(mat3x3<f32>(
        normalize(model[0].xyz),
        normalize(model[1].xyz),
        normalize(model[2].xyz)
    ));
    // todo: we could pass the instance index instead, and extract in frag shader
    out.inverse_rotation_0c = inv_rot[0];
    out.inverse_rotation_1c = inv_rot[1];
    out.inverse_rotation_2c = inv_rot[2];
    out.base_world_position = imposter_world_position;
    // Half-extents of the render mesh in world units (column lengths of the
    // model matrix scale the unit-cube half-extent of 0.5).
    out.content_half_extent = 0.5 * vec3<f32>(
        length(model[0].xyz),
        length(model[1].xyz),
        length(model[2].xyz),
    );

    let back = direction_view_to_world(vec3<f32>(0.0, 0.0, 1.0));
    out.world_position = mesh_functions::mesh_position_local_to_world(model, vec4<f32>(vertex.position, 1.0)).xyz;

    out.position = position_world_to_clip(out.world_position);

    return out;
}
