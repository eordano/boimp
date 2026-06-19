use bevy::{
    asset::{load_internal_asset, weak_handle},
    prelude::*,
    render::{
        render_asset::RenderAssetUsages,
        render_resource::{AsBindGroup, ShaderType},
    },
    // bevy 0.17 moved ShaderRef out of bevy_render into bevy_shader (facade: `bevy::shader`).
    shader::ShaderRef,
};
use wgpu::{Extent3d, TextureFormat};

use crate::{
    asset_loader::ImposterLoader,
    oct_coords::{GridMode, GRID_MASK},
};

pub const BINDINGS_HANDLE: Handle<Shader> = weak_handle!("41cdb8d9-25b9-4401-9986-33c19d184969");
pub const FRAGMENT_HANDLE: Handle<Shader> = weak_handle!("a7a24f93-e6a5-4ff0-b9c8-caf8c1c0cd97");
pub const SHARED_HANDLE: Handle<Shader> = weak_handle!("4ad87b7e-0802-4fd0-aa65-7de1474fd6bf");
pub const VERTEX_HANDLE: Handle<Shader> = weak_handle!("a3511116-2c4e-4f43-b69a-7c9391eaab54");

pub const RENDER_MULTISAMPLE_FLAG: u32 = 16;
pub const INDEXED_FLAG: u32 = 32;
/// V2 indexed format (palette + R8Uint indices). Mutually exclusive with
/// `INDEXED_FLAG`.
pub const INDEXED_V2_FLAG: u32 = 64;
/// Set together with `INDEXED_V2_FLAG` to indicate the idx12 variant —
/// runtime needs to combine a low-byte and high-nibble texture per pixel.
/// When unset under v2 we're on idx8 (single texture, low byte only).
pub const V2_HAS_HIGH_NIBBLE_FLAG: u32 = 128;
/// Set together with `INDEXED_V2_FLAG` to indicate the idx14 variant —
/// 14-bit indices stored as low-byte + full high-byte (with 2 unused MSBs).
/// Mutually exclusive with `V2_HAS_HIGH_NIBBLE_FLAG`.
pub const V2_HAS_HIGH_BYTE_FLAG: u32 = 256;
/// Set together with `INDEXED_V2_FLAG` to indicate the idx10s variant —
/// 10-bit indices (low-byte + 2 high bits packed 4 pixels per byte in a
/// quarter-width texture) addressing a tight 32-bit-per-entry palette of
/// mat+norm (no depth). Depth lives in a separate per-tile palette,
/// addressed by `(palette_idx, tile_index_in_grid)`. Mutually exclusive
/// with `V2_HAS_HIGH_NIBBLE_FLAG` and `V2_HAS_HIGH_BYTE_FLAG`.
pub const V2_IDX10S_FLAG: u32 = 512;
/// Set together with `INDEXED_V2_FLAG` to indicate the v3 variant — a
/// mat+norm-only 32-bit palette (no depth, no merge beyond a 4096 cap) with a
/// decoupled per-pixel 4-bit *direct* depth plane (reusing the depth-palette
/// binding as a half-width R8Uint texture). Index is 8-bit by default, or
/// 12-bit when `V2_HAS_HIGH_NIBBLE_FLAG` is also set. Takes dispatch priority
/// over the idx10s/idx12/idx14 flags.
pub const V2_V3_FLAG: u32 = 1024;

pub struct ImposterRenderPlugin;

impl Plugin for ImposterRenderPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            BINDINGS_HANDLE,
            "shaders/bindings.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            FRAGMENT_HANDLE,
            "shaders/fragment.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(app, SHARED_HANDLE, "shaders/shared.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, VERTEX_HANDLE, "shaders/vertex.wgsl", Shader::from_wgsl);

        app.add_plugins(MaterialPlugin::<Imposter>::default())
            .register_asset_loader(ImposterLoader)
            .add_systems(Startup, setup);
    }
}

/// provides a fallback image for imposter indices, for use with dynamic imposting
#[derive(Resource)]
pub struct DummyIndicesImage(pub Handle<Image>);

pub fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let image = Image::new(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        wgpu::TextureDimension::D2,
        vec![0, 0, 0, 0],
        TextureFormat::R32Uint,
        RenderAssetUsages::RENDER_WORLD,
    );
    commands.insert_resource(DummyIndicesImage(images.add(image)));
}

#[derive(ShaderType, Clone, Copy, PartialEq, Debug)]
pub struct ImposterData {
    pub center_and_scale: Vec4,
    pub packed_tile_offset: UVec2,
    pub packed_tile_size: UVec2,
    pub grid_size: u32,
    pub base_tile_size: u32,
    pub flags: u32,
    pub alpha: f32,
    pub multisample_amount: f32,
}

impl ImposterData {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        center: Vec3,
        scale: f32,
        grid_size: u32,
        base_tile_size: u32,
        packed_tile_offset: UVec2,
        packed_tile_size: UVec2,
        mode: GridMode,
        multisample: bool,
        indexed: bool,
        alpha: f32,
        multisample_amount: f32,
    ) -> Self {
        Self {
            center_and_scale: center.extend(scale),
            grid_size,
            base_tile_size,
            packed_tile_offset,
            packed_tile_size,
            flags: mode.as_flags()
                + if multisample {
                    RENDER_MULTISAMPLE_FLAG
                } else {
                    0
                }
                + if indexed { INDEXED_FLAG } else { 0 },
            alpha,
            multisample_amount,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ImposterKey(u32);

#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
#[bind_group_data(ImposterKey)]
pub struct Imposter {
    #[uniform(0)]
    pub data: ImposterData,
    #[texture(1, dimension = "2d", sample_type = "u_int")]
    pub pixels: Handle<Image>,
    // annoyingly we can't use an option here because bevy gives us an rgba8 fallback
    // Res<DummyIndicesImage> gives a default you can drop in
    #[texture(2, dimension = "2d", sample_type = "u_int")]
    pub indices: Handle<Image>,
    /// V2 idx12 high-nibble texture (R8Uint, half-width). Always bound,
    /// even on v1 / non-indexed assets — use the `DummyIndicesImage`
    /// fallback when there's nothing to read. The shader only reads it
    /// under the `INDEXED_V2_12` shader_def.
    #[texture(3, dimension = "2d", sample_type = "u_int")]
    pub indices_hi: Handle<Image>,
    /// V2 idx10s per-tile depth palette (R8Uint, `palette_len × tile_count`).
    /// Always bound — use `DummyIndicesImage` fallback for non-idx10s assets.
    /// The shader only reads it under the `INDEXED_V2_10S` shader_def,
    /// looking up `(palette_idx, tile_index_in_grid)`.
    #[texture(4, dimension = "2d", sample_type = "u_int")]
    pub depth_palette: Handle<Image>,
    pub alpha_mode: AlphaMode,
    pub vram_bytes: usize,
}

impl From<&Imposter> for ImposterKey {
    fn from(value: &Imposter) -> Self {
        Self(value.data.flags)
    }
}

impl Material for Imposter {
    fn vertex_shader() -> ShaderRef {
        VERTEX_HANDLE.into()
    }

    fn prepass_vertex_shader() -> ShaderRef {
        VERTEX_HANDLE.into()
    }

    fn fragment_shader() -> ShaderRef {
        FRAGMENT_HANDLE.into()
    }

    fn prepass_fragment_shader() -> ShaderRef {
        FRAGMENT_HANDLE.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }

    fn specialize(
        _: &bevy::pbr::MaterialPipeline,
        descriptor: &mut bevy::render::render_resource::RenderPipelineDescriptor,
        _: &bevy::mesh::MeshVertexBufferLayoutRef,
        key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), bevy::render::render_resource::SpecializedMeshPipelineError> {
        let vert_defs = &mut descriptor.vertex.shader_defs;
        let frag_defs = &mut descriptor.fragment.as_mut().unwrap().shader_defs;

        if (key.bind_group_data.0 & RENDER_MULTISAMPLE_FLAG) != 0 {
            frag_defs.push("MATERIAL_MULTISAMPLE".into());
        }
        let grid_mode = match key.bind_group_data.0 & GRID_MASK {
            i if i == GridMode::Hemispherical.as_flags() => "GRID_HEMISPHERICAL",
            i if i == GridMode::Spherical.as_flags() => "GRID_SPHERICAL",
            i if i == GridMode::Horizontal.as_flags() => "GRID_HORIZONTAL",
            _ => panic!(),
        };
        vert_defs.push(grid_mode.into());
        frag_defs.push(grid_mode.into());

        if (key.bind_group_data.0 & INDEXED_FLAG) != 0 {
            // legacy indexed
            frag_defs.push("INDEXED_PIXELS".into());
        }
        if (key.bind_group_data.0 & INDEXED_V2_FLAG) != 0 {
            frag_defs.push("INDEXED_V2".into());
            // Mutually-exclusive variant dispatch. v3 takes priority — it reuses
            // V2_HAS_HIGH_NIBBLE_FLAG to mean "12-bit index" *within* v3, so it
            // must be checked before the idx12 case.
            if (key.bind_group_data.0 & V2_V3_FLAG) != 0 {
                frag_defs.push("INDEXED_V3".into());
                if (key.bind_group_data.0 & V2_HAS_HIGH_NIBBLE_FLAG) != 0 {
                    frag_defs.push("INDEXED_V3_12".into());
                }
            } else if (key.bind_group_data.0 & V2_IDX10S_FLAG) != 0 {
                frag_defs.push("INDEXED_V2_10S".into());
            } else if (key.bind_group_data.0 & V2_HAS_HIGH_NIBBLE_FLAG) != 0 {
                frag_defs.push("INDEXED_V2_12".into());
            } else if (key.bind_group_data.0 & V2_HAS_HIGH_BYTE_FLAG) != 0 {
                frag_defs.push("INDEXED_V2_14".into());
            }
        }

        Ok(())
    }
}
