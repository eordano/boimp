# boimp: Bevy 0.16 → 0.17 port notes

Branch: `0.17` (off `feat/composite-bake-via-storage-texture`).

boimp is built against the **DCL bevy fork** (`/home/dcl/bevy-fork`, branch
`release-0.17-dcl`), not crates.io bevy, because it depends on DCL-fork-only
APIs (see "Fork dependencies" below). `Cargo.toml` bumps `bevy` to `0.17`,
`wgpu` to `25`, and adds a `[patch.crates-io] bevy = { path =
"/home/dcl/bevy-fork" }`.

## Render-API changes made

### Cargo
- `bevy` `0.16` (git robtfm/release-0.16-dcl) → `0.17` + path-patch to the local
  DCL fork.
- `wgpu` `24` → `25`.

### bevy_render crate split (`src/bake.rs`)
0.17 split `bevy_render`; camera / projection / visibility / primitive types
moved to `bevy_camera` (facade `bevy::camera`). Imports were re-homed:
- `bevy::render::camera::{CameraProjection, ScalingMode, CameraOutputMode}`
  → `bevy::camera::{CameraProjection, ScalingMode, CameraOutputMode}`
- `bevy::render::primitives::{Aabb, Sphere}`
  → `bevy::camera::primitives::{Aabb, Sphere}`
- `bevy::render::view::{NoFrustumCulling, PreviousVisibleEntities, RenderLayers,
  VisibilitySystems, VisibleEntities}`
  → `bevy::camera::visibility::{...}`
- `CameraRenderGraph`, `ExtractedCamera`, `ColorGrading`, `ExtractedView`,
  `NoIndirectDrawing`, `RenderVisibleEntities`, `RetainedViewEntity`,
  `ViewDepthTexture`, `ViewUniformOffset` stayed under `bevy::render::{camera,view}`.
- `OrthographicProjection` / `ClearColorConfig` resolve via `bevy::prelude`
  (now sourced from `bevy_camera`'s prelude). No code change needed.
- `MeshVertexBufferLayoutRef`: `bevy::render::mesh::…` → `bevy::mesh::…`
  (both `bake.rs` and `render.rs`).

### Material / prepass pipeline rework — the "erased material pipeline" refactor
This is the deep one. In 0.17 the material pipeline machinery is no longer
generic over `M`:
- `MaterialPipeline<M>` → non-generic `MaterialPipeline`. boimp's
  `Material::specialize` (in `render.rs`) signature: `&MaterialPipeline<Self>`
  → `&MaterialPipeline`. The trait still hands you a typed
  `MaterialPipelineKey<Self>` (the engine downcasts the erased key for you), so
  the body of `Imposter::specialize` is unchanged.
- `PrepassPipeline<M>` → non-generic `PrepassPipeline` + a per-material
  `PrepassPipelineSpecializer { pipeline, properties }` that implements
  `SpecializedMeshPipeline` with `Key = ErasedMaterialPipelineKey`.
  - boimp's `ImposterBakePipeline<M>` previously wrapped `PrepassPipeline<M>`
    and called `self.prepass_pipeline.specialize(MaterialPipelineKey<M>, …)`.
    Now there is a new `ImposterBakePipelineSpecializer<M>` that wraps a
    `PrepassPipelineSpecializer` (constructed per material from its
    `Arc<MaterialProperties>`), runs it with an `ErasedMaterialPipelineKey`,
    then applies boimp's bake overrides. `ImposterBakePipeline<M>` is now just
    a resource holding the cloned `PrepassPipeline` + frag shader + storage
    layout, used to build the per-material specializer inside the queue system.
  - `SpecializedMeshPipelines<ImposterBakePipeline<M>>` →
    `SpecializedMeshPipelines<ImposterBakePipelineSpecializer<M>>`.
- `PrepassPipeline` no longer impls `FromWorld`; it is created by the
  `init_prepass_pipeline` system in the new `RenderStartup` schedule. So
  `ImposterBakePipeline::<M>::from_world` was removed; pipeline init moved to a
  `RenderStartup` system `init_imposter_bake_pipeline::<M>` that reads
  `Res<PrepassPipeline>`.
- Prepared materials are now untyped/erased:
  - `RenderAssets<PreparedMaterial<M>>` → `ErasedRenderAssets<PreparedMaterial>`
    (look up by untyped `material_instance.asset_id`; filter on
    `asset_id.type_id() == TypeId::of::<M>()`).
  - `MaterialBindGroupAllocator<M>` → `MaterialBindGroupAllocators` (plural),
    indexed by `TypeId::of::<M>()`.
  - `material_bind_group.get_extra_data(slot)` is gone. The old `M::Data`
    bind-group-data now lives in `material.properties.material_key`
    (`ErasedMaterialKey`); boimp builds an `ErasedMaterialPipelineKey { mesh_key,
    material_key, type_id }` directly.
  - queue ordering: `.after(prepare_assets::<PreparedMaterial<M>>)` →
    `.after(prepare_erased_assets::<MeshMaterial3d<M>>)`.

### Draw command & bind-group indices (`src/bake.rs`, shaders)
0.17 prepass bind-group layout is now `[view(0), empty(1), mesh(2),
material(3)]` (was `[view(0), mesh(1), material(2)]`), and
`MATERIAL_BIND_GROUP_INDEX == 3`.
- `DrawImposter`: was `(SetItemPipeline, SetPrepassViewBindGroup<0>,
  SetMeshBindGroup<1>, SetMaterialBindGroup<M,2>, DrawMesh, CountRenderCommand)`.
  Now `(SetItemPipeline, SetPrepassViewBindGroup<0>,
  SetPrepassViewEmptyBindGroup<1>, SetMeshBindGroup<2>, SetMaterialBindGroup<3>,
  DrawMesh, CountRenderCommand)`. `SetMaterialBindGroup` is **no longer generic
  over `M`** (just an index), so `DrawImposter` is no longer generic and is
  registered **once** in `ImposterBakePlugin::finish` instead of per material
  plugin.
- The bake storage bind group, appended after the prepass layout, now lands at
  **group 4** (was 3). Updated:
  - bake node: `render_pass.set_bind_group(3, …)` → `set_bind_group(4, …)`.
  - `shaders/standard_material_imposter_baker.wgsl` and
    `shaders/imposter_imposter_baker.wgsl`: `@group(3)` → `@group(4)` for
    `bake_buffer` / `bake_dims`.
- Runtime Imposter material bindings: `shaders/bindings.wgsl` `@group(2)` →
  `@group(#{MATERIAL_BIND_GROUP})` (resolves to 3). This is the wgpu-25 /
  3D-material-bind-group migration.

### Misc render-resource API
- `FragmentState::entry_point` is now `Option<Cow<…>>`: `"x".into()` →
  `Some("x".into())` (bake specializer + blit pipeline).
- `fullscreen_shader_vertex_state()` free fn removed → `FullscreenShader`
  resource's `.to_vertex_state()` (read in `ImposterBlitPipeline::from_world`;
  `FullscreenShader` is created in `CorePipelinePlugin::build`, so it's
  available at `finish()` time). Import:
  `bevy::core_pipeline::FullscreenShader`.

### Unchanged (DCL fork kept back-compat)
- `RenderSet` still exists as a type alias for `RenderSystems`.
- `weak_handle!` and `load_internal_asset!` still exist.
- `ExtractedView` / `ExtractedCamera` field shapes unchanged.

## Fork dependencies — REQUIRED changes to `/home/dcl/bevy-fork`

boimp cannot finish compiling against the fork until these land:

1. **`RenderAssetTransferPriority` / `Image::transfer_priority`** — DCL-fork-only
   API (present in `release-0.16-dcl`, NOT yet cherry-picked into
   `release-0.17-dcl`). Used in `src/asset_loader.rs`
   (`ImposterLoaderSettings.transfer_priority`, `image.transfer_priority = …`).
   It lives in `bevy_asset::render_asset` + `transfer_priority` fields on
   `Image` / `Mesh` / image-loader settings. The parallel 0.17-dcl port must
   include this commit (it was in-flight — `crates/bevy_image/src/image.rs` was
   an unmerged file during this work).

2. **`PrepassPipelineSpecializer` field visibility** — its `pipeline` and
   `properties` fields are `pub(crate)` in upstream 0.17, so boimp (out of
   crate) cannot construct one. boimp's bake pipeline needs to build a
   `PrepassPipelineSpecializer` to reuse the prepass specialization logic
   (the alternative is copy/pasting ~250 lines of prepass specialize into
   boimp). Required fork change in
   `crates/bevy_pbr/src/prepass/mod.rs`:
   ```rust
   pub struct PrepassPipelineSpecializer {
       pub pipeline: PrepassPipeline,
       pub properties: Arc<MaterialProperties>,
   }
   ```
   (or add a `pub fn new(pipeline, properties)` constructor). I could not make
   this edit myself (the fork is owned by a parallel agent and edits there were
   blocked).

## Build status
Not yet verified clean: the DCL fork was mid-merge (unresolved conflict markers
in `bevy_core_pipeline`/`bevy_image`, plus the two items above) and did not
compile during this work. boimp's own source is ported to the 0.17 APIs as
documented; final `cargo build` verification is blocked on the fork compiling
and on fork change (2) above.

---

## Fork-integration completion (2026-06-19)

The crate now builds green against the DCL fork. `cargo build`,
`cargo build --examples`, and `cargo build --all-targets` all finish with 0
errors (warnings only). Fixes applied this pass:

### wgpu version bump (the big one)
`Cargo.toml`: `wgpu = "25"` -> `wgpu = "26"`. The fork's `bevy_render`/`bevy_image`
use `wgpu-types` 26, but boimp pinned wgpu 25, so two copies of `wgpu_types`
coexisted in the graph and every `Extent3d`/`TextureDimension`/`TextureFormat`
passed into `Image::new`/`GpuImage`/`create_buffer` was the wrong type
(~30 E0308 "arguments are incorrect" / "mismatched types"). Bumping to 26
unified the types and cleared all of them at once. `Image::new`'s signature
itself is unchanged in the fork (`size, dimension, data, format, asset_usage`);
the new `transfer_priority` field is filled internally by `new_uninit`, so
boimp's call sites needed no extra args.

### TextureFormat::pixel_size() now returns Result
The fork's `pixel_size()` returns `Result<usize, TextureAccessError>`. Added
`.unwrap()` at the 3 boimp call sites (all on the fixed `Rg32Uint` format, which
never errors): `bake.rs` lines ~1757, ~1771, ~1851. This cleared the 3 E0605
non-primitive-cast and 2 E0277 "cannot multiply usize by Result" errors.

### Shader types moved to bevy_shader
`ShaderRef` / `ShaderDefVal` left `bevy::render::render_resource` for
`bevy::shader` (facade for `bevy_shader`). Updated imports in `bake.rs` and
`render.rs`.

### RenderGraphApp -> RenderGraphExt
The render-graph app-extension trait was renamed; `add_render_sub_graph` lives on
`RenderGraphExt` now (`bevy::render::render_graph::RenderGraphExt`). Updated the
import in `bake.rs`.

### QueryItem / ROQueryItem gained a second lifetime
`QueryItem<'w, Q>` -> `QueryItem<'w, '_, Q>` (and same for `ROQueryItem`) in the
`ViewNode::run` and `RenderCommand::render` impls in `bake.rs`.

### ExtendedMaterial gained a base-material generic
`impl ... for ExtendedMaterial<E>` -> `impl<B: Material, ...> ... for
ExtendedMaterial<B, E>` to match bevy 0.17's `ExtendedMaterial<B, E>`.

### RenderPassColorAttachment gained depth_slice
Added `depth_slice: None` to the `wgpu::RenderPassColorAttachment` literal in
`bake.rs` (wgpu 26 field).

### Examples (0.17 API migrations, not fork-specific)
- `bevy::render::primitives::{Aabb, Sphere}` -> `bevy::camera::primitives::...`
  (culling `Sphere` with `.center`, distinct from the `prelude::Sphere` mesh
  primitive which only has `radius`); `bevy::render::view::RenderLayers` ->
  `bevy::camera::visibility::RenderLayers` (save_asset.rs, dynamic.rs).
- `Handle::clone_weak()` removed -> `.clone()` (save_asset.rs, dynamic.rs).
- `bevy::render::mesh::VertexAttributeValues` -> `bevy::mesh::VertexAttributeValues`
  (custom_mesh.rs).
- Cursor settings split out of `Window` into a `CursorOptions` component:
  query changed to `(&Window, &mut CursorOptions)` and field accesses moved off
  `window.cursor_options.*` (helpers/camera_controller.rs).

### Status
GREEN. `dcl-shell -c "cd /home/dcl/boimp-fork && cargo build"` succeeds; the
earlier "fork mid-merge" blocker noted above is resolved.
