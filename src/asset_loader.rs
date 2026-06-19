use core::str;
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{BufReader, Cursor, Read, Write},
    path::PathBuf,
};

use anyhow::anyhow;
use bevy::{
    asset::{AssetLoader, Handle, RenderAssetTransferPriority},
    log::{debug, info},
    math::{UVec2, Vec3},
    prelude::{AlphaMode, Image, TypePath},
    render::render_asset::RenderAssetUsages,
};
use image::{DynamicImage, ImageBuffer};
use serde::{Deserialize, Serialize};
use wgpu::{Extent3d, TextureFormat};

use crate::{
    oct_coords::GridMode,
    render::{
        Imposter, ImposterData, INDEXED_FLAG, INDEXED_V2_FLAG, RENDER_MULTISAMPLE_FLAG,
        V2_HAS_HIGH_BYTE_FLAG, V2_HAS_HIGH_NIBBLE_FLAG, V2_IDX10S_FLAG, V2_V3_FLAG,
    },
};

// bevy 0.18: AssetLoader now requires TypePath.
#[derive(TypePath)]
pub struct ImposterLoader;

#[derive(Serialize, Deserialize)]
pub struct ImposterLoaderSettings {
    // smooth sample the material texture
    pub multisample: bool,
    // additional multiplier
    pub alpha: f32,
    // roughly alpha mode. 0 -> Blend, 1 -> Opaque, (0-1) -> Mask
    // if you need more control you can modify the loaded asset (we can't put actual alpha mode here because it doesn't serialize)
    pub alpha_blend: f32,
    pub multisample_amount: f32,
    pub transfer_priority: RenderAssetTransferPriority,
    pub asset_usages: RenderAssetUsages,
}

impl Default for ImposterLoaderSettings {
    fn default() -> Self {
        Self {
            multisample: Default::default(),
            alpha: 1.0,
            alpha_blend: 0.0,
            multisample_amount: 0.99,
            transfer_priority: RenderAssetTransferPriority::default(),
            asset_usages: Default::default(),
        }
    }
}

impl AssetLoader for ImposterLoader {
    type Asset = Imposter;

    type Settings = ImposterLoaderSettings;

    type Error = anyhow::Error;

    fn load(
        &self,
        reader: &mut dyn bevy::asset::io::Reader,
        load_settings: &Self::Settings,
        load_context: &mut bevy::asset::LoadContext,
    ) -> impl bevy::tasks::ConditionalSendFuture<Output = Result<Self::Asset, Self::Error>> {
        Box::pin(async move {
            let mut bytes = Vec::new();
            reader
                .read_to_end(&mut bytes)
                .await
                .map_err(|_| anyhow!("read failed"))?;
            let cursor = Cursor::new(&bytes[..]);
            let mut zip = zip::ZipArchive::new(cursor)?;
            let settings = BufReader::new(zip.by_name("settings.txt")?)
                .bytes()
                .collect::<Result<Vec<_>, _>>()?;
            let mut parts = str::from_utf8(&settings)?.split(' ');
            let (
                Some(grid_size),
                Some(scale),
                Some(mode),
                Some(base_tile_size),
                Some(packed_offset_x),
                Some(packed_offset_y),
                Some(packed_size_x),
                Some(packed_size_y),
            ) = (
                parts.next(),
                parts.next(),
                parts.next(),
                parts.next(),
                parts.next(),
                parts.next(),
                parts.next(),
                parts.next(),
            )
            else {
                anyhow::bail!("bad format for settings: `{:?}`", settings);
            };
            let grid_size = grid_size.parse()?;
            let scale = scale.parse()?;
            let base_tile_size = base_tile_size.parse()?;
            let packed_tile_offset = UVec2::new(packed_offset_x.parse()?, packed_offset_y.parse()?);
            let packed_tile_size = UVec2::new(packed_size_x.parse()?, packed_size_y.parse()?);

            // V2 marker — present immediately after the original 8 fields.
            // V2 files contain `palette.png` plus either `idx.png` (idx8) or
            // `idx_lo.png` + `idx_hi.png` (idx12); the legacy decoder would
            // fail file-not-found on these names, which is the designed
            // coexistence story.
            let v2_marker = parts.next();
            if v2_marker == Some("v2") {
                let variant = parts
                    .next()
                    .ok_or_else(|| anyhow!("v2 settings missing variant"))?;
                let palette_count: u32 = parts
                    .next()
                    .ok_or_else(|| anyhow!("v2 settings missing palette_count"))?
                    .parse()?;
                let palette_x: u32 = parts
                    .next()
                    .ok_or_else(|| anyhow!("v2 settings missing palette_x"))?
                    .parse()?;
                let palette_y: u32 = parts
                    .next()
                    .ok_or_else(|| anyhow!("v2 settings missing palette_y"))?
                    .parse()?;
                return load_v2(
                    &mut zip,
                    load_settings,
                    load_context,
                    mode,
                    grid_size,
                    scale,
                    base_tile_size,
                    packed_tile_offset,
                    packed_tile_size,
                    variant,
                    palette_count,
                    palette_x,
                    palette_y,
                );
            }

            let is_indexed = zip.file_names().any(|n| n == "pixels.png");
            let (pixels_image, indices_image, vram_bytes) = if is_indexed {
                let raw_pixels = BufReader::new(zip.by_name("pixels.png")?)
                    .bytes()
                    .collect::<Result<Vec<_>, _>>()?;
                let mut reader = image::ImageReader::new(std::io::Cursor::new(raw_pixels));
                reader.set_format(image::ImageFormat::Png);
                reader.no_limits();
                let pixels_bytes = reader.decode()?.into_bytes();
                let pixels_x = (pixels_bytes.len() as f32 / 8.0).sqrt().ceil() as u32;
                let pixels_y = (pixels_bytes.len() as f32 / (8 * pixels_x) as f32).ceil() as u32;
                let mut pixels_image = Image::new(
                    Extent3d {
                        width: pixels_x,
                        height: pixels_y,
                        depth_or_array_layers: 1,
                    },
                    wgpu::TextureDimension::D2,
                    pixels_bytes,
                    TextureFormat::Rg32Uint,
                    load_settings.asset_usages,
                );
                pixels_image.transfer_priority = load_settings.transfer_priority;
                let pixels_image =
                    load_context.add_labeled_asset("pixels".to_owned(), pixels_image);

                let raw_indices = BufReader::new(zip.by_name("indices.png")?)
                    .bytes()
                    .collect::<Result<Vec<_>, _>>()?;
                let mut reader = image::ImageReader::new(std::io::Cursor::new(raw_indices));
                reader.set_format(image::ImageFormat::Png);
                reader.no_limits();
                let indices_bytes = reader.decode()?.into_bytes();

                let use_u16 = pixels_x * pixels_y < 65536;

                let size: UVec2 = packed_tile_size * grid_size;
                let width = if use_u16 { size.x.div_ceil(2) } else { size.x };
                debug!(
                    "load use_u16? {use_u16}, base size: {}, use size: {}, height: {}, total pix: {}",
                    size.x,
                    width,
                    size.y,
                    indices_bytes.len()
                );
                let mut indices_image = Image::new(
                    Extent3d {
                        width,
                        height: size.y,
                        depth_or_array_layers: 1,
                    },
                    wgpu::TextureDimension::D2,
                    indices_bytes,
                    TextureFormat::R32Uint,
                    load_settings.asset_usages,
                );
                indices_image.transfer_priority = load_settings.transfer_priority;
                let indices_image =
                    load_context.add_labeled_asset("indices".to_owned(), indices_image);
                (
                    pixels_image,
                    indices_image,
                    pixels_x * pixels_y * 8 + width * size.y * 4,
                )
            } else {
                let raw_image = BufReader::new(zip.by_name("texture.png")?)
                    .bytes()
                    .collect::<Result<Vec<_>, _>>()?;
                let mut reader = image::ImageReader::new(std::io::Cursor::new(raw_image));
                reader.set_format(image::ImageFormat::Png);
                reader.no_limits();
                let pixels_bytes = reader.decode()?.into_bytes();
                let size: UVec2 = packed_tile_size * grid_size;
                let mut pixels_image = Image::new(
                    Extent3d {
                        width: size.x,
                        height: size.y,
                        depth_or_array_layers: 1,
                    },
                    wgpu::TextureDimension::D2,
                    pixels_bytes,
                    TextureFormat::Rg32Uint,
                    load_settings.asset_usages,
                );
                pixels_image.transfer_priority = load_settings.transfer_priority;
                let pixels_image =
                    load_context.add_labeled_asset("texture".to_owned(), pixels_image);

                let mut indices_image = Image::new(
                    Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    wgpu::TextureDimension::D2,
                    vec![0, 0, 0, 0],
                    TextureFormat::R32Uint,
                    load_settings.asset_usages,
                );
                indices_image.transfer_priority = load_settings.transfer_priority;
                let indices_image =
                    load_context.add_labeled_asset("dummy_indices".to_owned(), indices_image);

                (pixels_image, indices_image, size.x * size.y * 8)
            };

            let flags = match load_settings.multisample {
                true => RENDER_MULTISAMPLE_FLAG,
                false => 0,
            } + match mode {
                "spherical" => GridMode::Spherical,
                "hemispherical" => GridMode::Hemispherical,
                "Horizontal" => GridMode::Horizontal,
                _ => anyhow::bail!("bad mode `{}`", mode),
            }
            .as_flags()
                + if is_indexed { INDEXED_FLAG } else { 0 };

            let alpha_mode = if load_settings.alpha_blend == 0.0 {
                AlphaMode::Blend
            } else if load_settings.alpha_blend == 1.0 {
                AlphaMode::Opaque
            } else {
                AlphaMode::Mask(load_settings.alpha_blend)
            };

            // V1 doesn't use the v2 high-nibble texture or per-tile depth
            // palette; bind 1x1 R8Uint dummies so the bind group layout has
            // something to point at.
            let indices_hi = make_dummy_r8_image(load_context, "v1_dummy_indices_hi");
            let depth_palette = make_dummy_r8_image(load_context, "v1_dummy_depth_palette");

            Ok(Imposter {
                data: ImposterData {
                    center_and_scale: Vec3::ZERO.extend(scale),
                    grid_size,
                    flags,
                    alpha: load_settings.alpha,
                    base_tile_size,
                    packed_tile_offset,
                    packed_tile_size,
                    multisample_amount: (1.0 - load_settings.multisample_amount).clamp(0.0, 0.99),
                },
                pixels: pixels_image,
                indices: indices_image,
                indices_hi,
                depth_palette,
                alpha_mode,
                vram_bytes: vram_bytes as usize,
            })
        })
    }

    fn extensions(&self) -> &[&str] {
        &["boimp"]
    }
}

fn make_dummy_r8_image(load_context: &mut bevy::asset::LoadContext, label: &str) -> Handle<Image> {
    let image = Image::new(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        wgpu::TextureDimension::D2,
        vec![0u8],
        TextureFormat::R8Uint,
        RenderAssetUsages::RENDER_WORLD,
    );
    load_context.add_labeled_asset(label.to_owned(), image)
}

#[allow(clippy::too_many_arguments)]
fn load_v2(
    zip: &mut zip::ZipArchive<Cursor<&[u8]>>,
    load_settings: &ImposterLoaderSettings,
    load_context: &mut bevy::asset::LoadContext,
    mode: &str,
    grid_size: u32,
    scale: f32,
    base_tile_size: u32,
    packed_tile_offset: UVec2,
    packed_tile_size: UVec2,
    variant: &str,
    _palette_count: u32,
    palette_x: u32,
    palette_y: u32,
) -> Result<Imposter, anyhow::Error> {
    // palette.png — RGBA8 layout. For idx8/12/14: width = palette_x*2, height =
    // palette_y, reinterpreted as Rg32Uint (8-byte mat+norm+depth per texel).
    // For idx10s: width = palette_x, height = palette_y, reinterpreted as
    // R32Uint (4-byte tight pack, depth lives in the per-tile depth palette).
    let raw_palette = BufReader::new(zip.by_name("palette.png")?)
        .bytes()
        .collect::<Result<Vec<_>, _>>()?;
    let mut reader = image::ImageReader::new(Cursor::new(raw_palette));
    reader.set_format(image::ImageFormat::Png);
    reader.no_limits();
    let palette_bytes = reader.decode()?.into_bytes();
    let (palette_format, palette_byte_size) = if variant == "idx10s" || variant.starts_with("v3") {
        (TextureFormat::R32Uint, 4u32)
    } else {
        (TextureFormat::Rg32Uint, 8u32)
    };
    let mut pixels_image = Image::new(
        Extent3d {
            width: palette_x,
            height: palette_y,
            depth_or_array_layers: 1,
        },
        wgpu::TextureDimension::D2,
        palette_bytes,
        palette_format,
        load_settings.asset_usages,
    );
    pixels_image.transfer_priority = load_settings.transfer_priority;
    let pixels_image = load_context.add_labeled_asset("palette".to_owned(), pixels_image);

    let total_size: UVec2 = packed_tile_size * grid_size;
    let palette_len = palette_x * palette_y;

    let mut variant_flags = INDEXED_V2_FLAG;
    let (indices_image, indices_hi_image, depth_palette_image, indices_bytes_count) = match variant
    {
        "idx8" => {
            let raw = BufReader::new(zip.by_name("idx.png")?)
                .bytes()
                .collect::<Result<Vec<_>, _>>()?;
            let mut r = image::ImageReader::new(Cursor::new(raw));
            r.set_format(image::ImageFormat::Png);
            r.no_limits();
            let bytes = r.decode()?.into_bytes();
            let mut img = Image::new(
                Extent3d {
                    width: total_size.x,
                    height: total_size.y,
                    depth_or_array_layers: 1,
                },
                wgpu::TextureDimension::D2,
                bytes,
                TextureFormat::R8Uint,
                load_settings.asset_usages,
            );
            img.transfer_priority = load_settings.transfer_priority;
            let idx = load_context.add_labeled_asset("idx".to_owned(), img);
            let dummy_hi = make_dummy_r8_image(load_context, "idx8_dummy_hi");
            let dummy_depth = make_dummy_r8_image(load_context, "idx8_dummy_depth");
            let count = total_size.x * total_size.y;
            (idx, dummy_hi, dummy_depth, count)
        }
        "idx12" => {
            variant_flags |= V2_HAS_HIGH_NIBBLE_FLAG;

            let raw_lo = BufReader::new(zip.by_name("idx_lo.png")?)
                .bytes()
                .collect::<Result<Vec<_>, _>>()?;
            let mut r_lo = image::ImageReader::new(Cursor::new(raw_lo));
            r_lo.set_format(image::ImageFormat::Png);
            r_lo.no_limits();
            let lo_bytes = r_lo.decode()?.into_bytes();
            let mut lo_img = Image::new(
                Extent3d {
                    width: total_size.x,
                    height: total_size.y,
                    depth_or_array_layers: 1,
                },
                wgpu::TextureDimension::D2,
                lo_bytes,
                TextureFormat::R8Uint,
                load_settings.asset_usages,
            );
            lo_img.transfer_priority = load_settings.transfer_priority;
            let lo = load_context.add_labeled_asset("idx_lo".to_owned(), lo_img);

            let hi_w = total_size.x.div_ceil(2);
            let raw_hi = BufReader::new(zip.by_name("idx_hi.png")?)
                .bytes()
                .collect::<Result<Vec<_>, _>>()?;
            let mut r_hi = image::ImageReader::new(Cursor::new(raw_hi));
            r_hi.set_format(image::ImageFormat::Png);
            r_hi.no_limits();
            let hi_bytes = r_hi.decode()?.into_bytes();
            let mut hi_img = Image::new(
                Extent3d {
                    width: hi_w,
                    height: total_size.y,
                    depth_or_array_layers: 1,
                },
                wgpu::TextureDimension::D2,
                hi_bytes,
                TextureFormat::R8Uint,
                load_settings.asset_usages,
            );
            hi_img.transfer_priority = load_settings.transfer_priority;
            let hi = load_context.add_labeled_asset("idx_hi".to_owned(), hi_img);

            let dummy_depth = make_dummy_r8_image(load_context, "idx12_dummy_depth");
            let count = total_size.x * total_size.y + hi_w * total_size.y;
            (lo, hi, dummy_depth, count)
        }
        "idx14" => {
            variant_flags |= V2_HAS_HIGH_BYTE_FLAG;

            let raw_lo = BufReader::new(zip.by_name("idx_lo.png")?)
                .bytes()
                .collect::<Result<Vec<_>, _>>()?;
            let mut r_lo = image::ImageReader::new(Cursor::new(raw_lo));
            r_lo.set_format(image::ImageFormat::Png);
            r_lo.no_limits();
            let lo_bytes = r_lo.decode()?.into_bytes();
            let mut lo_img = Image::new(
                Extent3d {
                    width: total_size.x,
                    height: total_size.y,
                    depth_or_array_layers: 1,
                },
                wgpu::TextureDimension::D2,
                lo_bytes,
                TextureFormat::R8Uint,
                load_settings.asset_usages,
            );
            lo_img.transfer_priority = load_settings.transfer_priority;
            let lo = load_context.add_labeled_asset("idx_lo".to_owned(), lo_img);

            // idx14 high byte is full-width (1 byte per pixel, low 6 bits
            // carry bits 8-13 of the 14-bit index).
            let raw_hi = BufReader::new(zip.by_name("idx_hi.png")?)
                .bytes()
                .collect::<Result<Vec<_>, _>>()?;
            let mut r_hi = image::ImageReader::new(Cursor::new(raw_hi));
            r_hi.set_format(image::ImageFormat::Png);
            r_hi.no_limits();
            let hi_bytes = r_hi.decode()?.into_bytes();
            let mut hi_img = Image::new(
                Extent3d {
                    width: total_size.x,
                    height: total_size.y,
                    depth_or_array_layers: 1,
                },
                wgpu::TextureDimension::D2,
                hi_bytes,
                TextureFormat::R8Uint,
                load_settings.asset_usages,
            );
            hi_img.transfer_priority = load_settings.transfer_priority;
            let hi = load_context.add_labeled_asset("idx_hi".to_owned(), hi_img);

            let dummy_depth = make_dummy_r8_image(load_context, "idx14_dummy_depth");
            let count = total_size.x * total_size.y * 2;
            (lo, hi, dummy_depth, count)
        }
        "idx10s" => {
            variant_flags |= V2_IDX10S_FLAG;

            // idx_lo: full-width Luma8 (bits 0-7 per pixel).
            let raw_lo = BufReader::new(zip.by_name("idx_lo.png")?)
                .bytes()
                .collect::<Result<Vec<_>, _>>()?;
            let mut r_lo = image::ImageReader::new(Cursor::new(raw_lo));
            r_lo.set_format(image::ImageFormat::Png);
            r_lo.no_limits();
            let lo_bytes = r_lo.decode()?.into_bytes();
            let mut lo_img = Image::new(
                Extent3d {
                    width: total_size.x,
                    height: total_size.y,
                    depth_or_array_layers: 1,
                },
                wgpu::TextureDimension::D2,
                lo_bytes,
                TextureFormat::R8Uint,
                load_settings.asset_usages,
            );
            lo_img.transfer_priority = load_settings.transfer_priority;
            let lo = load_context.add_labeled_asset("idx_lo".to_owned(), lo_img);

            // idx_hi: quarter-width Luma8 (bits 8-9 of 4 pixels per byte —
            // bits 0-1 = pix0, bits 2-3 = pix1, bits 4-5 = pix2, bits 6-7 = pix3).
            let hi_w = total_size.x.div_ceil(4);
            let raw_hi = BufReader::new(zip.by_name("idx_hi.png")?)
                .bytes()
                .collect::<Result<Vec<_>, _>>()?;
            let mut r_hi = image::ImageReader::new(Cursor::new(raw_hi));
            r_hi.set_format(image::ImageFormat::Png);
            r_hi.no_limits();
            let hi_bytes = r_hi.decode()?.into_bytes();
            let mut hi_img = Image::new(
                Extent3d {
                    width: hi_w,
                    height: total_size.y,
                    depth_or_array_layers: 1,
                },
                wgpu::TextureDimension::D2,
                hi_bytes,
                TextureFormat::R8Uint,
                load_settings.asset_usages,
            );
            hi_img.transfer_priority = load_settings.transfer_priority;
            let hi = load_context.add_labeled_asset("idx_hi".to_owned(), hi_img);

            // depth_palette: Luma8, palette_len/2 wide, grid_size² tall.
            // Each byte packs two 4-bit depths (low nibble = even slot,
            // high nibble = odd slot).
            let raw_dp = BufReader::new(zip.by_name("depth_palette.png")?)
                .bytes()
                .collect::<Result<Vec<_>, _>>()?;
            let mut r_dp = image::ImageReader::new(Cursor::new(raw_dp));
            r_dp.set_format(image::ImageFormat::Png);
            r_dp.no_limits();
            let dp_bytes = r_dp.decode()?.into_bytes();
            let tile_count = grid_size * grid_size;
            let dp_width = palette_len / 2;
            let mut dp_img = Image::new(
                Extent3d {
                    width: dp_width,
                    height: tile_count,
                    depth_or_array_layers: 1,
                },
                wgpu::TextureDimension::D2,
                dp_bytes,
                TextureFormat::R8Uint,
                load_settings.asset_usages,
            );
            dp_img.transfer_priority = load_settings.transfer_priority;
            let dp = load_context.add_labeled_asset("depth_palette".to_owned(), dp_img);

            let count = total_size.x * total_size.y + hi_w * total_size.y + dp_width * tile_count;
            (lo, hi, dp, count)
        }
        "v3idx8" | "v3idx12" => {
            variant_flags |= V2_V3_FLAG;
            let is12 = variant == "v3idx12";
            if is12 {
                variant_flags |= V2_HAS_HIGH_NIBBLE_FLAG;
            }

            // Low/8-bit index plane: idx.png (8-bit) or idx_lo.png (12-bit),
            // both full-width R8Uint.
            let lo_name = if is12 { "idx_lo.png" } else { "idx.png" };
            let raw_lo = BufReader::new(zip.by_name(lo_name)?)
                .bytes()
                .collect::<Result<Vec<_>, _>>()?;
            let mut r_lo = image::ImageReader::new(Cursor::new(raw_lo));
            r_lo.set_format(image::ImageFormat::Png);
            r_lo.no_limits();
            let lo_bytes = r_lo.decode()?.into_bytes();
            let mut lo_img = Image::new(
                Extent3d {
                    width: total_size.x,
                    height: total_size.y,
                    depth_or_array_layers: 1,
                },
                wgpu::TextureDimension::D2,
                lo_bytes,
                TextureFormat::R8Uint,
                load_settings.asset_usages,
            );
            lo_img.transfer_priority = load_settings.transfer_priority;
            let lo = load_context.add_labeled_asset("idx_lo".to_owned(), lo_img);

            // High nibble (12-bit only): half-width, two 4-bit nibbles per byte.
            let hi_w = total_size.x.div_ceil(2);
            let hi = if is12 {
                let raw_hi = BufReader::new(zip.by_name("idx_hi.png")?)
                    .bytes()
                    .collect::<Result<Vec<_>, _>>()?;
                let mut r_hi = image::ImageReader::new(Cursor::new(raw_hi));
                r_hi.set_format(image::ImageFormat::Png);
                r_hi.no_limits();
                let hi_bytes = r_hi.decode()?.into_bytes();
                let mut hi_img = Image::new(
                    Extent3d {
                        width: hi_w,
                        height: total_size.y,
                        depth_or_array_layers: 1,
                    },
                    wgpu::TextureDimension::D2,
                    hi_bytes,
                    TextureFormat::R8Uint,
                    load_settings.asset_usages,
                );
                hi_img.transfer_priority = load_settings.transfer_priority;
                load_context.add_labeled_asset("idx_hi".to_owned(), hi_img)
            } else {
                make_dummy_r8_image(load_context, "v3idx8_dummy_hi")
            };

            // depth.png — half-width R8Uint, 4-bit per pixel (two per byte).
            // Bound on the depth-palette slot; the shader reads it per-pixel.
            let d_w = total_size.x.div_ceil(2);
            let raw_d = BufReader::new(zip.by_name("depth.png")?)
                .bytes()
                .collect::<Result<Vec<_>, _>>()?;
            let mut r_d = image::ImageReader::new(Cursor::new(raw_d));
            r_d.set_format(image::ImageFormat::Png);
            r_d.no_limits();
            let d_bytes = r_d.decode()?.into_bytes();
            let mut d_img = Image::new(
                Extent3d {
                    width: d_w,
                    height: total_size.y,
                    depth_or_array_layers: 1,
                },
                wgpu::TextureDimension::D2,
                d_bytes,
                TextureFormat::R8Uint,
                load_settings.asset_usages,
            );
            d_img.transfer_priority = load_settings.transfer_priority;
            let depth = load_context.add_labeled_asset("depth".to_owned(), d_img);

            let hi_count = if is12 { hi_w * total_size.y } else { 0 };
            let count = total_size.x * total_size.y + hi_count + d_w * total_size.y;
            (lo, hi, depth, count)
        }
        other => anyhow::bail!("unknown v2 variant `{other}`"),
    };

    let flags = match load_settings.multisample {
        true => RENDER_MULTISAMPLE_FLAG,
        false => 0,
    } + match mode {
        "spherical" => GridMode::Spherical,
        "hemispherical" => GridMode::Hemispherical,
        "Horizontal" => GridMode::Horizontal,
        _ => anyhow::bail!("bad mode `{}`", mode),
    }
    .as_flags()
        + variant_flags;

    let alpha_mode = if load_settings.alpha_blend == 0.0 {
        AlphaMode::Blend
    } else if load_settings.alpha_blend == 1.0 {
        AlphaMode::Opaque
    } else {
        AlphaMode::Mask(load_settings.alpha_blend)
    };

    let vram_bytes = (palette_x * palette_y * palette_byte_size + indices_bytes_count) as usize;

    Ok(Imposter {
        data: ImposterData {
            center_and_scale: Vec3::ZERO.extend(scale),
            grid_size,
            flags,
            alpha: load_settings.alpha,
            base_tile_size,
            packed_tile_offset,
            packed_tile_size,
            multisample_amount: (1.0 - load_settings.multisample_amount).clamp(0.0, 0.99),
        },
        pixels: pixels_image,
        indices: indices_image,
        indices_hi: indices_hi_image,
        depth_palette: depth_palette_image,
        alpha_mode,
        vram_bytes,
    })
}

pub fn pack_asset(grid_size: usize, image: &Image) -> (Image, UVec2, UVec2) {
    let width = image.width() as usize;
    let pixels_per_tile = width / grid_size;
    let mut used_x = std::iter::repeat_n(false, pixels_per_tile).collect::<Vec<_>>();
    let mut used_y = std::iter::repeat_n(false, pixels_per_tile).collect::<Vec<_>>();

    let data: &[u32] = bytemuck::cast_slice(image.data.as_ref().unwrap());

    for grid_x in 0..grid_size {
        for grid_y in 0..grid_size {
            for (pix_x, used_x) in used_x.iter_mut().enumerate() {
                for (pix_y, used_y) in used_y.iter_mut().enumerate() {
                    let y = grid_y * pixels_per_tile + pix_y;
                    let x = grid_x * pixels_per_tile + pix_x;
                    if data[(y * width + x) * 2] != 0 {
                        *used_x = true;
                        *used_y = true;
                    }
                }
            }
        }
    }

    let x_start = used_x
        .iter()
        .enumerate()
        .find(|(_, b)| **b)
        .unwrap_or((0, &true))
        .0;
    let x_end = used_x
        .iter()
        .enumerate()
        .rev()
        .find(|(_, b)| **b)
        .unwrap_or((0, &true))
        .0;
    let y_start = used_y
        .iter()
        .enumerate()
        .find(|(_, b)| **b)
        .unwrap_or((0, &true))
        .0;
    let y_end = used_y
        .iter()
        .enumerate()
        .rev()
        .find(|(_, b)| **b)
        .unwrap_or((0, &true))
        .0;
    let x_count = x_end - x_start + 1;
    let y_count = y_end - y_start + 1;
    let new_width = x_count * grid_size;
    let x_ratio = x_count as f32 / pixels_per_tile as f32;
    let y_ratio = y_count as f32 / pixels_per_tile as f32;
    let total_ratio = x_ratio * y_ratio;
    debug!("ratio: {total_ratio} ({x_ratio} * {y_ratio})");
    if total_ratio == 0.0 {
        std::process::exit(1);
    }

    let mut new_data = Vec::from_iter(std::iter::repeat_n(
        0u32,
        x_count * y_count * 2 * grid_size * grid_size,
    ));
    for grid_y in 0..grid_size {
        for grid_x in 0..grid_size {
            for pix_y in 0..y_count {
                let source_x = grid_x * pixels_per_tile + x_start;
                let source_y = grid_y * pixels_per_tile + y_start + pix_y;
                let target_x = grid_x * x_count;
                let target_y = grid_y * y_count + pix_y;

                new_data[(target_y * new_width + target_x) * 2
                    ..(target_y * new_width + target_x + x_count) * 2]
                    .copy_from_slice(
                        &data[(source_y * width + source_x) * 2
                            ..(source_y * width + source_x + x_count) * 2],
                    );
            }
        }
    }

    let new_data_u8 = new_data
        .into_iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();

    let new_image = Image::new(
        Extent3d {
            width: new_width as u32,
            height: (y_count * grid_size) as u32,
            depth_or_array_layers: 1,
        },
        wgpu::TextureDimension::D2,
        new_data_u8,
        wgpu::TextureFormat::Rg32Uint,
        Default::default(),
    );
    (
        new_image,
        UVec2::new(x_start as u32, y_start as u32),
        UVec2::new(x_count as u32, y_count as u32),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn write_asset(
    path: &PathBuf,
    scale: f32,
    grid_size: u32,
    tile_size: u32,
    mode: GridMode,
    image: Image,
    pack: bool,
    index: bool,
) -> Result<(), anyhow::Error> {
    std::fs::create_dir_all(path.parent().unwrap())?;
    let file = std::fs::File::create(path)?;
    let mut zip = zip::ZipWriter::new(file);
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);

    //trim blank edges
    let (image, packed_offset, packed_size) = if pack {
        pack_asset(grid_size as usize, &image)
    } else {
        (image, UVec2::ZERO, UVec2::splat(tile_size))
    };

    let mut wrote_indexed = false;
    if index {
        // gather unique pixel pairs
        let mut pixels = BTreeSet::<[u8; 8]>::default();
        for chunk in image.data.as_ref().unwrap().chunks_exact(8) {
            pixels.insert(chunk.try_into().unwrap());
        }

        let pixels_x = (pixels.len() as f32).sqrt().ceil() as u32;
        let pixels_y = (pixels.len() as f32 / pixels_x as f32).ceil() as u32;

        let unique_pixel_count = pixels_x * pixels_y;
        let use_u16 = unique_pixel_count < 65536;

        let base_pixel_count = image.width() * image.height();
        let total_index_size_bytes =
            unique_pixel_count * 8 + base_pixel_count * if use_u16 { 2 } else { 4 };
        let base_size = base_pixel_count * 8;
        if total_index_size_bytes < base_size {
            wrote_indexed = true;

            // write unique pixels to an image
            let mut pixel_data = pixels.iter().copied().flatten().collect::<Vec<_>>();
            // pad to square
            pixel_data.extend(std::iter::repeat_n(
                0u8,
                ((pixels_x * pixels_y * 8) as usize).saturating_sub(pixel_data.len()),
            ));
            let pixels_image = Image::new(
                Extent3d {
                    width: pixels_x,
                    height: pixels_y,
                    depth_or_array_layers: 1,
                },
                wgpu::TextureDimension::D2,
                pixel_data,
                TextureFormat::Rg32Uint,
                Default::default(),
            );

            // write pixels to zip
            let dyn_image = DynamicImage::ImageRgba8(
                ImageBuffer::from_raw(
                    pixels_image.width() * 2,
                    pixels_image.height(),
                    pixels_image.data.unwrap(),
                )
                .unwrap(),
            );
            let mut cursor = Cursor::new(Vec::default());
            dyn_image
                .write_to(&mut cursor, image::ImageFormat::Png)
                .unwrap();
            zip.start_file("pixels.png", options)?;
            zip.write_all(&cursor.into_inner())?;

            // write indices to another image
            debug!(
                "use u16? {}*{}={} < 65536 - {}",
                pixels_x,
                pixels_y,
                pixels_x * pixels_y,
                use_u16
            );
            let pixel_lookup = pixels
                .into_iter()
                .enumerate()
                .map(|(ix, p)| (p, ix))
                .collect::<BTreeMap<_, _>>();
            let mut pixel_indices = image
                .data
                .as_ref()
                .unwrap()
                .chunks_exact(8)
                .flat_map(|chunk| {
                    let chunk: [u8; 8] = chunk.try_into().unwrap();
                    let index = *pixel_lookup.get(&chunk).unwrap();
                    if use_u16 {
                        (index as u16).to_le_bytes().to_vec()
                    } else {
                        (index as u32).to_le_bytes().to_vec()
                    }
                })
                .collect::<Vec<_>>();

            let width = if use_u16 {
                if image.width() & 1 == 1 {
                    // pad each line to u32 boundary
                    for i in 0..image.height() {
                        pixel_indices.insert(
                            (image.width() * 2 + i * (image.width() * 2 + 2)) as usize,
                            0,
                        );
                        pixel_indices.insert(
                            (image.width() * 2 + i * (image.width() * 2 + 2)) as usize,
                            0,
                        );
                    }
                    image.width() / 2 + 1
                } else {
                    image.width() / 2
                }
            } else {
                image.width()
            };
            let indices_image = Image::new(
                Extent3d {
                    width,
                    height: image.height(),
                    depth_or_array_layers: 1,
                },
                wgpu::TextureDimension::D2,
                pixel_indices,
                TextureFormat::R32Uint,
                Default::default(),
            );

            // write indices to zip
            let dyn_image = DynamicImage::ImageRgba8(
                ImageBuffer::from_raw(
                    indices_image.width(),
                    indices_image.height(),
                    indices_image.data.unwrap(),
                )
                .unwrap(),
            );
            let mut cursor = Cursor::new(Vec::default());
            dyn_image
                .write_to(&mut cursor, image::ImageFormat::Png)
                .unwrap();
            zip.start_file("indices.png", options)?;
            zip.write_all(&cursor.into_inner())?;
        }
    }

    if !wrote_indexed {
        // write image directly
        let dyn_image = DynamicImage::ImageRgba8(
            ImageBuffer::from_raw(image.width() * 2, image.height(), image.data.unwrap()).unwrap(),
        );
        let mut cursor = Cursor::new(Vec::default());
        dyn_image
            .write_to(&mut cursor, image::ImageFormat::Png)
            .unwrap();
        zip.start_file("texture.png", options)?;
        zip.write_all(&cursor.into_inner())?;
    }

    // write settings
    zip.start_file("settings.txt", options)?;
    let mode = match mode {
        GridMode::Spherical => "spherical",
        GridMode::Hemispherical => "hemispherical",
        GridMode::Horizontal => "Horizontal",
    };
    zip.write_all(
        format!(
            "{grid_size} {scale} {mode} {tile_size} {} {} {} {}",
            packed_offset.x, packed_offset.y, packed_size.x, packed_size.y
        )
        .as_bytes(),
    )?;
    zip.finish()?;
    info!("saved imposter to `{}`", path.to_string_lossy());
    Ok(())
}

/// Imposter writer (v2 format). Quantises the bake-output pixel pairs into a
/// fixed-cap palette and writes one of three variants based on which
/// smallest layout meets `rgb_rmse_threshold`:
///   - idx8 — ≤256 entries, 1 B/pixel (full-size index byte)
///   - idx12 — ≤4096 entries, 1.5 B/pixel (low byte + half-width packed nibble)
///   - idx14 — ≤16384 entries, 2 B/pixel (low byte + full-size high byte)
///
/// On-disk file set:
/// - `settings.txt` — legacy 8 fields + ` v2 {variant} {palette_count} {palette_x} {palette_y}`
/// - `palette.png` — RGBA8 of `palette_x*2 × palette_y`; each Rg32Uint entry is two adjacent RGBA8 pixels
/// - `idx.png` — (idx8 only) Luma8 `total_w × total_h`
/// - `idx_lo.png` — (idx12 / idx14) Luma8 `total_w × total_h`, bits 0-7
/// - `idx_hi.png` — (idx12) Luma8 `ceil(total_w/2) × total_h`, bits 8-11 packed two pixels per byte (low nibble for the even-x pixel, high nibble for the odd)
/// - `idx_hi.png` — (idx14) Luma8 `total_w × total_h`, bits 8-13 in the low 6 bits per pixel
///
/// Old decoders fail file-not-found on the new filenames; that's the
/// designed-in coexistence story.
#[allow(clippy::too_many_arguments)]
pub fn write_asset_v2(
    path: &PathBuf,
    scale: f32,
    grid_size: u32,
    tile_size: u32,
    mode: GridMode,
    image: Image,
    shrink: bool,
    rgb_rmse_threshold: f32,
) -> Result<(), anyhow::Error> {
    use std::io::Write;

    let (image, packed_offset, packed_size) = if shrink {
        crate::asset_loader::pack_asset(grid_size as usize, &image)
    } else {
        (image, UVec2::ZERO, UVec2::splat(tile_size))
    };

    let total_w = image.width();
    let total_h = image.height();

    let image_bytes = image
        .data
        .as_ref()
        .ok_or_else(|| anyhow!("bake image has no data"))?;
    let packs: Vec<[u8; 8]> = image_bytes
        .chunks_exact(8)
        .map(|c| c.try_into().unwrap())
        .collect();
    if packs.len() != (total_w as usize) * (total_h as usize) {
        anyhow::bail!(
            "pack count {} doesn't match {}×{}",
            packs.len(),
            total_w,
            total_h
        );
    }

    // V2 write chain: idx8 → idx10s. idx12 and idx14 are no longer written
    // (the idx10s variant subsumes them with better palette utilisation
    // and per-tile depth precision). The loader still handles all four
    // variants for backward compatibility with previously-baked caches.
    //
    // Env override: `BOIMP_V2_NO_IDX10S` falls back to the classic idx8/
    // idx12/idx14 chain — for A/B regression testing only.
    // When idx8 (legacy, mat+norm+depth in 256) isn't accurate enough, emit
    // the v3 format (mat+norm-only palette to <=4096, decoupled per-pixel
    // 4-bit depth). `BOIMP_V2_V3_IDX10S` falls back to the old idx10s emit, and
    // `BOIMP_V2_NO_V3` falls back to the classic idx8/idx12/idx14 chain — both
    // for A/B testing only. The loader still reads all variants.
    let allow_v3 = std::env::var("BOIMP_V2_NO_V3").is_err();
    let q256 = crate::quantize::quantize(&packs, 256);
    if q256.rgb_rmse >= rgb_rmse_threshold && allow_v3 {
        let emit_idx10s = std::env::var("BOIMP_V2_V3_IDX10S").is_ok();
        let writer = if emit_idx10s {
            write_asset_v2_idx10s
        } else {
            write_asset_v2_v3
        };
        return writer(
            path,
            scale,
            grid_size,
            tile_size,
            mode,
            &packs,
            total_w,
            total_h,
            packed_offset,
            packed_size,
        );
    }

    // Legacy chain (only reached under `BOIMP_V2_NO_IDX10S`, or when idx8
    // already fits the threshold).
    let no_idx14 = std::env::var("BOIMP_V2_NO_IDX14").is_ok();
    let (variant, q) = if q256.rgb_rmse < rgb_rmse_threshold {
        ("idx8", q256)
    } else {
        let q4096 = crate::quantize::quantize(&packs, 4096);
        if no_idx14 || q4096.rgb_rmse < rgb_rmse_threshold {
            ("idx12", q4096)
        } else {
            ("idx14", crate::quantize::quantize(&packs, 16384))
        }
    };

    let palette_count = q.palette.len() as u32;
    // Lay the palette out as a 2D Rg32Uint texture roughly square. The
    // shader looks up via `(idx % palette_x, idx / palette_x)`; storage is
    // dimensioned in *Rg32Uint texels*, so the PNG width on disk is
    // `palette_x * 2` (each texel is two RGBA8 pixels).
    let palette_x = ((palette_count as f32).sqrt().ceil() as u32).max(1);
    let palette_y = palette_count.div_ceil(palette_x).max(1);

    std::fs::create_dir_all(path.parent().unwrap())?;
    let file = std::fs::File::create(path)?;
    let mut zip = zip::ZipWriter::new(file);
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);

    // palette.png — pad up to `palette_x * palette_y` entries with zeros.
    {
        let buf_len = (palette_x as usize) * (palette_y as usize) * 8;
        let mut palette_data = vec![0u8; buf_len];
        for (i, entry) in q.palette.iter().enumerate() {
            palette_data[i * 8..(i + 1) * 8].copy_from_slice(entry);
        }
        let dyn_img = DynamicImage::ImageRgba8(
            ImageBuffer::from_raw(palette_x * 2, palette_y, palette_data)
                .ok_or_else(|| anyhow!("palette buffer size mismatch"))?,
        );
        let mut cursor = Cursor::new(Vec::<u8>::new());
        dyn_img.write_to(&mut cursor, image::ImageFormat::Png)?;
        zip.start_file("palette.png", options)?;
        zip.write_all(&cursor.into_inner())?;
    }

    if variant == "idx8" {
        let mut data = vec![0u8; (total_w as usize) * (total_h as usize)];
        for (i, &idx) in q.indices.iter().enumerate() {
            data[i] = idx as u8;
        }
        let dyn_img = DynamicImage::ImageLuma8(
            ImageBuffer::from_raw(total_w, total_h, data)
                .ok_or_else(|| anyhow!("idx8 buffer size mismatch"))?,
        );
        let mut cursor = Cursor::new(Vec::<u8>::new());
        dyn_img.write_to(&mut cursor, image::ImageFormat::Png)?;
        zip.start_file("idx.png", options)?;
        zip.write_all(&cursor.into_inner())?;
    } else {
        // idx_lo.png — bits 0-7 per pixel, full-size Luma8. (idx12 + idx14)
        let mut lo_data = vec![0u8; (total_w as usize) * (total_h as usize)];
        for (i, &idx) in q.indices.iter().enumerate() {
            lo_data[i] = (idx & 0xFF) as u8;
        }
        let dyn_lo = DynamicImage::ImageLuma8(
            ImageBuffer::from_raw(total_w, total_h, lo_data)
                .ok_or_else(|| anyhow!("idx_lo buffer size mismatch"))?,
        );
        let mut cursor = Cursor::new(Vec::<u8>::new());
        dyn_lo.write_to(&mut cursor, image::ImageFormat::Png)?;
        zip.start_file("idx_lo.png", options)?;
        zip.write_all(&cursor.into_inner())?;

        if variant == "idx12" {
            // idx_hi.png — bits 8-11 packed two pixels per byte, half-width
            // (ceil) Luma8. Convention: low nibble = even-x pixel, high nibble =
            // odd-x pixel. The shader reverses this via `(byte >> ((x & 1) * 4))
            // & 0xF`.
            let hi_w = total_w.div_ceil(2);
            let mut hi_data = vec![0u8; (hi_w as usize) * (total_h as usize)];
            for y in 0..total_h as usize {
                for xp in 0..hi_w as usize {
                    let x_a = xp * 2;
                    let x_b = xp * 2 + 1;
                    let a = if x_a < total_w as usize {
                        ((q.indices[y * total_w as usize + x_a] >> 8) & 0xF) as u8
                    } else {
                        0
                    };
                    let b = if x_b < total_w as usize {
                        ((q.indices[y * total_w as usize + x_b] >> 8) & 0xF) as u8
                    } else {
                        0
                    };
                    hi_data[y * hi_w as usize + xp] = a | (b << 4);
                }
            }
            let dyn_hi = DynamicImage::ImageLuma8(
                ImageBuffer::from_raw(hi_w, total_h, hi_data)
                    .ok_or_else(|| anyhow!("idx_hi buffer size mismatch"))?,
            );
            let mut cursor = Cursor::new(Vec::<u8>::new());
            dyn_hi.write_to(&mut cursor, image::ImageFormat::Png)?;
            zip.start_file("idx_hi.png", options)?;
            zip.write_all(&cursor.into_inner())?;
        } else {
            // idx14: idx_hi.png is full-width Luma8 with bits 8-13 in the low
            // 6 bits per pixel (top 2 bits unused).
            let mut hi_data = vec![0u8; (total_w as usize) * (total_h as usize)];
            for (i, &idx) in q.indices.iter().enumerate() {
                hi_data[i] = ((idx >> 8) & 0x3F) as u8;
            }
            let dyn_hi = DynamicImage::ImageLuma8(
                ImageBuffer::from_raw(total_w, total_h, hi_data)
                    .ok_or_else(|| anyhow!("idx_hi buffer size mismatch"))?,
            );
            let mut cursor = Cursor::new(Vec::<u8>::new());
            dyn_hi.write_to(&mut cursor, image::ImageFormat::Png)?;
            zip.start_file("idx_hi.png", options)?;
            zip.write_all(&cursor.into_inner())?;
        }
    }

    let mode_s = match mode {
        GridMode::Spherical => "spherical",
        GridMode::Hemispherical => "hemispherical",
        GridMode::Horizontal => "Horizontal",
    };
    zip.start_file("settings.txt", options)?;
    zip.write_all(
        format!(
            "{grid_size} {scale} {mode_s} {tile_size} {} {} {} {} v2 {variant} {palette_count} {palette_x} {palette_y}",
            packed_offset.x, packed_offset.y, packed_size.x, packed_size.y
        )
        .as_bytes(),
    )?;
    zip.finish()?;
    info!(
        "saved imposter v2 ({variant}, palette={palette_count}, RGB RMSE={:.2}) to `{}`",
        q.rgb_rmse,
        path.to_string_lossy()
    );
    Ok(())
}

/// Emit the idx10s on-disk format: 1024-entry mat+norm-only palette, 10-bit
/// per-pixel index, per-tile depth palette. See `quantize::quantize_10s` for
/// the bake-time algorithm.
///
/// File layout in the zip:
///   - `palette.png`: RGBA8 of `palette_x × palette_y` (= R32Uint texels of
///     the same dims). Each pixel = one tight 32-bit mat+norm palette entry.
///   - `idx_lo.png`: Luma8 `total_w × total_h`, bits 0-7 of the 10-bit index.
///   - `idx_hi.png`: Luma8 `ceil(total_w/4) × total_h`, bits 8-9 packed 4
///     pixels per byte. Convention: bits 0-1 = pixel 0, bits 2-3 = pixel 1,
///     bits 4-5 = pixel 2, bits 6-7 = pixel 3 (column-major within the byte).
///   - `depth_palette.png`: Luma8 `palette_x*palette_y × tile_count`, one
///     u8 depth per (palette_idx, tile_idx_in_grid).
#[allow(clippy::too_many_arguments)]
fn write_asset_v2_idx10s(
    path: &PathBuf,
    scale: f32,
    grid_size: u32,
    tile_size: u32,
    mode: GridMode,
    packs: &[[u8; 8]],
    total_w: u32,
    total_h: u32,
    packed_offset: UVec2,
    packed_size: UVec2,
) -> Result<(), anyhow::Error> {
    use std::io::Write;

    // Compute tile index per pixel. Pixels are row-major within the image;
    // each tile occupies a `packed_size.x × packed_size.y` rectangle, arranged
    // in a `grid_size × grid_size` grid.
    let num_tiles = (grid_size * grid_size) as usize;
    let mut tile_indices: Vec<u16> = Vec::with_capacity(packs.len());
    for y in 0..total_h {
        for x in 0..total_w {
            let tile_x = x / packed_size.x;
            let tile_y = y / packed_size.y;
            tile_indices.push((tile_y * grid_size + tile_x) as u16);
        }
    }

    let q = crate::quantize::quantize_10s(packs, &tile_indices, num_tiles, 1024);
    let palette_count = q.palette.len() as u32;
    // R32Uint texels — one palette entry per texel. Square-ish layout for the
    // shader's `idx → (idx % palette_x, idx / palette_x)` lookup.
    let palette_x = ((palette_count as f32).sqrt().ceil() as u32).max(1);
    let palette_y = palette_count.div_ceil(palette_x).max(1);

    std::fs::create_dir_all(path.parent().unwrap())?;
    let file = std::fs::File::create(path)?;
    let mut zip = zip::ZipWriter::new(file);
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);

    // palette.png — RGBA8 width=palette_x, height=palette_y, each pixel = 1
    // u32 palette entry (4 bytes).
    {
        let buf_len = (palette_x as usize) * (palette_y as usize) * 4;
        let mut palette_data = vec![0u8; buf_len];
        for (i, &entry) in q.palette.iter().enumerate() {
            palette_data[i * 4..(i + 1) * 4].copy_from_slice(&entry.to_le_bytes());
        }
        let dyn_img = DynamicImage::ImageRgba8(
            ImageBuffer::from_raw(palette_x, palette_y, palette_data)
                .ok_or_else(|| anyhow!("palette buffer size mismatch"))?,
        );
        let mut cursor = Cursor::new(Vec::<u8>::new());
        dyn_img.write_to(&mut cursor, image::ImageFormat::Png)?;
        zip.start_file("palette.png", options)?;
        zip.write_all(&cursor.into_inner())?;
    }

    // idx_lo.png — full-width Luma8, low 8 bits of the 10-bit index.
    {
        let mut lo_data = vec![0u8; (total_w as usize) * (total_h as usize)];
        for (i, &idx) in q.indices.iter().enumerate() {
            lo_data[i] = (idx & 0xFF) as u8;
        }
        let dyn_lo = DynamicImage::ImageLuma8(
            ImageBuffer::from_raw(total_w, total_h, lo_data)
                .ok_or_else(|| anyhow!("idx_lo buffer size mismatch"))?,
        );
        let mut cursor = Cursor::new(Vec::<u8>::new());
        dyn_lo.write_to(&mut cursor, image::ImageFormat::Png)?;
        zip.start_file("idx_lo.png", options)?;
        zip.write_all(&cursor.into_inner())?;
    }

    // idx_hi.png — quarter-width Luma8, bits 8-9 packed 4 pixels per byte.
    {
        let hi_w = total_w.div_ceil(4);
        let mut hi_data = vec![0u8; (hi_w as usize) * (total_h as usize)];
        for y in 0..total_h as usize {
            for xp in 0..hi_w as usize {
                let mut byte = 0u8;
                for offset in 0..4u32 {
                    let x = (xp as u32) * 4 + offset;
                    if x < total_w {
                        let hi_bits = (q.indices[y * total_w as usize + x as usize] >> 8) & 0x3;
                        byte |= (hi_bits as u8) << (offset * 2);
                    }
                }
                hi_data[y * hi_w as usize + xp] = byte;
            }
        }
        let dyn_hi = DynamicImage::ImageLuma8(
            ImageBuffer::from_raw(hi_w, total_h, hi_data)
                .ok_or_else(|| anyhow!("idx_hi buffer size mismatch"))?,
        );
        let mut cursor = Cursor::new(Vec::<u8>::new());
        dyn_hi.write_to(&mut cursor, image::ImageFormat::Png)?;
        zip.start_file("idx_hi.png", options)?;
        zip.write_all(&cursor.into_inner())?;
    }

    // depth_palette.png — Luma8 width=palette_len/2, height=num_tiles. Each
    // byte packs two 4-bit depth values: low nibble = even slot, high
    // nibble = odd slot. quantize_10s already emits this layout.
    {
        let palette_len = palette_x * palette_y;
        let dp_width = palette_len / 2;
        let dyn_dp = DynamicImage::ImageLuma8(
            ImageBuffer::from_raw(dp_width, num_tiles as u32, q.depth_palette.clone())
                .ok_or_else(|| anyhow!("depth_palette buffer size mismatch"))?,
        );
        let mut cursor = Cursor::new(Vec::<u8>::new());
        dyn_dp.write_to(&mut cursor, image::ImageFormat::Png)?;
        zip.start_file("depth_palette.png", options)?;
        zip.write_all(&cursor.into_inner())?;
    }

    let mode_s = match mode {
        GridMode::Spherical => "spherical",
        GridMode::Hemispherical => "hemispherical",
        GridMode::Horizontal => "Horizontal",
    };
    zip.start_file("settings.txt", options)?;
    zip.write_all(
        format!(
            "{grid_size} {scale} {mode_s} {tile_size} {} {} {} {} v2 idx10s {palette_count} {palette_x} {palette_y}",
            packed_offset.x, packed_offset.y, packed_size.x, packed_size.y
        )
        .as_bytes(),
    )?;
    zip.finish()?;
    info!(
        "saved imposter v2 (idx10s, palette={palette_count}, RGB RMSE={:.2}) to `{}`",
        q.rgb_rmse,
        path.to_string_lossy()
    );
    Ok(())
}

/// Emit the v3 on-disk format: mat+norm-only palette (merged to <=4096),
/// 8- or 12-bit per-pixel index (chosen by palette size), and a per-pixel
/// 4-bit *direct* depth plane (no per-tile depth palette — depth is uniform
/// across viewing angles). See `quantize::quantize_v3`.
///
/// File layout in the zip:
///   - `palette.png`: RGBA8 `palette_x × palette_y` (= R32Uint texels), one
///     tight 32-bit mat+norm entry per texel.
///   - 8-bit (`palette <= 256`): `idx.png` Luma8 `total_w × total_h`.
///   - 12-bit: `idx_lo.png` (bits 0-7, full width) + `idx_hi.png` (bits 8-11,
///     half-width, two nibbles per byte: low=even-x, high=odd-x).
///   - `depth.png`: Luma8 `ceil(total_w/2) × total_h`, 4-bit depth per pixel
///     packed two per byte (low=even-x, high=odd-x).
#[allow(clippy::too_many_arguments)]
fn write_asset_v2_v3(
    path: &PathBuf,
    scale: f32,
    grid_size: u32,
    tile_size: u32,
    mode: GridMode,
    packs: &[[u8; 8]],
    total_w: u32,
    total_h: u32,
    packed_offset: UVec2,
    packed_size: UVec2,
) -> Result<(), anyhow::Error> {
    use std::io::Write;

    let q = crate::quantize::quantize_v3(packs, 4096);
    let palette_count = q.palette.len() as u32;
    let use_idx8 = palette_count <= 256;
    let variant = if use_idx8 { "v3idx8" } else { "v3idx12" };
    let palette_x = ((palette_count as f32).sqrt().ceil() as u32).max(1);
    let palette_y = palette_count.div_ceil(palette_x).max(1);

    std::fs::create_dir_all(path.parent().unwrap())?;
    let file = std::fs::File::create(path)?;
    let mut zip = zip::ZipWriter::new(file);
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);

    // palette.png — R32Uint (one u32 mat+norm entry per RGBA8 texel).
    {
        let buf_len = (palette_x as usize) * (palette_y as usize) * 4;
        let mut palette_data = vec![0u8; buf_len];
        for (i, &entry) in q.palette.iter().enumerate() {
            palette_data[i * 4..(i + 1) * 4].copy_from_slice(&entry.to_le_bytes());
        }
        let dyn_img = DynamicImage::ImageRgba8(
            ImageBuffer::from_raw(palette_x, palette_y, palette_data)
                .ok_or_else(|| anyhow!("palette buffer size mismatch"))?,
        );
        let mut cursor = Cursor::new(Vec::<u8>::new());
        dyn_img.write_to(&mut cursor, image::ImageFormat::Png)?;
        zip.start_file("palette.png", options)?;
        zip.write_all(&cursor.into_inner())?;
    }

    if use_idx8 {
        let mut data = vec![0u8; (total_w as usize) * (total_h as usize)];
        for (i, &idx) in q.indices.iter().enumerate() {
            data[i] = idx as u8;
        }
        let dyn_img = DynamicImage::ImageLuma8(
            ImageBuffer::from_raw(total_w, total_h, data)
                .ok_or_else(|| anyhow!("idx buffer size mismatch"))?,
        );
        let mut cursor = Cursor::new(Vec::<u8>::new());
        dyn_img.write_to(&mut cursor, image::ImageFormat::Png)?;
        zip.start_file("idx.png", options)?;
        zip.write_all(&cursor.into_inner())?;
    } else {
        let mut lo_data = vec![0u8; (total_w as usize) * (total_h as usize)];
        for (i, &idx) in q.indices.iter().enumerate() {
            lo_data[i] = (idx & 0xFF) as u8;
        }
        let dyn_lo = DynamicImage::ImageLuma8(
            ImageBuffer::from_raw(total_w, total_h, lo_data)
                .ok_or_else(|| anyhow!("idx_lo buffer size mismatch"))?,
        );
        let mut cursor = Cursor::new(Vec::<u8>::new());
        dyn_lo.write_to(&mut cursor, image::ImageFormat::Png)?;
        zip.start_file("idx_lo.png", options)?;
        zip.write_all(&cursor.into_inner())?;

        // idx_hi.png — bits 8-11, half-width, two nibbles per byte.
        let hi_w = total_w.div_ceil(2);
        let mut hi_data = vec![0u8; (hi_w as usize) * (total_h as usize)];
        for y in 0..total_h as usize {
            for xp in 0..hi_w as usize {
                let xa = xp * 2;
                let xb = xp * 2 + 1;
                let a = if xa < total_w as usize {
                    ((q.indices[y * total_w as usize + xa] >> 8) & 0xF) as u8
                } else {
                    0
                };
                let b = if xb < total_w as usize {
                    ((q.indices[y * total_w as usize + xb] >> 8) & 0xF) as u8
                } else {
                    0
                };
                hi_data[y * hi_w as usize + xp] = a | (b << 4);
            }
        }
        let dyn_hi = DynamicImage::ImageLuma8(
            ImageBuffer::from_raw(hi_w, total_h, hi_data)
                .ok_or_else(|| anyhow!("idx_hi buffer size mismatch"))?,
        );
        let mut cursor = Cursor::new(Vec::<u8>::new());
        dyn_hi.write_to(&mut cursor, image::ImageFormat::Png)?;
        zip.start_file("idx_hi.png", options)?;
        zip.write_all(&cursor.into_inner())?;
    }

    // depth.png — half-width Luma8, 4-bit depth per pixel, two per byte.
    {
        let d_w = total_w.div_ceil(2);
        let mut d_data = vec![0u8; (d_w as usize) * (total_h as usize)];
        for y in 0..total_h as usize {
            for xp in 0..d_w as usize {
                let xa = xp * 2;
                let xb = xp * 2 + 1;
                let a = if xa < total_w as usize {
                    q.depths[y * total_w as usize + xa] & 0xF
                } else {
                    0
                };
                let b = if xb < total_w as usize {
                    q.depths[y * total_w as usize + xb] & 0xF
                } else {
                    0
                };
                d_data[y * d_w as usize + xp] = a | (b << 4);
            }
        }
        let dyn_d = DynamicImage::ImageLuma8(
            ImageBuffer::from_raw(d_w, total_h, d_data)
                .ok_or_else(|| anyhow!("depth buffer size mismatch"))?,
        );
        let mut cursor = Cursor::new(Vec::<u8>::new());
        dyn_d.write_to(&mut cursor, image::ImageFormat::Png)?;
        zip.start_file("depth.png", options)?;
        zip.write_all(&cursor.into_inner())?;
    }

    let mode_s = match mode {
        GridMode::Spherical => "spherical",
        GridMode::Hemispherical => "hemispherical",
        GridMode::Horizontal => "Horizontal",
    };
    zip.start_file("settings.txt", options)?;
    zip.write_all(
        format!(
            "{grid_size} {scale} {mode_s} {tile_size} {} {} {} {} v2 {variant} {palette_count} {palette_x} {palette_y}",
            packed_offset.x, packed_offset.y, packed_size.x, packed_size.y
        )
        .as_bytes(),
    )?;
    zip.finish()?;
    info!(
        "saved imposter v2 ({variant}, palette={palette_count}, RGB RMSE={:.2}) to `{}`",
        q.rgb_rmse,
        path.to_string_lossy()
    );
    Ok(())
}
