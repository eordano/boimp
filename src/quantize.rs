//! Median-cut palette quantizer for imposter bake output.
//!
//! Operates on the 8-byte packed material/normal/depth representation that
//! the bake pipeline writes per pixel (see `pack_props` / `pack_pbrinput` in
//! `shaders/shared.wgsl`). Aggregates unique 64-bit packs, runs a weighted
//! median-cut in 9-D (R, G, B, A, roughness, metallic, normal_x, normal_y,
//! depth), and emits a palette of at most `k` entries plus a per-pixel
//! palette index, with a pixel-weighted RGB RMSE so callers can gate format
//! variants on quantization quality.
//!
//! Flags (bits 28-31 of the lower u32 — the unlit/emissive bits) aren't
//! averaged: each bucket records the weighted-mode flag and the centroid
//! pack uses that. Averaging unlit + emissive together would produce a
//! nonsense flag enum, whereas the median-cut tends to split flag-mixed
//! buckets early because the flag dim has high spread when in use.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

pub struct QuantResult {
    /// At most `k` entries. Each entry is an [u8; 8] packed in the same
    /// layout the bake pipeline writes (see `shaders/shared.wgsl`).
    pub palette: Vec<[u8; 8]>,
    /// One palette index per input pixel.
    pub indices: Vec<u16>,
    /// Pixel-weighted RMSE between original and quantized RGB, on the 0-255
    /// scale. Used as the threshold gate for picking idx8 vs idx12.
    pub rgb_rmse: f32,
}

const DIMS: usize = 9;

/// Per-dimension importance weights for the median-cut split-dimension
/// selection. The cut picks the dim with the largest *weighted* range, so
/// these tilt the cut toward dims whose quantisation error has the biggest
/// visual impact.
///
/// Ranked roughly by how forgivable an error is:
///  - Depth: highest. A wrong depth bends the runtime parallax decode
///    into the *wrong texel* (white sampled from a black region, etc).
///  - Normal: shading direction → specular highlights, fresnel, terminator.
///    Subtle drifts are visible on smooth surfaces.
///  - Roughness / metallic: PBR lighting response. Wrong values flip a
///    surface between matte and glossy or between dielectric and metal.
///  - Alpha: visibility gate (alpha=0 is already pre-extracted, so this
///    only matters for semi-transparent fragments).
///  - RGB: most forgivable — a slightly wrong hue at the right place
///    reads as "lighting variation" to the eye.
///
/// Dim order matches the `coords` layout: R, G, B, A, roughness, metallic,
/// normal_x, normal_y, depth.
const DIM_WEIGHTS: [f32; DIMS] = [
    1.0, 1.0, 1.0, // R, G, B — forgivable colour drift
    1.0, // alpha
    0.5, 0.5, // roughness, metallic — PBR response, visible
    1.0, 1.0, // normal x, y — shading direction (5-bit pre-quantised at bake)
    3.0, // depth — spatial sampling error, weighted highest
];

#[derive(Clone)]
struct Point {
    /// 9-D location in normalised [0, 1] coordinates per dimension.
    coords: [f32; DIMS],
    /// Pixel count carrying this unique key.
    count: u32,
    /// Visibility-weighted weight = `count * alpha`. Drives both the
    /// weighted-median split position and the centroid averaging — alpha
    /// is a "volume control" for the importance of every other channel
    /// (RGB / normal / depth / roughness / metallic), so a barely-visible
    /// low-alpha pack shouldn't tug a cluster centroid around like a
    /// fully-opaque pack does.
    weight: f64,
    /// Original unit normal.
    normal: [f32; 3],
    /// Raw flag nibble (bits 28-31 of the lower u32) — kept discrete and
    /// resolved by weighted vote per bucket rather than averaged.
    flags: u32,
}

pub fn quantize(packs: &[[u8; 8]], k: usize) -> QuantResult {
    if packs.is_empty() || k == 0 {
        return QuantResult {
            palette: vec![],
            indices: vec![],
            rgb_rmse: 0.0,
        };
    }

    // 1. Aggregate unique keys with pixel counts. The median-cut works over
    //    the unique-key set; the weight on each point is its pixel count, so
    //    e.g. an empty silhouette pixel with hundreds of thousands of repeats
    //    still gets a single palette slot.
    let mut counts: HashMap<[u8; 8], u32> = HashMap::new();
    for p in packs {
        *counts.entry(*p).or_insert(0) += 1;
    }

    // Pre-extract anything with alpha=0 (silhouette + transparent surface
    // remnants) and pin it to a single reserved palette slot.
    //
    // Two motivations:
    //  - Correctness: the all-zero silhouette pack `[0; 8]` ties with every
    //    non-empty pack on whichever dim they happen to share a 0 on, and
    //    HashMap-iteration tie-breaks aren't deterministic — some non-empty
    //    pixels get bundled into the silhouette bucket and render as
    //    alpha=0 (invisible).
    //  - Budget: alpha=0 fragments render identically regardless of their
    //    RGB / normal / depth (the shader short-circuits on alpha=0 in
    //    parallax + composite + multisample). Letting the median-cut burn
    //    1000+ palette slots subdividing transparent fragments along other
    //    dimensions wastes a huge fraction of the K budget on rendering
    //    equivalents — those slots are better spent on visible content.
    //
    // Bit layout: alpha occupies bits 15-19 of the lower u32 (see
    // `pack_rgba_roughness_metallic_flags` in shaders/shared.wgsl).
    let alpha_zero_keys: Vec<[u8; 8]> = counts
        .keys()
        .filter(|k| {
            let lower = u32::from_le_bytes(k[0..4].try_into().unwrap());
            ((lower >> 15) & 0x1F) == 0
        })
        .copied()
        .collect();
    let has_empty = !alpha_zero_keys.is_empty();
    for k in &alpha_zero_keys {
        counts.remove(k);
    }

    let key_list: Vec<[u8; 8]> = counts.keys().copied().collect();
    let points: Vec<Point> = key_list
        .iter()
        .map(|k| unpack_point(*k, *counts.get(k).unwrap()))
        .collect();
    let key_to_idx: HashMap<[u8; 8], usize> =
        key_list.iter().enumerate().map(|(i, k)| (*k, i)).collect();

    // 2. Recursive median-cut over the unique (non-empty) points. If we
    //    reserved slot 0 for empty, the cut budget shrinks to k-1; bucket
    //    IDs start at 1 so they don't collide with the reserved slot.
    let cut_budget = if has_empty { k.saturating_sub(1) } else { k };
    let mut indices: Vec<usize> = (0..points.len()).collect();
    let mut bucket_of = vec![0usize; points.len()];
    let mut next_id = if has_empty { 1usize } else { 0usize };
    if cut_budget > 0 && !points.is_empty() {
        median_cut(
            &points,
            &mut indices,
            &mut bucket_of,
            &mut next_id,
            cut_budget,
        );
    }
    let palette_used = next_id;

    // 3. Compute weighted centroids per bucket.
    //
    //  - RGB / roughness / metallic / normal / depth are averaged by
    //    visibility weight (`count * alpha`) — barely-visible points don't
    //    drift the centroid.
    //  - The centroid's *alpha* itself is averaged by raw count, so it
    //    reflects "what fraction of the bucket is actually visible" rather
    //    than self-amplifying high-alpha members.
    //  - Flag bits are discrete; resolved by raw-count weighted mode.
    //
    // Slot 0 stays zeroed when `has_empty` — the silhouette pack is encoded
    // directly without centroiding.
    let mut sum_rgb = vec![[0.0f64; 3]; palette_used];
    let mut sum_alpha = vec![0.0f64; palette_used];
    let mut sum_rm = vec![[0.0f64; 2]; palette_used];
    let mut sum_normal = vec![[0.0f64; 3]; palette_used];
    let mut sum_depth = vec![0.0f64; palette_used];
    let mut flag_votes: Vec<HashMap<u32, u64>> = vec![HashMap::new(); palette_used];
    let mut wsum_vis = vec![0.0f64; palette_used];
    let mut wsum_count = vec![0u64; palette_used];
    for (pi, p) in points.iter().enumerate() {
        let b = bucket_of[pi];
        let w = p.weight;
        let c = p.count as f64;
        for (d, s) in sum_rgb[b].iter_mut().enumerate() {
            *s += p.coords[d] as f64 * w;
        }
        sum_alpha[b] += p.coords[3] as f64 * c;
        sum_rm[b][0] += p.coords[4] as f64 * w;
        sum_rm[b][1] += p.coords[5] as f64 * w;
        for (d, s) in sum_normal[b].iter_mut().enumerate() {
            *s += p.normal[d] as f64 * w;
        }
        sum_depth[b] += p.coords[8] as f64 * w;
        *flag_votes[b].entry(p.flags).or_insert(0) += p.count as u64;
        wsum_vis[b] += w;
        wsum_count[b] += p.count as u64;
    }

    let mut palette: Vec<[u8; 8]> = Vec::with_capacity(palette_used);
    for b in 0..palette_used {
        if has_empty && b == 0 {
            palette.push([0u8; 8]);
            continue;
        }
        // If the bucket somehow ended up entirely-transparent (shouldn't
        // happen after pre-extraction, but be defensive), fall back to
        // count-weighted averaging for the visibility-weighted channels.
        let wv = wsum_vis[b].max(1e-9);
        let wc = wsum_count[b].max(1) as f64;
        let rgba = [
            (sum_rgb[b][0] / wv) as f32,
            (sum_rgb[b][1] / wv) as f32,
            (sum_rgb[b][2] / wv) as f32,
            (sum_alpha[b] / wc) as f32,
        ];
        let rough = (sum_rm[b][0] / wv) as f32;
        let metal = (sum_rm[b][1] / wv) as f32;
        let depth = (sum_depth[b] / wv) as f32;
        let n_avg = [
            (sum_normal[b][0] / wv) as f32,
            (sum_normal[b][1] / wv) as f32,
            (sum_normal[b][2] / wv) as f32,
        ];
        let nlen = (n_avg[0] * n_avg[0] + n_avg[1] * n_avg[1] + n_avg[2] * n_avg[2])
            .sqrt()
            .max(1e-9);
        let normal = [n_avg[0] / nlen, n_avg[1] / nlen, n_avg[2] / nlen];
        let flags = flag_votes[b]
            .iter()
            .max_by_key(|(_, c)| **c)
            .map(|(f, _)| *f)
            .unwrap_or(0);
        palette.push(centroid_to_pack(rgba, rough, metal, flags, normal, depth));
    }

    // 4. Per-pixel palette index. Any alpha=0 pack collapses to the
    //    reserved slot 0; everything else looks up its median-cut leaf.
    let pixel_indices: Vec<u16> = packs
        .iter()
        .map(|p| {
            if has_empty && is_alpha_zero(p) {
                0
            } else {
                bucket_of[*key_to_idx.get(p).unwrap()] as u16
            }
        })
        .collect();

    // 5. Visibility-weighted RGB RMSE between each original pixel and the
    //    re-decoded palette entry. Weighting by `alpha` (per the volume-
    //    control reading) means the threshold gate cares about errors on
    //    visible content — a 20-unit RGB drift on an alpha=0.05 fragment
    //    is essentially invisible and shouldn't push us off the cheaper
    //    idx8 variant. Alpha=0 pixels naturally drop out (zero weight).
    let palette_rgb8: Vec<[f32; 3]> = palette.iter().map(|p| unpack_rgb8(*p)).collect();
    let mut rgb_sq_w: f64 = 0.0;
    let mut total_alpha: f64 = 0.0;
    for p in packs {
        let alpha_bits = {
            let lower = u32::from_le_bytes(p[0..4].try_into().unwrap());
            (lower >> 15) & 0x1F
        };
        if alpha_bits == 0 {
            continue;
        }
        let alpha = alpha_bits as f64 / 31.0;
        let pal = if has_empty && is_alpha_zero(p) {
            // unreachable given the alpha_bits == 0 check above, but kept
            // for symmetry with the index-builder above.
            palette_rgb8[0]
        } else {
            let ki = *key_to_idx.get(p).unwrap();
            palette_rgb8[bucket_of[ki]]
        };
        let pt_rgb = unpack_rgb8(*p);
        let dr = (pt_rgb[0] - pal[0]) as f64;
        let dg = (pt_rgb[1] - pal[1]) as f64;
        let db = (pt_rgb[2] - pal[2]) as f64;
        rgb_sq_w += (dr * dr + dg * dg + db * db) * alpha;
        total_alpha += alpha;
    }
    let rgb_rmse = if total_alpha > 0.0 {
        (rgb_sq_w / total_alpha).sqrt() as f32
    } else {
        0.0
    };

    QuantResult {
        palette,
        indices: pixel_indices,
        rgb_rmse,
    }
}

fn unpack_point(key: [u8; 8], count: u32) -> Point {
    let lower = u32::from_le_bytes(key[0..4].try_into().unwrap());
    let upper = u32::from_le_bytes(key[4..8].try_into().unwrap());
    let r = (lower & 0x1F) as f32 / 31.0;
    let g = ((lower >> 5) & 0x1F) as f32 / 31.0;
    let b = ((lower >> 10) & 0x1F) as f32 / 31.0;
    let a = ((lower >> 15) & 0x1F) as f32 / 31.0;
    let rough = ((lower >> 20) & 0xF) as f32 / 15.0;
    let metal = ((lower >> 24) & 0xF) as f32 / 15.0;
    let flags = lower >> 28;
    let nx = (upper & 0xFFF) as f32 / 4095.0;
    let ny = ((upper >> 12) & 0xFFF) as f32 / 4095.0;
    let depth = ((upper >> 24) & 0xFF) as f32 / 255.0;
    let normal = uv_to_normal(nx, ny);
    Point {
        coords: [r, g, b, a, rough, metal, nx, ny, depth],
        count,
        weight: count as f64 * a as f64,
        normal,
        flags,
    }
}

fn is_alpha_zero(pack: &[u8; 8]) -> bool {
    let lower = u32::from_le_bytes(pack[0..4].try_into().unwrap());
    ((lower >> 15) & 0x1F) == 0
}

fn unpack_rgb8(pack: [u8; 8]) -> [f32; 3] {
    let lower = u32::from_le_bytes(pack[0..4].try_into().unwrap());
    let scale = 255.0 / 31.0;
    [
        (lower & 0x1F) as f32 * scale,
        ((lower >> 5) & 0x1F) as f32 * scale,
        ((lower >> 10) & 0x1F) as f32 * scale,
    ]
}

/// Mirrors `spherical_normal_from_uv` in `shaders/shared.wgsl`.
fn uv_to_normal(nx_q: f32, ny_q: f32) -> [f32; 3] {
    let x = nx_q * 2.0 - 1.0;
    let z = ny_q * 2.0 - 1.0;
    let y = 1.0 - x.abs() - z.abs();
    let (nx, ny, nz) = if y < 0.0 {
        (
            x.signum() * (1.0 - z.abs()),
            y,
            z.signum() * (1.0 - x.abs()),
        )
    } else {
        (x, y, z)
    };
    let len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-9);
    [nx / len, ny / len, nz / len]
}

/// Mirrors `spherical_uv_from_normal` in `shaders/shared.wgsl` — inverts
/// `uv_to_normal` so an averaged unit normal can be re-encoded in the
/// 12+12-bit storage format.
fn normal_to_uv(n: [f32; 3]) -> [f32; 2] {
    let sx = if n[0] >= 0.0 { 1.0 } else { -1.0 };
    let sy = if n[1] >= 0.0 { 1.0 } else { -1.0 };
    let sz = if n[2] >= 0.0 { 1.0 } else { -1.0 };
    let sum = n[0] * sx + n[1] * sy + n[2] * sz;
    if sum.abs() < 1e-9 {
        return [0.5, 0.5];
    }
    let oct = [n[0] / sum, n[1] / sum, n[2] / sum];
    let (u, v) = if oct[1] < 0.0 {
        (sx * (1.0 - oct[2].abs()), sz * (1.0 - oct[0].abs()))
    } else {
        (oct[0], oct[2])
    };
    [(u + 1.0) * 0.5, (v + 1.0) * 0.5]
}

fn pack_bits_f(value: f32, count: u32) -> u32 {
    let mask = (1u32 << count) - 1;
    (value.clamp(0.0, 1.0) * mask as f32 + 0.5) as u32
}

fn centroid_to_pack(
    rgba: [f32; 4],
    rough: f32,
    metal: f32,
    flags: u32,
    normal: [f32; 3],
    depth: f32,
) -> [u8; 8] {
    let lower = pack_bits_f(rgba[0], 5)
        | (pack_bits_f(rgba[1], 5) << 5)
        | (pack_bits_f(rgba[2], 5) << 10)
        | (pack_bits_f(rgba[3], 5) << 15)
        | (pack_bits_f(rough, 4) << 20)
        | (pack_bits_f(metal, 4) << 24)
        | ((flags & 0xF) << 28);
    let nuv = normal_to_uv(normal);
    let upper =
        pack_bits_f(nuv[0], 12) | (pack_bits_f(nuv[1], 12) << 12) | (pack_bits_f(depth, 8) << 24);
    let mut out = [0u8; 8];
    out[0..4].copy_from_slice(&lower.to_le_bytes());
    out[4..8].copy_from_slice(&upper.to_le_bytes());
    out
}

/// One bucket pending splitting in the priority-queue median-cut.
///
/// `indices` is sorted by `split_dim` so a subsequent split can simply slice
/// at `split_at` without re-sorting. `split_score` is the weighted spread
/// along `split_dim` — the heap pops the bucket with the largest score
/// first, so budget flows to wherever the spread is biggest regardless of
/// where in the recursion tree it sits.
struct BucketCandidate {
    indices: Vec<usize>,
    split_dim: usize,
    split_score: f32,
    split_at: usize,
}

impl PartialEq for BucketCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.split_score == other.split_score
    }
}
impl Eq for BucketCandidate {}
impl PartialOrd for BucketCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for BucketCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap on split_score; `total_cmp` is NaN-safe (though scores
        // here are computed from finite spreads so NaNs shouldn't occur).
        self.split_score.total_cmp(&other.split_score)
    }
}

/// Find the widest weighted-spread dim, sort `indices` along it, locate the
/// weighted-median split position, and bundle the result into a candidate.
fn evaluate_split(points: &[Point], mut indices: Vec<usize>) -> BucketCandidate {
    let mut best_dim = 0usize;
    let mut best_score = 0.0f32;
    for d in 0..DIMS {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &i in indices.iter() {
            let v = points[i].coords[d];
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        let score = (hi - lo) * DIM_WEIGHTS[d];
        if score > best_score {
            best_score = score;
            best_dim = d;
        }
    }

    if best_score > 0.0 && indices.len() > 1 {
        indices.sort_by(|&a, &b| {
            points[a].coords[best_dim]
                .partial_cmp(&points[b].coords[best_dim])
                .unwrap()
        });
    }

    // Visibility-weighted split (`count * alpha`); falls back to raw count
    // if all weights are zero (alpha-only zero — shouldn't happen after
    // pre-extraction, but defensive).
    let total_w: f64 = indices.iter().map(|&i| points[i].weight).sum();
    let fallback = total_w <= 0.0;
    let total_metric: f64 = if fallback {
        indices.iter().map(|&i| points[i].count as f64).sum()
    } else {
        total_w
    };
    let mut running: f64 = 0.0;
    let mut split_at = indices.len() / 2;
    if indices.len() > 1 {
        for (pos, &i) in indices.iter().enumerate() {
            running += if fallback {
                points[i].count as f64
            } else {
                points[i].weight
            };
            if running * 2.0 >= total_metric {
                split_at = (pos + 1).clamp(1, indices.len() - 1);
                break;
            }
        }
    }

    BucketCandidate {
        indices,
        split_dim: best_dim,
        split_score: best_score,
        split_at,
    }
}

/// Priority-queue median-cut. Maintains a max-heap of unsplit buckets
/// keyed by their largest weighted spread; each step pops the bucket with
/// the worst remaining spread and splits it. Budget naturally flows to
/// wherever the variance is highest, instead of being statically halved
/// down the recursion tree (the old recursive variant could leave a
/// high-variance sub-tree under-budgeted while a low-variance sub-tree
/// terminated early with budget to spare).
fn median_cut(
    points: &[Point],
    indices: &mut [usize],
    bucket_of: &mut [usize],
    next_id: &mut usize,
    k: usize,
) {
    if k == 0 || indices.is_empty() {
        return;
    }
    if k == 1 || indices.len() == 1 {
        let id = *next_id;
        *next_id += 1;
        for &i in indices.iter() {
            bucket_of[i] = id;
        }
        return;
    }

    let initial = evaluate_split(points, indices.to_vec());

    // A bucket is splittable iff it has at least 2 points and at least one
    // dim with non-zero weighted spread. Anything else goes directly to
    // leaves (a 1-point bucket trivially encodes its sole input; a
    // zero-spread bucket would have all-identical points and can't be
    // meaningfully cut).
    let mut heap: BinaryHeap<BucketCandidate> = BinaryHeap::new();
    let mut leaves: Vec<Vec<usize>> = Vec::new();

    if initial.split_score > 0.0 && initial.indices.len() > 1 {
        heap.push(initial);
    } else {
        leaves.push(initial.indices);
    }

    while leaves.len() + heap.len() < k {
        let Some(cand) = heap.pop() else {
            break;
        };
        // cand.indices is sorted by cand.split_dim from the previous
        // evaluate_split, so we can slice cleanly at split_at.
        let right_indices = cand.indices[cand.split_at..].to_vec();
        let mut left_indices = cand.indices;
        left_indices.truncate(cand.split_at);

        for half in [
            evaluate_split(points, left_indices),
            evaluate_split(points, right_indices),
        ] {
            if half.split_score > 0.0 && half.indices.len() > 1 {
                heap.push(half);
            } else {
                leaves.push(half.indices);
            }
        }
    }

    // Any candidates still in the heap when budget runs out become leaves
    // as-is (they're valid buckets with their current spread; we just
    // didn't have room to subdivide them further).
    leaves.extend(heap.into_iter().map(|c| c.indices));

    for bucket_indices in leaves {
        let id = *next_id;
        *next_id += 1;
        for i in bucket_indices {
            bucket_of[i] = id;
        }
    }
}
