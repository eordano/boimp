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

use bevy::log::{debug, info};

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

// =============================================================================
// idx10s — bitmap-driven agglomerative quantiser
// =============================================================================
//
// Variant of the quantizer for the idx10s on-disk format. Pulls depth out of
// the palette entirely and stores it in a per-tile depth palette, addressed by
// the same 10-bit pixel index. The main palette stores only (mat, norm),
// packed tightly as one u32 per entry.
//
// Bake algorithm differs from `quantize`: pixels are grouped by EXACT
// (mat, norm) match, with each group carrying a per-tile depth-bucket bitmap.
// The slot cost of a group = `max over tiles of popcount(bitmap[tile])` — a
// group with one (mat, norm) appearing at N distinct depths in some tile
// costs N palette entries (one per depth slot per tile-bucket-class).
// Greedy agglomerative merge then reduces total slot cost to <= k - 1
// (slot 0 reserved for alpha=0).

/// Depth-bucket bit count used by the per-tile bitmaps. Each pixel's stored
/// depth byte is mapped onto one of these buckets. 64 buckets fits in a u64
/// per tile for fast popcount; the per-tile depth palette stores the exact
/// 8-bit averaged depth per (tile, slot), so the bucket granularity here is
/// purely for the merge-decision bookkeeping.
const DEPTH_BUCKETS_10S: usize = 64;

const DIMS_10S: usize = 8;

/// Per-dimension weights for the merge-cost error metric. Same channel order
/// as `DIMS` minus the depth dim — depth is no longer in the palette so it
/// doesn't participate in the (mat, norm) distance.
///
/// Weights contribute *quadratically* to the error (delta is multiplied by
/// weight, then squared). Heavily-weighted dims become "expensive" to merge
/// across, so the algorithm preserves their diversity at the expense of
/// less-weighted dims.
///
/// Calibration: idx10s' per-tile depth palette gives near-exact runtime
/// stability under motion, so less-perfect normals are forgivable for the
/// sake of preserving colour diversity. RGB is overweighted to 2.0 (4×
/// effective) and normals are underweighted to 0.5 (0.25× effective) —
/// 16× ratio in favour of colour over normal.
const DIM_WEIGHTS_10S: [f32; DIMS_10S] = [
    2.0, 2.0, 2.0, // R, G, B — overweighted to preserve colour diversity
    1.0, // alpha
    1.0, 1.0, // roughness, metallic — PBR response, preserved at parity
    0.5, 0.5, // normal x, y — underweighted (depth-palette stabilises motion)
];

pub struct QuantResult10s {
    /// Up to `k` u32 entries. Tight 32-bit pack:
    ///   bits 0-3:   r           (4 bits)
    ///   bits 4-7:   g           (4 bits)
    ///   bits 8-11:  b           (4 bits)
    ///   bits 12-14: alpha       (3 bits)
    ///   bits 15-16: roughness   (2 bits)
    ///   bits 17-18: metallic    (2 bits)
    ///   bits 19-22: flags       (4 bits)
    ///   bits 23-26: normal_x    (4 bits)
    ///   bits 27-30: normal_y    (4 bits)
    ///   bit  31:    unused
    /// Slot 0 reserved for alpha=0 (zero pack).
    pub palette: Vec<u32>,
    /// One 10-bit palette index per input pixel (low 10 bits used; high bits
    /// always zero).
    pub indices: Vec<u16>,
    /// Per-tile depth palette, packed 2 depths per byte. Layout:
    /// `depth_palette[tile_idx * (k/2) + (slot >> 1)]` holds two 4-bit
    /// depths: the low nibble is the even-`slot`, the high nibble is the
    /// odd-`slot`. Unused (tile, slot) cells are 0. Length = `num_tiles * k/2`.
    ///
    /// 4 bits per depth is exact given the bake's 4-bit depth pre-quant —
    /// each (tile, slot) holds the byte value of one pre-quant level, which
    /// fits losslessly in 4 bits as the level index in [0, 15].
    pub depth_palette: Vec<u8>,
    /// Pixel-weighted RGB RMSE on the 0-255 scale (alpha-weighted, same
    /// convention as `quantize`).
    pub rgb_rmse: f32,
}

#[derive(Clone)]
struct Group10s {
    /// Per-tile depth-bucket bitmap. `bitmap[t]` bit `d` ⇔ at least one pixel
    /// in this group is in tile `t` at depth bucket `d`.
    bitmap: Vec<u64>,
    /// `(pixel_index, depth_bucket, tile_index)` per pixel in this group.
    pixels: Vec<(u32, u8, u16)>,
    /// 8-D weighted centroid in normalised [0, 1] coords. Order matches
    /// `DIM_WEIGHTS_10S`.
    centroid: [f32; DIMS_10S],
    /// Visibility weight: sum of alpha across member pixels. Used to weight
    /// the centroid average when groups merge.
    weight: f64,
    /// Weighted-mode flag value. Pre-merge: identical across all members of
    /// a group (since groups key on exact mat+norm bytes). After a merge
    /// between two flag-disagreeing groups, we keep the heavier source's
    /// flags — flags are a discrete enum, averaging them is nonsense.
    flags: u32,
    /// Total pixel count in this group.
    count: u32,
    /// Cached `max over tiles of popcount(bitmap[t])`. Recomputed on merge.
    /// Avoids an O(num_tiles) recompute on every pairwise cost evaluation.
    cached_slots: u32,
}

impl Group10s {
    fn new(num_tiles: usize) -> Self {
        Self {
            bitmap: vec![0u64; num_tiles],
            pixels: Vec::new(),
            centroid: [0.0; DIMS_10S],
            weight: 0.0,
            flags: 0,
            count: 0,
            cached_slots: 0,
        }
    }

    fn recompute_slots(&mut self) {
        self.cached_slots = self
            .bitmap
            .iter()
            .map(|b| b.count_ones())
            .max()
            .unwrap_or(0);
    }

    fn slot_count(&self) -> u32 {
        self.cached_slots
    }
}

/// Maps a stored 8-bit depth byte onto a 6-bit bucket index in [0, 63].
/// Pre-quant guarantees the stored byte is one of 64 specific values; this
/// is the inverse round-trip.
fn depth_bucket(stored_byte: u32) -> u8 {
    ((stored_byte * 63 + 127) / 255) as u8
}

pub fn quantize_10s(
    packs: &[[u8; 8]],
    tile_indices: &[u16],
    num_tiles: usize,
    k: usize,
) -> QuantResult10s {
    assert_eq!(packs.len(), tile_indices.len());
    let palette_len = k.max(1);

    if packs.is_empty() || k <= 1 {
        return QuantResult10s {
            palette: vec![0; palette_len],
            indices: vec![0; packs.len()],
            depth_palette: vec![0; num_tiles * palette_len.div_ceil(2)],
            rgb_rmse: 0.0,
        };
    }

    // ----- 1. Pre-extract alpha=0 + group remaining by exact (mat, norm) ---
    //
    // Slot 0 is reserved for alpha=0 (matches the convention `quantize` uses,
    // and the runtime's "alpha=0 means discard" short-circuits make the
    // stored depth / material irrelevant for empty pixels).
    //
    // Group key = (lower_u32, upper_u32_without_depth_byte). Pre-quantised
    // data has tight equality between visually-equivalent pixels, so the
    // exact-byte key naturally clusters them.
    let mut groups: HashMap<(u32, u32), Group10s> = HashMap::new();

    for (i, p) in packs.iter().enumerate() {
        if is_alpha_zero(p) {
            continue;
        }
        let lo = u32::from_le_bytes(p[0..4].try_into().unwrap());
        let hi = u32::from_le_bytes(p[4..8].try_into().unwrap());
        let key = (lo, hi & 0x00FF_FFFF);
        let depth = depth_bucket((hi >> 24) & 0xFF);
        let tile_idx = tile_indices[i];

        let group = groups
            .entry(key)
            .or_insert_with(|| Group10s::new(num_tiles));
        group.bitmap[tile_idx as usize] |= 1u64 << depth;
        group.pixels.push((i as u32, depth, tile_idx));

        // Centroid: all pixels in a single group share exact mat+norm bytes,
        // so this just records that value on first insertion. Weight
        // accumulation matters during merging.
        if group.count == 0 {
            let r = (lo & 0x1F) as f32 / 31.0;
            let g = ((lo >> 5) & 0x1F) as f32 / 31.0;
            let b = ((lo >> 10) & 0x1F) as f32 / 31.0;
            let a = ((lo >> 15) & 0x1F) as f32 / 31.0;
            let rough = ((lo >> 20) & 0xF) as f32 / 15.0;
            let metal = ((lo >> 24) & 0xF) as f32 / 15.0;
            let nx = (hi & 0xFFF) as f32 / 4095.0;
            let ny = ((hi >> 12) & 0xFFF) as f32 / 4095.0;
            group.centroid = [r, g, b, a, rough, metal, nx, ny];
            group.flags = lo >> 28;
        }
        let alpha_norm = ((lo >> 15) & 0x1F) as f64 / 31.0;
        group.weight += alpha_norm;
        group.count += 1;
    }

    // ----- 2. Convert to Vec, compute total slot cost ----------------------
    let mut groups: Vec<Group10s> = groups.into_values().collect();
    for g in groups.iter_mut() {
        g.recompute_slots();
    }
    let initial_total_slots: u32 = groups.iter().map(|g| g.slot_count()).sum();
    debug!(
        "quantize_10s: {} groups after enumeration, {} initial slots, target {}",
        groups.len(),
        initial_total_slots,
        k.saturating_sub(1),
    );

    // ----- 3. Agglomerative merge until total slots <= k - 1 ---------------
    //
    // Budget is `k - 1` because slot 0 is reserved for alpha=0.
    let target_slots = (k.saturating_sub(1)) as u32;
    if initial_total_slots > target_slots {
        agglomerative_merge_10s(&mut groups, target_slots);
    }

    // ----- 4. Allocate palette slots ---------------------------------------
    //
    // Slot 0 is alpha=0 (zero pack). Subsequent slots are allocated to groups
    // contiguously by group.slot_count(). Each group writes the same (mat,
    // norm) into all its slots — different slots only differ in their
    // per-tile depth palette values.
    // palette_len is the k cap (1024 typical) and must be even — the depth
    // palette packs two 4-bit entries per byte, so an odd palette_len would
    // leave a half-byte at the tail of each tile row.
    assert!(
        palette_len.is_multiple_of(2),
        "palette_len must be even for the 2-per-byte depth pack"
    );
    let depth_palette_stride = palette_len / 2;
    let mut palette: Vec<u32> = vec![0; palette_len];
    let mut depth_palette: Vec<u8> = vec![0; num_tiles * depth_palette_stride];
    let mut indices: Vec<u16> = vec![0; packs.len()];

    let mut next_slot: u32 = 1;
    for group in &groups {
        let g_slots = group.slot_count();
        if g_slots == 0 {
            // Empty group (got fully consumed by merges into others); skip.
            continue;
        }
        if next_slot as usize + g_slots as usize > palette_len {
            // Shouldn't happen if merge converged; defensive fallback would
            // be to lossily merge remaining over-budget pixels into the
            // last slot. Trigger an assert here to catch the bug if it ever
            // ships.
            panic!(
                "quantize_10s: palette overflow during layout (next={}, group_slots={}, k={})",
                next_slot, g_slots, palette_len
            );
        }

        let group_pack = pack_palette_entry_10s(group.centroid, group.flags);
        for s in 0..g_slots {
            palette[(next_slot + s) as usize] = group_pack;
        }

        // Per-tile slot-offset table: for each tile, map depth_bucket → slot
        // offset within this group's allocation. Iterate set bits of the
        // tile's bitmap to assign offsets in sorted-by-bucket order.
        //
        // Also compute per-(tile, slot) average depth from the group's pixel
        // list; that's what the runtime samples from the depth palette.
        let mut bucket_to_offset = vec![[u8::MAX; DEPTH_BUCKETS_10S]; num_tiles];
        let mut depth_sum = vec![[0u32; DEPTH_BUCKETS_10S]; num_tiles];
        let mut depth_count = vec![[0u32; DEPTH_BUCKETS_10S]; num_tiles];
        for t in 0..num_tiles {
            let mut bits = group.bitmap[t];
            let mut offset = 0u8;
            while bits != 0 {
                let lsb = bits.trailing_zeros() as usize;
                bucket_to_offset[t][lsb] = offset;
                offset += 1;
                bits &= bits - 1;
            }
        }
        // Sum stored-byte depths per (tile, bucket); we'll average and write.
        for &(pix_idx, bucket, tile_idx) in &group.pixels {
            let p = &packs[pix_idx as usize];
            let stored = (u32::from_le_bytes(p[4..8].try_into().unwrap()) >> 24) & 0xFF;
            depth_sum[tile_idx as usize][bucket as usize] += stored;
            depth_count[tile_idx as usize][bucket as usize] += 1;
        }
        for t in 0..num_tiles {
            for b in 0..DEPTH_BUCKETS_10S {
                let off = bucket_to_offset[t][b];
                if off == u8::MAX {
                    continue;
                }
                let count = depth_count[t][b];
                if count == 0 {
                    continue;
                }
                let avg_byte = (depth_sum[t][b] / count) as u32;
                // Convert 8-bit averaged byte → 4-bit pre-quant level. With
                // 4-bit depth pre-quant this round-trips exactly (each
                // average corresponds to a single pre-quant level since the
                // bucket mapping in `depth_bucket` is 1-to-1 with input
                // levels). Even with future coarser bucketing the loss is
                // ≤ ±1 in the 16-level depth space.
                let nibble = ((avg_byte * 15 + 127) / 255) as u8 & 0xF;
                let slot = next_slot as usize + off as usize;
                let byte_idx = t * depth_palette_stride + (slot >> 1);
                let shift = (slot & 1) * 4;
                depth_palette[byte_idx] |= nibble << shift;
            }
        }

        // Emit per-pixel indices via the bucket→offset table.
        for &(pix_idx, bucket, tile_idx) in &group.pixels {
            let off = bucket_to_offset[tile_idx as usize][bucket as usize];
            debug_assert_ne!(off, u8::MAX);
            indices[pix_idx as usize] = next_slot as u16 + off as u16;
        }

        next_slot += g_slots;
    }

    // alpha=0 pixels stay at index 0 (default).

    // ----- 5. RGB RMSE for the quality gate --------------------------------
    //
    // Mirrors `quantize`'s implementation but reads RGB from the 32-bit pack
    // format (4 bits per channel). Visibility-weighted by per-pixel alpha so
    // mostly-transparent pixels don't dominate the score.
    let mut rgb_sq_w: f64 = 0.0;
    let mut total_alpha: f64 = 0.0;
    for (i, p) in packs.iter().enumerate() {
        let lo = u32::from_le_bytes(p[0..4].try_into().unwrap());
        let alpha_bits = (lo >> 15) & 0x1F;
        if alpha_bits == 0 {
            continue;
        }
        let alpha = alpha_bits as f64 / 31.0;
        let idx = indices[i] as usize;
        let pal_pack = palette[idx];
        // Read 4-bit RGB from the tight pack, scale to 0..255 for parity
        // with `quantize`'s reporting convention.
        let pal_r = (pal_pack & 0xF) as f32 * (255.0 / 15.0);
        let pal_g = ((pal_pack >> 4) & 0xF) as f32 * (255.0 / 15.0);
        let pal_b = ((pal_pack >> 8) & 0xF) as f32 * (255.0 / 15.0);
        let pt = unpack_rgb8(*p);
        let dr = (pt[0] - pal_r) as f64;
        let dg = (pt[1] - pal_g) as f64;
        let db = (pt[2] - pal_b) as f64;
        rgb_sq_w += (dr * dr + dg * dg + db * db) * alpha;
        total_alpha += alpha;
    }
    let rgb_rmse = if total_alpha > 0.0 {
        (rgb_sq_w / total_alpha).sqrt() as f32
    } else {
        0.0
    };

    QuantResult10s {
        palette,
        indices,
        depth_palette,
        rgb_rmse,
    }
}

/// Greedy agglomerative merge. Picks the pair maximising `slots_saved / error`
/// each iteration and merges until total slot count ≤ `target`.
///
/// Implementation note: O(N²) per iteration without a heap. Acceptable for
/// typical N ≤ ~1500 unique (mat, norm) groups; will need a stale-tolerant
/// max-heap optimisation if profiling shows it's the bottleneck.
/// k-NN-window agglomerative merge. Avoids the O(N²) per-iteration scan by:
///
/// 1. Sorting groups by a packed (RGB-first) key — neighbours in 8-D
///    centroid space tend to land in a small window in the sort.
/// 2. Building an initial set of candidate edges from each group's
///    K-neighbour window in the sort.
/// 3. Heap-popping by score and merging. After each merge, re-pushing
///    candidates from the merged group's *current* sort position window.
///
/// Approximation: nearest-neighbour-in-RGB ordering misses pairs that are
/// close in (alpha, normal, rough, metal) but far in RGB. Acceptable for a
/// first pass; quality can be tightened later by widening K or extending
/// the sort key with normal/alpha dims.
fn agglomerative_merge_10s(groups: &mut Vec<Group10s>, target: u32) {
    /// K-window: each group considers this many neighbours on each side of
    /// its sort position as merge candidates. Bigger K → better merge
    /// quality, larger initial heap, more per-merge work.
    const K_WINDOW: usize = 32;

    let start = std::time::Instant::now();
    let n = groups.len();
    let mut total: u32 = groups.iter().map(|g| g.slot_count()).sum();
    let mut merges_done: u32 = 0;
    let mut last_log = start;
    let mut last_log_total = total;

    debug!(
        "agglomerative_merge_10s: starting, {n} groups, {total} total slots, target {target}, K-window {K_WINDOW}",
    );

    if total <= target || n < 2 {
        return;
    }

    // Sort indices by a packed centroid key (RGB-first then normal). Adjacent
    // entries are likely close in 8-D distance.
    let keys: Vec<u32> = groups.iter().map(centroid_sort_key).collect();
    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by_key(|&i| keys[i]);
    // position[group_id] = its index in `sorted`
    let mut position: Vec<u32> = vec![0; n];
    for (pos, &gid) in sorted.iter().enumerate() {
        position[gid] = pos as u32;
    }

    let mut alive: Vec<bool> = vec![true; n];
    let mut generations: Vec<u32> = vec![0; n];
    let mut heap: BinaryHeap<MergeEdge> = BinaryHeap::with_capacity(n * K_WINDOW);

    // Initial population: every (sort-adjacent) pair within K_WINDOW.
    for pos_i in 0..sorted.len() {
        let i = sorted[pos_i];
        for offset in 1..=K_WINDOW {
            let pos_j = pos_i + offset;
            if pos_j >= sorted.len() {
                break;
            }
            let j = sorted[pos_j];
            if let Some(edge) = build_edge(groups, &generations, i, j) {
                heap.push(edge);
            }
        }
    }

    debug!(
        "agglomerative_merge_10s: initial heap built with {} candidates in {:.1}s",
        heap.len(),
        start.elapsed().as_secs_f32(),
    );

    // Main loop: pop best, validate, merge.
    let mut stale_pops: u64 = 0;
    while total > target {
        let edge = match heap.pop() {
            Some(e) => e,
            None => {
                // Heap exhausted before reaching target. Re-sort surviving
                // groups by their *current* centroids — merges shift centroids
                // so the original sort no longer reflects current k-NN
                // structure. Then rebuild the K-window heap.
                //
                // For typical content this rescan only runs once or twice,
                // after the bulk of merges has dropped the group count enough
                // that re-sorting is cheap. If the alive count is small
                // enough we fall through to a full brute-force rescan.
                let alive_count = alive.iter().filter(|&&a| a).count();
                debug!(
                    "merge: heap exhausted at {total} slots (target {target}); resorting and rebuilding heap ({alive_count} alive groups)",
                );

                let mut alive_ids: Vec<u32> = (0..groups.len() as u32)
                    .filter(|&i| alive[i as usize])
                    .collect();
                alive_ids.sort_by_key(|&i| centroid_sort_key(&groups[i as usize]));

                // Update sorted + position so future merges use the new layout.
                sorted.clear();
                sorted.extend(alive_ids.iter().map(|&i| i as usize));
                for (pos, &gid) in sorted.iter().enumerate() {
                    position[gid] = pos as u32;
                }

                let mut pushed = 0u64;
                // For small alive counts, full O(N²) brute-force is cheap and
                // recovers any positive-save pairs the K-window would miss.
                let use_brute = alive_count <= 4096;
                let effective_k = if use_brute {
                    alive_count.saturating_sub(1)
                } else {
                    K_WINDOW
                };

                for pos_i in 0..sorted.len() {
                    let i = sorted[pos_i];
                    let max_off = if use_brute {
                        sorted.len() - pos_i - 1
                    } else {
                        effective_k.min(sorted.len() - pos_i - 1)
                    };
                    for offset in 1..=max_off {
                        let pos_j = pos_i + offset;
                        let j = sorted[pos_j];
                        if let Some(edge) = build_edge(groups, &generations, i, j) {
                            heap.push(edge);
                            pushed += 1;
                        }
                    }
                }

                debug!(
                    "merge: rebuild pushed {pushed} candidates ({}, K={effective_k})",
                    if use_brute { "brute-force" } else { "K-window" },
                );
                if pushed == 0 {
                    debug!(
                        "merge: no positive-saving candidates remain; stopping at {total} slots"
                    );
                    break;
                }
                continue;
            }
        };

        // Liveness + generation check.
        if !alive[edge.a as usize] || !alive[edge.b as usize] {
            stale_pops += 1;
            continue;
        }
        if generations[edge.a as usize] != edge.gen_a || generations[edge.b as usize] != edge.gen_b
        {
            stale_pops += 1;
            continue;
        }

        // Merge b into a (keeping a alive, marking b dead).
        let (a, b) = (edge.a as usize, edge.b as usize);
        let pre_a = groups[a].slot_count();
        let b_slots = groups[b].slot_count();
        // Move b out of groups slot — replace with a sentinel empty group so
        // indices stay valid. Cheap: take() swaps a default-constructed group
        // into b's slot.
        let g_b = std::mem::replace(&mut groups[b], Group10s::new(0));
        merge_into(&mut groups[a], g_b);
        let post_a = groups[a].slot_count();
        total = total - pre_a - b_slots + post_a;
        alive[b] = false;
        generations[a] = generations[a].wrapping_add(1);

        // Push new candidates from a's current sort window.
        let pos_a = position[a] as usize;
        let lo = pos_a.saturating_sub(K_WINDOW);
        let hi = (pos_a + K_WINDOW + 1).min(sorted.len());
        for pos_k in lo..hi {
            let k = sorted[pos_k];
            if k == a || !alive[k] {
                continue;
            }
            if let Some(new_edge) = build_edge(groups, &generations, a, k) {
                heap.push(new_edge);
            }
        }

        merges_done += 1;

        let now = std::time::Instant::now();
        let elapsed_since_log = now.duration_since(last_log).as_secs_f32();
        if elapsed_since_log >= 1.0 || merges_done.is_multiple_of(1024) {
            let total_elapsed = now.duration_since(start).as_secs_f32();
            let slots_reduced_window = last_log_total.saturating_sub(total);
            let merges_per_sec = (merges_done as f32) / total_elapsed;
            debug!(
                "merge progress: {merges_done} merges ({merges_per_sec:.0}/s), {total} slots (Δ-{slots_reduced_window} since last log), heap {} entries, stale-pops {stale_pops}, {total_elapsed:.1}s",
                heap.len(),
            );
            last_log = now;
            last_log_total = total;
        }
    }

    // Compact: remove dead groups (replaced by empty sentinels).
    groups.retain(|g| g.count > 0);

    debug!(
        "agglomerative_merge_10s: done after {merges_done} merges in {:.1}s — {} groups, {total} slots, {stale_pops} stale pops",
        start.elapsed().as_secs_f32(),
        groups.len(),
    );
}

/// Sort key packing RGB + normal channels into a u32. Adjacent values in
/// sort order are close in this 5-D subspace; the other 3 channels (alpha,
/// rough, metal) get handled by the K-window slack.
fn centroid_sort_key(g: &Group10s) -> u32 {
    let r = (g.centroid[0].clamp(0.0, 1.0) * 15.0 + 0.5) as u32 & 0xF;
    let g_ = (g.centroid[1].clamp(0.0, 1.0) * 15.0 + 0.5) as u32 & 0xF;
    let b = (g.centroid[2].clamp(0.0, 1.0) * 15.0 + 0.5) as u32 & 0xF;
    let nx = (g.centroid[6].clamp(0.0, 1.0) * 15.0 + 0.5) as u32 & 0xF;
    let ny = (g.centroid[7].clamp(0.0, 1.0) * 15.0 + 0.5) as u32 & 0xF;
    // RGB in the high bits so similar-coloured groups cluster first; nx/ny
    // tie-break.
    (r << 28) | (g_ << 24) | (b << 20) | (nx << 16) | (ny << 12)
}

/// Heap entry for the merge candidate queue. Ordered by score (max-heap):
/// `BinaryHeap` pops the largest score first.
#[derive(Clone, Copy)]
struct MergeEdge {
    /// Score bits: `f64::to_bits` of the score (slots_saved / error). For
    /// non-negative finite f64s the bit representation is monotonic w.r.t.
    /// the numeric value, so `cmp` on `u64` is the right ordering.
    score_bits: u64,
    a: u32,
    b: u32,
    gen_a: u32,
    gen_b: u32,
}

impl PartialEq for MergeEdge {
    fn eq(&self, other: &Self) -> bool {
        self.score_bits == other.score_bits
    }
}
impl Eq for MergeEdge {}
impl PartialOrd for MergeEdge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MergeEdge {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score_bits.cmp(&other.score_bits)
    }
}

/// Build a `MergeEdge` for groups `i` and `j` (caller ensures both alive).
/// Returns `None` if the merge isn't useful (non-positive savings).
fn build_edge(groups: &[Group10s], generations: &[u32], i: usize, j: usize) -> Option<MergeEdge> {
    let (a, b) = if i < j { (i, j) } else { (j, i) };
    let (slots_saved, error) = merge_cost_10s(&groups[a], &groups[b]);
    if slots_saved < 1 {
        return None;
    }
    // Quadratic error penalty. Linear (`slots_saved / error`) over-rewards
    // big slot saves regardless of colour similarity — the bitmap-OR
    // mechanism makes cross-tile merges save many slots even when the two
    // groups have very different (mat, norm). With error squared, low-error
    // merges dominate strongly: e.g. (slots=1, err=0.5) scores 4.0 while
    // (slots=64, err=10) scores 0.64.
    let score = slots_saved as f64 / (error * error + 1e-9);
    let score_bits = if score.is_finite() && score > 0.0 {
        score.to_bits()
    } else if score.is_infinite() {
        u64::MAX
    } else {
        0
    };
    Some(MergeEdge {
        score_bits,
        a: a as u32,
        b: b as u32,
        gen_a: generations[a],
        gen_b: generations[b],
    })
}

fn merge_cost_10s(a: &Group10s, b: &Group10s) -> (i32, f64) {
    // slots_saved = a.slots + b.slots - max_t popcount(a.bitmap[t] | b.bitmap[t])
    let a_slots = a.slot_count() as i32;
    let b_slots = b.slot_count() as i32;
    let new_slots = a
        .bitmap
        .iter()
        .zip(b.bitmap.iter())
        .map(|(x, y)| (x | y).count_ones())
        .max()
        .unwrap_or(0) as i32;
    let slots_saved = a_slots + b_slots - new_slots;

    // error = ward's linkage on the (mat, norm) centroids, weighted by visibility
    let mut dist_sq = 0.0f64;
    for d in 0..DIMS_10S {
        let delta = (a.centroid[d] - b.centroid[d]) as f64 * DIM_WEIGHTS_10S[d] as f64;
        dist_sq += delta * delta;
    }
    let factor = if a.weight + b.weight > 0.0 {
        (a.weight * b.weight) / (a.weight + b.weight)
    } else {
        0.0
    };
    let error_added = factor * dist_sq;

    (slots_saved, error_added)
}

fn merge_into(dst: &mut Group10s, src: Group10s) {
    // Bitmap OR
    for (d, s) in dst.bitmap.iter_mut().zip(src.bitmap.iter()) {
        *d |= *s;
    }
    // Pixel list concat
    dst.pixels.extend(src.pixels);
    // Centroid: weighted average
    let total_w = dst.weight + src.weight;
    if total_w > 0.0 {
        for d in 0..DIMS_10S {
            let combined =
                dst.centroid[d] as f64 * dst.weight + src.centroid[d] as f64 * src.weight;
            dst.centroid[d] = (combined / total_w) as f32;
        }
    }
    // Flags: keep heavier source's flags
    if src.weight > dst.weight {
        dst.flags = src.flags;
    }
    dst.weight = total_w;
    dst.count += src.count;
    dst.recompute_slots();
}

/// Encode the 32-bit tight palette entry for the idx10s format.
fn pack_palette_entry_10s(centroid: [f32; DIMS_10S], flags: u32) -> u32 {
    // RGB: 4 bits each (16 levels)
    let r = pack_bits_f(centroid[0], 4);
    let g = pack_bits_f(centroid[1], 4);
    let b = pack_bits_f(centroid[2], 4);
    // Alpha: 3 bits (8 levels)
    let a = pack_bits_f(centroid[3], 3);
    // Roughness, metallic: 2 bits each (4 levels)
    let rough = pack_bits_f(centroid[4], 2);
    let metal = pack_bits_f(centroid[5], 2);
    // Normal: 4 bits per axis (16 levels). Centroid stores normal as the
    // averaged unit normal then re-quantises here. Since groups key on the
    // exact stored octahedral bytes, the unmerged centroid already round-
    // trips; merged groups may shift slightly toward the merged neighbours,
    // which is the intended behaviour of the agglomerative merge.
    let nx = pack_bits_f(centroid[6], 4);
    let ny = pack_bits_f(centroid[7], 4);

    r | (g << 4)
        | (b << 8)
        | (a << 12)
        | (rough << 15)
        | (metal << 17)
        | ((flags & 0xF) << 19)
        | (nx << 23)
        | (ny << 27)
}
