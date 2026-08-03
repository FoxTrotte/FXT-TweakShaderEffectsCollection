/*
 * Blender compositor Vector Blur port for Tweak Shader / After Effects.
 *
 * Single-vector-layer version for After Effects' default EXR channel mapping:
 *   - Green = vector.x = previous X
 *   - Blue  = vector.y = previous Y
 *   - Alpha = vector.z = next X
 *   - Red   = vector.w = next Y
 *
 * Blur methods:
 *   - Blender Blur Emulation: Blender compositor-style tiled, depth-aware gather.
 *   - Custom Motion Blur: full-resolution soft velocity field followed by a
 *     cone/cylinder, depth-aware gather
 *
 */

#pragma utility_block(ShaderInputs)
layout(push_constant) uniform ShaderInputs {
    float time;
    float time_delta;
    float frame_rate;
    uint  frame_index;
    vec4  mouse;
    vec4  date;
    vec3  resolution;
    uint  pass_index;
};

/* Tile textures are fixed-size because Tweak Shader pass dimensions are compile-time values.
 * 256 tiles * 32 pixels supports images up to 8192 x 8192.
 */
#pragma pass(0, target="max_velocity_tex", width=256, height=256)
#pragma pass(1, target="dilated_velocity_tex", width=256, height=256)
/* Full-resolution passes used by custom Motion Blur. */
#pragma pass(2, target="custom_velocity_setup_tex")
#pragma pass(3, target="custom_velocity_horizontal_tex")
#pragma pass(4, target="custom_velocity_soft_tex")

#pragma sampler(name="linear_sampler", linear, clamp)
layout(set=0, binding=1) uniform sampler linear_sampler;
#pragma sampler(name="nearest_sampler", nearest, clamp)
layout(set=0, binding=2) uniform sampler nearest_sampler;

#pragma input(image, name="input_image")
layout(set=0, binding=3) uniform texture2D input_image;

/* After Effects default channel mapping for a single vector layer:
 *   G = vector.x = previous X
 *   B = vector.y = previous Y
 *   A = vector.z = next X
 *   R = vector.w = next Y
 */
#pragma input(image, name="velocity_tex")
layout(set=0, binding=4) uniform texture2D velocity_tex;

#pragma input(image, name="depth_tex")
layout(set=0, binding=5) uniform texture2D depth_tex;

/* Context-managed textures produced by the pre-passes. */
layout(set=0, binding=6) uniform texture2D max_velocity_tex;
layout(set=0, binding=7) uniform texture2D dilated_velocity_tex;
layout(set=0, binding=8) uniform texture2D custom_velocity_setup_tex;
layout(set=0, binding=9) uniform texture2D custom_velocity_horizontal_tex;
layout(set=0, binding=10) uniform texture2D custom_velocity_soft_tex;

#pragma input(int, name=samples, default=32, min=1, max=256)
#pragma input(float, name=shutter, default=0.5, min=0.0, max=4.0)
#pragma input(float, name=velocity_scale, default=1.0, min=-10.0, max=10.0)
#pragma input(float, name=max_blur_pixels, default=256.0, min=1.0, max=512.0)
/* Blender Blur Emulation only. */
#pragma input(float, name=depth_scale, default=100.0, min=0.0, max=10000.0)
#pragma input(bool, name=use_depth, default=true)
#pragma input(bool, name=swap_xy, default=false)
#pragma input(bool, name=flip_x, default=false)
#pragma input(bool, name=flip_y, default=false)
/* Blender Blur Emulation only. */
#pragma input(bool, name=tile_jitter, default=true)
/* custom Motion Blur controls. */
#pragma input(float, name=custom_velocity_spread, default=8.0, min=0.0, max=16.0)
#pragma input(float, name=custom_edge_softness, default=2.0, min=0.25, max=12.0)
#pragma input(float, name=custom_depth_scale, default=100.0, min=0.0, max=10000.0)
#pragma input(bool, name=custom_use_depth, default=true)
#pragma input(float, name=custom_motion_threshold, default=0.05, min=0.0, max=4.0)
#pragma input(int, name=blur_method, default=0, values=[0, 1], labels=["Blender Blur Emulation", "Custom Motion Blur"])
#pragma input(int, name=debug_mode, default=0, values=[0, 1, 2], labels=["Final Image", "Previous Frame Vector G X B Y", "Next Frame Vector A X R Y"])

layout(set=0, binding=11) uniform CustomInputs {
    int   samples;
    float shutter;
    float velocity_scale;
    float max_blur_pixels;
    float depth_scale;
    int   use_depth;
    int   swap_xy;
    int   flip_x;
    int   flip_y;
    int   tile_jitter;
    float custom_velocity_spread;
    float custom_edge_softness;
    float custom_depth_scale;
    int   custom_use_depth;
    float custom_motion_threshold;
    int   blur_method;
    int   debug_mode;
};

layout(location=0) out vec4 out_color;

const int   TILE_SIZE = 32;
const int   TILE_TARGET_SIZE = 256;
const int   MAX_SAMPLES = 256;
const int   MAX_DILATION_TILES = 16; /* 16 * 32 = 512 pixels. */
const int   MAX_CUSTOM_FILTER_RADIUS = 16;
const float SQRT_2 = 1.4142135623730951;

ivec2 image_size()
{
    return textureSize(sampler2D(input_image, nearest_sampler), 0);
}

ivec2 tile_count()
{
    ivec2 count = (image_size() + ivec2(TILE_SIZE - 1)) / TILE_SIZE;
    return min(count, ivec2(TILE_TARGET_SIZE));
}

bool outside(ivec2 p, ivec2 size)
{
    return any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, size));
}

vec2 clamp_motion_length(vec2 motion)
{
    float limit = min(max(max_blur_pixels, 1.0), float(MAX_DILATION_TILES * TILE_SIZE));
    float len = length(motion);
    return (len > limit && len > 0.0) ? motion * (limit / len) : motion;
}

vec2 orient_vector(vec2 v)
{
    if (swap_xy != 0) {
        v = v.yx;
    }
    if (flip_x != 0) {
        v.x = -v.x;
    }
    if (flip_y != 0) {
        v.y = -v.y;
    }
    return v * velocity_scale;
}

/* Return motion in pixel units, already shutter-scaled exactly as Blender's gather expects:
 * XY = previous motion; ZW = inverted next motion.
 *
 * After Effects default mapping into RGBA:
 *   packed.g = vector.x = previous X
 *   packed.b = vector.y = previous Y
 *   packed.a = vector.z = next X
 *   packed.r = vector.w = next Y
 */
vec4 decode_motion(vec4 packed)
{
    vec2 previous = vec2(packed.g, packed.b);
    vec2 next = vec2(packed.a, packed.r);

    previous = orient_vector(previous);
    next = orient_vector(next);

    /* Blender divides the UI shutter by two because it gathers one interval before and one after. */
    float shutter_half = max(shutter, 0.0) * 0.5;
    previous = clamp_motion_length(previous * shutter_half);
    next = clamp_motion_length(next * -shutter_half);
    return vec4(previous, next);
}

vec4 read_motion_texel(ivec2 p)
{
    ivec2 size = textureSize(sampler2D(velocity_tex, nearest_sampler), 0);
    p = clamp(p, ivec2(0), size - 1);
    vec4 packed = texelFetch(sampler2D(velocity_tex, nearest_sampler), p, 0);
    return decode_motion(packed);
}

vec4 read_motion_uv(vec2 uv)
{
    vec4 packed = texture(sampler2D(velocity_tex, linear_sampler), uv);
    return decode_motion(packed);
}

/* The decoded previous and next vectors represent the two temporal sides;
 * subtracting them gives the complete endpoint-to-endpoint motion span.
 */
vec2 full_motion_from_decoded(vec4 motion)
{
    return clamp_motion_length(motion.xy - motion.zw);
}

vec2 read_full_motion_uv(vec2 uv)
{
    return full_motion_from_decoded(read_motion_uv(uv));
}

vec2 read_full_motion_texel(ivec2 p)
{
    return full_motion_from_decoded(read_motion_texel(p));
}

vec2 longer_motion(vec2 a, vec2 b)
{
    return dot(a, a) > dot(b, b) ? a : b;
}

/* Blender's atomic payload compares ceil(length), then source tile X, then source tile Y. */
bool payload_is_better(vec2 candidate,
                       ivec2 candidate_tile,
                       vec2 current,
                       ivec2 current_tile)
{
    int candidate_length = int(min(ceil(length(candidate)), 16383.0));
    int current_length = int(min(ceil(length(current)), 16383.0));
    if (candidate_length != current_length) {
        return candidate_length > current_length;
    }
    if (candidate_tile.x != current_tile.x) {
        return candidate_tile.x > current_tile.x;
    }
    return candidate_tile.y > current_tile.y;
}

vec4 fetch_max_tile(ivec2 tile)
{
    ivec2 size = textureSize(sampler2D(max_velocity_tex, nearest_sampler), 0);
    tile = clamp(tile, ivec2(0), size - 1);
    return texelFetch(sampler2D(max_velocity_tex, nearest_sampler), tile, 0);
}

vec4 fetch_dilated_tile(ivec2 tile)
{
    ivec2 size = textureSize(sampler2D(dilated_velocity_tex, nearest_sampler), 0);
    tile = clamp(tile, ivec2(0), size - 1);
    return texelFetch(sampler2D(dilated_velocity_tex, nearest_sampler), tile, 0);
}

/* Pass 0: exact 32x32 reduction, independently for previous and next motion. */
void render_max_velocity_pass()
{
    ivec2 tile = ivec2(gl_FragCoord.xy);
    ivec2 tiles = tile_count();

    if (outside(tile, tiles) || any(greaterThanEqual(tile, ivec2(TILE_TARGET_SIZE)))) {
        out_color = vec4(0.0);
        return;
    }

    ivec2 size = image_size();
    ivec2 origin = tile * TILE_SIZE;
    vec2 max_previous = vec2(0.0);
    vec2 max_next = vec2(0.0);

    for (int y = 0; y < TILE_SIZE; ++y) {
        for (int x = 0; x < TILE_SIZE; ++x) {
            ivec2 p = origin + ivec2(x, y);
            if (outside(p, size)) {
                continue;
            }
            vec4 motion = read_motion_texel(p);
            max_previous = longer_motion(motion.xy, max_previous);
            max_next = longer_motion(motion.zw, max_next);
        }
    }

    out_color = vec4(max_previous, max_next);
}

bool tile_inside_motion_rect(ivec2 target, ivec2 source, vec2 motion, ivec2 tiles)
{
    ivec2 end_tile = source + ivec2(sign(motion) * ceil(abs(motion) / float(TILE_SIZE)));
    ivec2 min_tile = clamp(min(source, end_tile), ivec2(0), tiles - 1);
    ivec2 max_tile = clamp(max(source, end_tile), ivec2(0), tiles - 1);
    return all(greaterThanEqual(target, min_tile)) && all(lessThanEqual(target, max_tile));
}

bool tile_inside_motion_line(ivec2 target, ivec2 source, vec2 motion)
{
    float magnitude = length(motion);
    vec2 direction = magnitude != 0.0 ? motion / magnitude : motion;
    vec2 normal = vec2(-direction.y, direction.x);
    float distance_to_line = dot(normal, vec2(source - target));
    return abs(distance_to_line) < SQRT_2;
}

bool motion_reaches_tile(ivec2 target, ivec2 source, vec2 motion, ivec2 tiles)
{
    return tile_inside_motion_rect(target, source, motion, tiles) &&
           tile_inside_motion_line(target, source, motion);
}

/* Pass 1: inverse form of Blender's conservative scatter dilation.
 * A source tile contributes both temporal vectors when either of its temporal paths reaches
 * the target tile, matching Blender's CPU and GPU implementations.
 */
void render_dilation_pass()
{
    ivec2 target = ivec2(gl_FragCoord.xy);
    ivec2 tiles = tile_count();

    if (outside(target, tiles) || any(greaterThanEqual(target, ivec2(TILE_TARGET_SIZE)))) {
        out_color = vec4(0.0);
        return;
    }

    int radius = int(ceil(min(max(max_blur_pixels, 1.0),
                              float(MAX_DILATION_TILES * TILE_SIZE)) /
                          float(TILE_SIZE)));

    vec2 best_previous = vec2(0.0);
    vec2 best_next = vec2(0.0);
    ivec2 best_previous_tile = ivec2(0);
    ivec2 best_next_tile = ivec2(0);

    for (int oy = -MAX_DILATION_TILES; oy <= MAX_DILATION_TILES; ++oy) {
        if (abs(oy) > radius) {
            continue;
        }
        for (int ox = -MAX_DILATION_TILES; ox <= MAX_DILATION_TILES; ++ox) {
            if (abs(ox) > radius) {
                continue;
            }

            ivec2 source = target + ivec2(ox, oy);
            if (outside(source, tiles)) {
                continue;
            }

            vec4 source_motion = fetch_max_tile(source);
            bool reaches = motion_reaches_tile(target, source, source_motion.xy, tiles) ||
                           motion_reaches_tile(target, source, source_motion.zw, tiles);
            if (!reaches) {
                continue;
            }

            if (payload_is_better(source_motion.xy,
                                  source,
                                  best_previous,
                                  best_previous_tile)) {
                best_previous = source_motion.xy;
                best_previous_tile = source;
            }
            if (payload_is_better(source_motion.zw,
                                  source,
                                  best_next,
                                  best_next_tile)) {
                best_next = source_motion.zw;
                best_next_tile = source;
            }
        }
    }

    out_color = vec4(best_previous, best_next);
}

float interleaved_gradient_noise(ivec2 p)
{
    return fract(52.9829189 * fract(0.06711056 * float(p.x) + 0.00583715 * float(p.y)));
}

vec2 spread_compare(float center_motion_length,
                    float sample_motion_length,
                    float offset_length)
{
    return clamp(vec2(center_motion_length, sample_motion_length) - offset_length + 1.0,
                 0.0,
                 1.0);
}

vec2 depth_compare(float center_depth, float sample_depth)
{
    if (use_depth == 0) {
        return vec2(1.0);
    }
    vec2 scale = vec2(depth_scale, -depth_scale);
    return clamp(0.5 + scale * (sample_depth - center_depth), 0.0, 1.0);
}

float direction_compare(vec2 offset, vec2 sample_motion, float sample_motion_length)
{
    if (sample_motion_length < 0.5) {
        return 1.0;
    }
    return dot(offset, sample_motion) > 0.0 ? 1.0 : 0.0;
}

vec2 sample_weights(float center_depth,
                    float sample_depth,
                    float center_motion_length,
                    float sample_motion_length,
                    float offset_length)
{
    return depth_compare(center_depth, sample_depth) *
           spread_compare(center_motion_length, sample_motion_length, offset_length);
}

struct Accumulator {
    vec4 foreground;
    vec4 background;
    vec3 weight; /* X background, Y foreground, Z accepted direction count. */
};

float sample_depth_uv(vec2 uv, float center_depth)
{
    if (use_depth == 0) {
        return center_depth;
    }
    return texture(sampler2D(depth_tex, linear_sampler), uv).r;
}

void gather_sample(vec2 screen_uv,
                   float center_depth,
                   float center_motion_length,
                   vec2 offset,
                   float offset_length,
                   bool next_interval,
                   inout Accumulator accum)
{
    vec2 sample_uv = screen_uv - offset / vec2(image_size());
    vec4 sample_vectors = read_motion_uv(sample_uv);
    vec2 sample_motion = next_interval ? sample_vectors.zw : sample_vectors.xy;
    float sample_motion_length = length(sample_motion);
    float sample_depth = sample_depth_uv(sample_uv, center_depth);
    vec4 sample_color = texture(sampler2D(input_image, linear_sampler), sample_uv);

    vec2 direct = sample_weights(center_depth,
                                 sample_depth,
                                 center_motion_length,
                                 sample_motion_length,
                                 offset_length);
    float direction = direction_compare(offset, sample_motion, sample_motion_length);
    vec2 weights = direct * direction;

    accum.foreground += sample_color * weights.y;
    accum.background += sample_color * weights.x;
    accum.weight += vec3(weights, direction);
}

void gather_blur(vec2 screen_uv,
                 vec2 center_motion,
                 float center_depth,
                 vec2 max_motion,
                 float random_offset,
                 bool next_interval,
                 inout Accumulator accum)
{
    float center_length = length(center_motion);
    float max_length = length(max_motion);

    if (max_length < center_length) {
        max_motion = center_motion;
        max_length = center_length;
    }
    if (max_length < 0.5) {
        return;
    }

    int sample_count = clamp(samples, 1, MAX_SAMPLES);
    float increment = 1.0 / float(sample_count);
    float t = random_offset * increment;

    for (int i = 0; i < MAX_SAMPLES; ++i) {
        if (i >= sample_count) {
            break;
        }
        gather_sample(screen_uv,
                      center_depth,
                      center_length,
                      max_motion * t,
                      max_length * t,
                      next_interval,
                      accum);
        t += increment;
    }

    if (center_length < 0.5) {
        return;
    }

    t = random_offset * increment;
    for (int i = 0; i < MAX_SAMPLES; ++i) {
        if (i >= sample_count) {
            break;
        }
        gather_sample(screen_uv,
                      center_depth,
                      center_length,
                      center_motion * t,
                      center_length * t,
                      next_interval,
                      accum);
        t += increment;
    }
}

vec4 visualize_motion(vec2 motion)
{
    float limit = max(max_blur_pixels, 1.0);
    vec2 encoded = 0.5 + 0.5 * clamp(motion / limit, -1.0, 1.0);
    float magnitude = clamp(length(motion) / limit, 0.0, 1.0);
    return vec4(encoded, magnitude, 1.0);
}

float gaussian_weight(float x, float sigma)
{
    float safe_sigma = max(sigma, 0.5);
    return exp(-0.5 * (x * x) / (safe_sigma * safe_sigma));
}

/* Premultiplied full motion in RG, moving-object mask in B,
 * and normalization coverage in A. The mask is intentionally continuous.
 */
void render_custom_velocity_setup_pass()
{
    ivec2 texel = ivec2(gl_FragCoord.xy);
    ivec2 size = image_size();
    if (outside(texel, size)) {
        out_color = vec4(0.0);
        return;
    }

    vec2 uv = (vec2(texel) + 0.5) / vec2(size);
    vec2 motion = read_full_motion_uv(uv);
    float magnitude = length(motion);
    float threshold = max(custom_motion_threshold, 0.0001);
    float object_mask = smoothstep(threshold * 0.25, threshold, magnitude);

    out_color = vec4(motion * object_mask, object_mask, 1.0);
}

vec4 gaussian_velocity_filter_setup(vec2 uv, vec2 axis)
{
    ivec2 source_size = textureSize(sampler2D(custom_velocity_setup_tex, nearest_sampler), 0);
    vec2 inv_size = 1.0 / vec2(source_size);
    int radius = clamp(int(round(custom_velocity_spread)), 0, MAX_CUSTOM_FILTER_RADIUS);

    if (radius == 0) {
        return texture(sampler2D(custom_velocity_setup_tex, linear_sampler), uv);
    }

    float sigma = max(float(radius) * 0.45, 0.75);
    vec4 sum = vec4(0.0);
    float weight_sum = 0.0;

    for (int i = -MAX_CUSTOM_FILTER_RADIUS; i <= MAX_CUSTOM_FILTER_RADIUS; ++i) {
        if (abs(i) > radius) {
            continue;
        }
        float weight = gaussian_weight(float(i), sigma);
        vec2 sample_uv = clamp(uv + axis * float(i) * inv_size, vec2(0.0), vec2(1.0));
        sum += texture(sampler2D(custom_velocity_setup_tex, linear_sampler), sample_uv) * weight;
        weight_sum += weight;
    }

    return sum / max(weight_sum, 0.000001);
}

vec4 gaussian_velocity_filter_horizontal(vec2 uv, vec2 axis)
{
    ivec2 source_size = textureSize(sampler2D(custom_velocity_horizontal_tex, nearest_sampler), 0);
    vec2 inv_size = 1.0 / vec2(source_size);
    int radius = clamp(int(round(custom_velocity_spread)), 0, MAX_CUSTOM_FILTER_RADIUS);

    if (radius == 0) {
        return texture(sampler2D(custom_velocity_horizontal_tex, linear_sampler), uv);
    }

    float sigma = max(float(radius) * 0.45, 0.75);
    vec4 sum = vec4(0.0);
    float weight_sum = 0.0;

    for (int i = -MAX_CUSTOM_FILTER_RADIUS; i <= MAX_CUSTOM_FILTER_RADIUS; ++i) {
        if (abs(i) > radius) {
            continue;
        }
        float weight = gaussian_weight(float(i), sigma);
        vec2 sample_uv = clamp(uv + axis * float(i) * inv_size, vec2(0.0), vec2(1.0));
        sum += texture(sampler2D(custom_velocity_horizontal_tex, linear_sampler), sample_uv) * weight;
        weight_sum += weight;
    }

    return sum / max(weight_sum, 0.000001);
}

void render_custom_velocity_horizontal_pass()
{
    ivec2 texel = ivec2(gl_FragCoord.xy);
    ivec2 size = image_size();
    if (outside(texel, size)) {
        out_color = vec4(0.0);
        return;
    }
    vec2 uv = (vec2(texel) + 0.5) / vec2(size);
    out_color = gaussian_velocity_filter_setup(uv, vec2(1.0, 0.0));
}

void render_custom_velocity_vertical_pass()
{
    ivec2 texel = ivec2(gl_FragCoord.xy);
    ivec2 size = image_size();
    if (outside(texel, size)) {
        out_color = vec4(0.0);
        return;
    }
    vec2 uv = (vec2(texel) + 0.5) / vec2(size);
    out_color = gaussian_velocity_filter_horizontal(uv, vec2(0.0, 1.0));
}

float custom_soft_depth_compare(float center_depth, float sample_depth)
{
    if (custom_use_depth == 0) {
        return 1.0;
    }
    /* Blender's Z pass uses smaller positive values for nearer surfaces. */
    return clamp(1.0 - (sample_depth - center_depth) * custom_depth_scale, 0.0, 1.0);
}

float custom_cone_weight(vec2 x, vec2 y, vec2 velocity)
{
    float radius = length(velocity);
    float distance_xy = length(x - y);
    float softness = max(custom_edge_softness, 0.25);
    return 1.0 - smoothstep(max(radius - softness, 0.0), radius + softness, distance_xy);
}

float custom_cylinder_weight(vec2 x, vec2 y, vec2 velocity)
{
    float radius = length(velocity);
    float distance_xy = length(x - y);
    float softness = max(custom_edge_softness, 0.25);
    return 1.0 - smoothstep(max(radius - softness, 0.0), radius + softness, distance_xy);
}

float custom_sample_depth(vec2 uv, float fallback_depth)
{
    if (custom_use_depth == 0) {
        return fallback_depth;
    }
    return texture(sampler2D(depth_tex, linear_sampler), uv).r;
}

void render_custom_motion_blur_pass()
{
    ivec2 texel = ivec2(gl_FragCoord.xy);
    ivec2 size = image_size();
    if (outside(texel, size)) {
        out_color = vec4(0.0);
        return;
    }

    vec2 uv = (vec2(texel) + 0.5) / vec2(size);
    vec4 decoded_center = read_motion_uv(uv);

    if (debug_mode == 1) {
        out_color = visualize_motion(decoded_center.xy);
        return;
    }
    if (debug_mode == 2) {
        out_color = visualize_motion(decoded_center.zw);
        return;
    }

    vec4 center_color = texture(sampler2D(input_image, linear_sampler), uv);
    float center_depth = custom_sample_depth(uv, 0.0);
    vec2 center_position = uv * vec2(size);
    vec2 raw_center_velocity = full_motion_from_decoded(decoded_center);

    /* The Gaussian premultiplied field extends moving velocity into nearby background
     * while fading continuously, which is the key to soft outward silhouettes.
     */
    vec4 soft_velocity_sample = texture(sampler2D(custom_velocity_soft_tex, linear_sampler), uv);
    float soft_mask = clamp(soft_velocity_sample.b / max(soft_velocity_sample.a, 0.000001), 0.0, 1.0);
    vec2 normalized_soft_velocity = soft_velocity_sample.rg / max(soft_velocity_sample.b, 0.0001);
    vec2 combined_velocity = clamp_motion_length(normalized_soft_velocity * soft_mask);

    float combined_length = length(combined_velocity);
    if (combined_length < 0.01) {
        out_color = center_color;
        return;
    }

    int sample_count = clamp(samples, 2, MAX_SAMPLES);
    vec2 inv_size = 1.0 / vec2(size);

    vec4 color_accum = center_color * 0.0001;
    float color_weight = 0.0001;
    vec4 rejected_accum = center_color * 0.0001;
    float rejected_weight = 0.0001;

    /* Sample from the center outward in alternating directions,
     * but keep the full user-selected sample count and full-resolution inputs.
     */
    for (int e = 0; e < MAX_SAMPLES; ++e) {
        if (e >= sample_count) {
            break;
        }

        int pair_index = e / 2;
        int pair_count = (sample_count + 1) / 2;
        float sign_value = (e % 2 == 0) ? 1.0 : -1.0;
        float centered_step = (float(pair_index) + 1.0) / max(float(pair_count), 1.0);
        float delta = sign_value * centered_step * 0.5;

        vec2 sample_uv = clamp(uv + delta * combined_velocity * inv_size,
                               vec2(0.0), vec2(1.0));
        vec2 sample_position = sample_uv * vec2(size);
        vec4 sample_color = texture(sampler2D(input_image, linear_sampler), sample_uv);
        float sample_depth = custom_sample_depth(sample_uv, center_depth);
        vec2 sample_velocity = read_full_motion_uv(sample_uv);

        float sample_in_front = custom_soft_depth_compare(center_depth, sample_depth);
        float center_in_front = custom_soft_depth_compare(sample_depth, center_depth);

        float weight = 0.0;
        /* A moving sample in front can cover the current pixel. */
        weight += sample_in_front * custom_cone_weight(sample_position,
                                                       center_position,
                                                       sample_velocity);
        /* A moving current pixel can reveal or mix the background sample behind it. */
        weight += center_in_front * custom_cone_weight(center_position,
                                                       sample_position,
                                                       raw_center_velocity);
        /* Coherent overlapping motion receives extra support. */
        weight += 2.0 * custom_cylinder_weight(sample_position,
                                               center_position,
                                               sample_velocity) *
                        custom_cylinder_weight(center_position,
                                               sample_position,
                                               raw_center_velocity);

        weight = clamp(weight, 0.0, 1.0);
        color_accum += sample_color * weight;
        color_weight += weight;

        float rejected = 1.0 - weight;
        rejected_accum += sample_color * rejected;
        rejected_weight += rejected;
    }

    vec4 blurred_color = color_accum / max(color_weight, 0.0001);
    vec4 behind_color = rejected_accum / max(rejected_weight, 0.0001);
    float moving_fraction = clamp(color_weight / float(sample_count), 0.0, 1.0);
    vec4 reconstructed = mix(behind_color, blurred_color, moving_fraction);

    /* Full-resolution recombination with a continuous mask avoids the hard silhouette
     * produced by a strictly per-pixel directional blur.
     */
    float effect_mask = clamp(max(soft_mask, moving_fraction), 0.0, 1.0);
    out_color = mix(center_color, reconstructed, effect_mask);
}

void render_final_pass()
{
    ivec2 texel = ivec2(gl_FragCoord.xy);
    ivec2 size = image_size();
    if (outside(texel, size)) {
        out_color = vec4(0.0);
        return;
    }

    vec2 uv = (vec2(texel) + 0.5) / vec2(size);
    vec4 center_motion = read_motion_uv(uv);

    float center_depth = 0.0;
    if (use_depth != 0) {
        ivec2 depth_size = textureSize(sampler2D(depth_tex, nearest_sampler), 0);
        ivec2 depth_texel = clamp(texel, ivec2(0), depth_size - 1);
        center_depth = texelFetch(sampler2D(depth_tex, nearest_sampler), depth_texel, 0).r;
    }

    float random_value = interleaved_gradient_noise(texel);

    /* Match Blender's current implementation: one scalar diagonal tile offset, biased negative. */
    int jitter = tile_jitter != 0
                     ? int(random_value * 2.0 - float(TILE_SIZE) * 0.25)
                     : 0;
    ivec2 tile = (texel + ivec2(jitter)) / TILE_SIZE;
    tile = clamp(tile, ivec2(0), tile_count() - 1);

    vec4 tile_max = fetch_max_tile(tile);
    vec4 dilated_max = fetch_dilated_tile(tile);

    if (debug_mode == 1) {
        out_color = visualize_motion(center_motion.xy);
        return;
    }
    if (debug_mode == 2) {
        out_color = visualize_motion(center_motion.zw);
        return;
    }

    vec4 center_color = texture(sampler2D(input_image, linear_sampler), uv);
    Accumulator accum;
    accum.foreground = vec4(0.0);
    accum.background = vec4(0.0);
    accum.weight = vec3(0.0, 0.0, 1.0);

    gather_blur(uv,
                center_motion.xy,
                center_depth,
                dilated_max.xy,
                random_value,
                false,
                accum);
    gather_blur(uv,
                center_motion.zw,
                center_depth,
                dilated_max.zw,
                random_value,
                true,
                accum);

    int sample_count = clamp(samples, 1, MAX_SAMPLES);
    float fallback_weight = 1.0 / (50.0 * float(sample_count) * 4.0);
    accum.background += center_color * fallback_weight;
    accum.weight.x += fallback_weight;
    center_color = accum.background / accum.weight.x;

    accum.foreground += accum.background;
    accum.weight.y += accum.weight.x;

    float blend_factor = clamp(1.0 - accum.weight.y / accum.weight.z, 0.0, 1.0);
    out_color = accum.foreground / accum.weight.z + center_color * blend_factor;
}

void main()
{
    if (pass_index == 0u) {
        render_max_velocity_pass();
    }
    else if (pass_index == 1u) {
        render_dilation_pass();
    }
    else if (pass_index == 2u) {
        render_custom_velocity_setup_pass();
    }
    else if (pass_index == 3u) {
        render_custom_velocity_horizontal_pass();
    }
    else if (pass_index == 4u) {
        render_custom_velocity_vertical_pass();
    }
    else {
        if (blur_method == 0) {
            render_final_pass();
        }
        else {
            render_custom_motion_blur_pass();
        }
    }
}
