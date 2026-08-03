/*
 * AMD FidelityFX FSR1 + CAS for Tweak Shader / After Effects
 *
 * Pipeline:
 *   Pass 0: FSR1 EASU edge-adaptive spatial upscaling
 *   Pass 1: Optional FSR1 RCAS sharpening
 *   Main  : Optional standalone FidelityFX CAS sharpening
 *
 * zoom_percent selects a centered region of the source image and reconstructs
 * that smaller region to the full effect output using FSR1 EASU.
 *
 * Examples:
 *   100% = no enlargement
 *   150% = crop to 66.7% of the source width/height, then upscale
 *   200% = crop to 50% of the source width/height, then upscale
 *   400% = crop to 25% of the source width/height, then upscale
 *
 * After Effects keeps the layer/effect canvas size fixed; this shader enlarges
 * the image content inside that canvas rather than changing composition size.
 *
 * Based on the AMD FidelityFX FSR1 and FidelityFX CAS reference algorithms.
 * The agyild mpv GLSL port was also consulted as a GLSL integration reference.
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
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

/*
 * Both intermediate passes run at the final effect resolution.
 * EASU internally samples a virtual lower-resolution zoom crop.
 */
#pragma pass(0, target="easu_buffer")
#pragma pass(1, target="rcas_buffer")

#pragma sampler(name="linear_sampler", linear, clamp)
layout(set=0, binding=1) uniform sampler linear_sampler;

#pragma sampler(name="nearest_sampler", nearest, clamp)
layout(set=0, binding=2) uniform sampler nearest_sampler;

#pragma input(image, name="input_image")
layout(set=0, binding=3) uniform texture2D input_image;

layout(set=0, binding=4) uniform texture2D easu_buffer;
layout(set=0, binding=5) uniform texture2D rcas_buffer;

/*
 * zoom_percent:
 *   100 = no zoom
 *   150 = 1.5x enlargement
 *   200 = 2x enlargement
 *   400 = 4x enlargement
 */
#pragma input(float, name=zoom_percent, default=200.0, min=100.0, max=400.0)
#pragma input(bool, name=enable_rcas, default=true)
#pragma input(float, name=rcas_strength, default=0.50, min=0.00, max=1.00)
#pragma input(bool, name=rcas_denoise, default=true)
#pragma input(bool, name=enable_cas, default=true)
#pragma input(float, name=cas_strength, default=0.25, min=0.00, max=1.00)
#pragma input(bool, name=input_is_linear, default=false)
#pragma input(float, name=effect_mix, default=1.00, min=0.00, max=1.00)
#pragma input(bool, name=preserve_alpha, default=true)

layout(set=0, binding=6) uniform CustomInputs {
    float zoom_percent;
    int   enable_rcas;
    float rcas_strength;
    int   rcas_denoise;
    int   enable_cas;
    float cas_strength;
    int   input_is_linear;
    float effect_mix;
    int   preserve_alpha;
};

layout(location=0) out vec4 out_color;

const float EPSILON = 1e-6;
const float RCAS_LIMIT = 0.1875;

/* ------------------------------------------------------------------------- */
/* General texture and color helpers.                                         */
/* ------------------------------------------------------------------------- */

vec2 output_size()
{
    return max(resolution.xy, vec2(1.0));
}

ivec2 output_size_i()
{
    return ivec2(max(floor(output_size() + 0.5), vec2(1.0)));
}

ivec2 output_pixel()
{
    return clamp(
        ivec2(floor(gl_FragCoord.xy)),
        ivec2(0),
        output_size_i() - ivec2(1));
}

vec2 output_uv()
{
    return gl_FragCoord.xy / output_size();
}

float zoom_factor()
{
    return clamp(zoom_percent * 0.01, 1.0, 4.0);
}

ivec2 virtual_source_size_i()
{
    /*
     * A 200% zoom uses approximately half as many source samples along each
     * axis, then EASU reconstructs them to the full output resolution.
     */
    vec2 size_f = max(
        floor(output_size() / zoom_factor() + 0.5),
        vec2(1.0));

    return ivec2(size_f);
}

vec2 zoom_crop_uv(vec2 normalized_crop_uv)
{
    /*
     * Map 0..1 coordinates inside the selected crop back into the original
     * source image. The crop remains centered.
     */
    return vec2(0.5) +
           (normalized_crop_uv - vec2(0.5)) / zoom_factor();
}

vec3 clamp_algorithm_input(vec3 color)
{
    /*
     * The reference algorithms expect normalized non-negative input.
     * Clamping also prevents RCAS NaNs from negative values.
     */
    return clamp(color, vec3(0.0), vec3(1.0));
}

vec3 linear_to_perceptual(vec3 color)
{
    return pow(max(color, vec3(0.0)), vec3(1.0 / 2.2));
}

vec3 perceptual_to_linear(vec3 color)
{
    return pow(max(color, vec3(0.0)), vec3(2.2));
}

vec3 to_fsr_space(vec3 color)
{
    color = clamp_algorithm_input(color);
    return input_is_linear != 0 ? linear_to_perceptual(color) : color;
}

vec3 from_fsr_space(vec3 color)
{
    color = clamp_algorithm_input(color);
    return input_is_linear != 0 ? perceptual_to_linear(color) : color;
}

vec4 sample_input_linear(vec2 uv)
{
    return texture(
        sampler2D(input_image, linear_sampler),
        clamp(uv, vec2(0.0), vec2(1.0)));
}

vec4 sample_easu_nearest(vec2 uv)
{
    return texture(
        sampler2D(easu_buffer, nearest_sampler),
        clamp(uv, vec2(0.0), vec2(1.0)));
}

vec4 sample_rcas_nearest(vec2 uv)
{
    return texture(
        sampler2D(rcas_buffer, nearest_sampler),
        clamp(uv, vec2(0.0), vec2(1.0)));
}

vec3 load_virtual_source(ivec2 pixel)
{
    ivec2 source_size = virtual_source_size_i();
    ivec2 p = clamp(pixel, ivec2(0), source_size - ivec2(1));

    /*
     * Read one virtual low-resolution texel from the centered zoom crop.
     * EASU later reconstructs this crop to the full output dimensions.
     */
    vec2 crop_uv = (vec2(p) + vec2(0.5)) / vec2(source_size);
    vec2 source_uv = zoom_crop_uv(crop_uv);

    return to_fsr_space(sample_input_linear(source_uv).rgb);
}

vec4 load_easu_pixel(ivec2 pixel)
{
    ivec2 size_i = output_size_i();
    ivec2 p = clamp(pixel, ivec2(0), size_i - ivec2(1));
    vec2 uv = (vec2(p) + vec2(0.5)) / vec2(size_i);
    return sample_easu_nearest(uv);
}

vec4 load_rcas_pixel(ivec2 pixel)
{
    ivec2 size_i = output_size_i();
    ivec2 p = clamp(pixel, ivec2(0), size_i - ivec2(1));
    vec2 uv = (vec2(p) + vec2(0.5)) / vec2(size_i);
    return sample_rcas_nearest(uv);
}

float fsr_luma(vec3 color)
{
    /* AMD's inexpensive approximate luma multiplied by two. */
    return color.b * 0.5 + color.r * 0.5 + color.g;
}

/* ------------------------------------------------------------------------- */
/* FSR1 EASU.                                                                 */
/* ------------------------------------------------------------------------- */

void easu_set(
    inout vec2 direction,
    inout float edge_length,
    vec2 subpixel,
    vec2 bilinear_corner,
    float luma_a,
    float luma_b,
    float luma_c,
    float luma_d,
    float luma_e)
{
    float weight =
        mix(1.0 - subpixel.x, subpixel.x, bilinear_corner.x) *
        mix(1.0 - subpixel.y, subpixel.y, bilinear_corner.y);

    float dc = luma_d - luma_c;
    float cb = luma_c - luma_b;
    float len_x = max(abs(dc), abs(cb));
    float dir_x = luma_d - luma_b;

    len_x = clamp(abs(dir_x) / max(len_x, EPSILON), 0.0, 1.0);
    len_x *= len_x;

    float ec = luma_e - luma_c;
    float ca = luma_c - luma_a;
    float len_y = max(abs(ec), abs(ca));
    float dir_y = luma_e - luma_a;

    len_y = clamp(abs(dir_y) / max(len_y, EPSILON), 0.0, 1.0);
    len_y *= len_y;

    direction += vec2(dir_x, dir_y) * weight;
    edge_length += (len_x + len_y) * weight;
}

void easu_tap(
    inout vec3 accumulated_color,
    inout float accumulated_weight,
    vec2 offset,
    vec2 direction,
    vec2 anisotropic_length,
    float negative_lobe,
    float clipping_point,
    vec3 color)
{
    vec2 rotated;
    rotated.x = offset.x * direction.x + offset.y * direction.y;
    rotated.y = offset.x * -direction.y + offset.y * direction.x;
    rotated *= anisotropic_length;

    float distance_squared = dot(rotated, rotated);
    distance_squared = min(distance_squared, clipping_point);

    /*
     * Modified Lanczos-2 approximation from the FSR1 reference.
     */
    float base_window = (2.0 / 5.0) * distance_squared - 1.0;
    float lobe_window = negative_lobe * distance_squared - 1.0;

    base_window *= base_window;
    lobe_window *= lobe_window;
    base_window =
        (25.0 / 16.0) * base_window -
        ((25.0 / 16.0) - 1.0);

    float weight = base_window * lobe_window;
    accumulated_color += color * weight;
    accumulated_weight += weight;
}

vec3 fsr_easu(ivec2 output_position)
{
    vec2 source_size = vec2(virtual_source_size_i());
    vec2 destination_size = output_size();
    vec2 scale = source_size / destination_size;

    /*
     * Map the destination pixel center to source pixel space.
     */
    vec2 source_position =
        vec2(output_position) * scale +
        0.5 * scale -
        0.5;

    ivec2 base = ivec2(floor(source_position));
    vec2 subpixel = fract(source_position);

    /*
     * 12-tap source neighborhood:
     *
     *      b c
     *    e f g h
     *    i j k l
     *      n o
     */
    vec3 b = load_virtual_source(base + ivec2( 0, -1));
    vec3 c = load_virtual_source(base + ivec2( 1, -1));
    vec3 e = load_virtual_source(base + ivec2(-1,  0));
    vec3 f = load_virtual_source(base + ivec2( 0,  0));
    vec3 g = load_virtual_source(base + ivec2( 1,  0));
    vec3 h = load_virtual_source(base + ivec2( 2,  0));
    vec3 i = load_virtual_source(base + ivec2(-1,  1));
    vec3 j = load_virtual_source(base + ivec2( 0,  1));
    vec3 k = load_virtual_source(base + ivec2( 1,  1));
    vec3 l = load_virtual_source(base + ivec2( 2,  1));
    vec3 n = load_virtual_source(base + ivec2( 0,  2));
    vec3 o = load_virtual_source(base + ivec2( 1,  2));

    float b_l = fsr_luma(b);
    float c_l = fsr_luma(c);
    float e_l = fsr_luma(e);
    float f_l = fsr_luma(f);
    float g_l = fsr_luma(g);
    float h_l = fsr_luma(h);
    float i_l = fsr_luma(i);
    float j_l = fsr_luma(j);
    float k_l = fsr_luma(k);
    float l_l = fsr_luma(l);
    float n_l = fsr_luma(n);
    float o_l = fsr_luma(o);

    vec2 direction = vec2(0.0);
    float edge_length = 0.0;

    easu_set(
        direction, edge_length, subpixel, vec2(0.0, 0.0),
        b_l, e_l, f_l, g_l, j_l);

    easu_set(
        direction, edge_length, subpixel, vec2(1.0, 0.0),
        c_l, f_l, g_l, h_l, k_l);

    easu_set(
        direction, edge_length, subpixel, vec2(0.0, 1.0),
        f_l, i_l, j_l, k_l, n_l);

    easu_set(
        direction, edge_length, subpixel, vec2(1.0, 1.0),
        g_l, j_l, k_l, l_l, o_l);

    float direction_length_squared = dot(direction, direction);

    if (direction_length_squared < (1.0 / 32768.0)) {
        direction = vec2(1.0, 0.0);
    }
    else {
        direction *= inversesqrt(direction_length_squared);
    }

    edge_length *= 0.5;
    edge_length *= edge_length;

    float max_direction_component =
        max(abs(direction.x), abs(direction.y));

    float stretch =
        dot(direction, direction) /
        max(max_direction_component, EPSILON);

    vec2 anisotropic_length = vec2(
        1.0 + (stretch - 1.0) * edge_length,
        1.0 - 0.5 * edge_length);

    float negative_lobe =
        0.5 + ((0.25 - 0.04) - 0.5) * edge_length;

    float clipping_point =
        1.0 / max(negative_lobe, EPSILON);

    vec3 accumulated_color = vec3(0.0);
    float accumulated_weight = 0.0;

    easu_tap(
        accumulated_color, accumulated_weight,
        vec2( 0.0, -1.0) - subpixel,
        direction, anisotropic_length,
        negative_lobe, clipping_point, b);

    easu_tap(
        accumulated_color, accumulated_weight,
        vec2( 1.0, -1.0) - subpixel,
        direction, anisotropic_length,
        negative_lobe, clipping_point, c);

    easu_tap(
        accumulated_color, accumulated_weight,
        vec2(-1.0,  1.0) - subpixel,
        direction, anisotropic_length,
        negative_lobe, clipping_point, i);

    easu_tap(
        accumulated_color, accumulated_weight,
        vec2( 0.0,  1.0) - subpixel,
        direction, anisotropic_length,
        negative_lobe, clipping_point, j);

    easu_tap(
        accumulated_color, accumulated_weight,
        vec2( 0.0,  0.0) - subpixel,
        direction, anisotropic_length,
        negative_lobe, clipping_point, f);

    easu_tap(
        accumulated_color, accumulated_weight,
        vec2(-1.0,  0.0) - subpixel,
        direction, anisotropic_length,
        negative_lobe, clipping_point, e);

    easu_tap(
        accumulated_color, accumulated_weight,
        vec2( 1.0,  1.0) - subpixel,
        direction, anisotropic_length,
        negative_lobe, clipping_point, k);

    easu_tap(
        accumulated_color, accumulated_weight,
        vec2( 2.0,  1.0) - subpixel,
        direction, anisotropic_length,
        negative_lobe, clipping_point, l);

    easu_tap(
        accumulated_color, accumulated_weight,
        vec2( 2.0,  0.0) - subpixel,
        direction, anisotropic_length,
        negative_lobe, clipping_point, h);

    easu_tap(
        accumulated_color, accumulated_weight,
        vec2( 1.0,  0.0) - subpixel,
        direction, anisotropic_length,
        negative_lobe, clipping_point, g);

    easu_tap(
        accumulated_color, accumulated_weight,
        vec2( 1.0,  2.0) - subpixel,
        direction, anisotropic_length,
        negative_lobe, clipping_point, o);

    easu_tap(
        accumulated_color, accumulated_weight,
        vec2( 0.0,  2.0) - subpixel,
        direction, anisotropic_length,
        negative_lobe, clipping_point, n);

    vec3 reconstructed =
        accumulated_color / max(accumulated_weight, EPSILON);

    /*
     * Deringing clamp against the nearest 2x2 source neighborhood.
     */
    vec3 minimum_color = min(min(f, g), min(j, k));
    vec3 maximum_color = max(max(f, g), max(j, k));

    return clamp(reconstructed, minimum_color, maximum_color);
}

/* ------------------------------------------------------------------------- */
/* FSR1 RCAS.                                                                 */
/* ------------------------------------------------------------------------- */

float safe_divide(float numerator, float denominator)
{
    if (abs(denominator) < EPSILON) {
        denominator = denominator < 0.0 ? -EPSILON : EPSILON;
    }

    return numerator / denominator;
}

vec3 fsr_rcas(ivec2 pixel)
{
    vec3 b = clamp_algorithm_input(
        load_easu_pixel(pixel + ivec2( 0, -1)).rgb);

    vec3 d = clamp_algorithm_input(
        load_easu_pixel(pixel + ivec2(-1,  0)).rgb);

    vec3 e = clamp_algorithm_input(
        load_easu_pixel(pixel).rgb);

    vec3 f = clamp_algorithm_input(
        load_easu_pixel(pixel + ivec2( 1,  0)).rgb);

    vec3 h = clamp_algorithm_input(
        load_easu_pixel(pixel + ivec2( 0,  1)).rgb);

    float b_l = fsr_luma(b);
    float d_l = fsr_luma(d);
    float e_l = fsr_luma(e);
    float f_l = fsr_luma(f);
    float h_l = fsr_luma(h);

    float noise_factor =
        0.25 * b_l +
        0.25 * d_l +
        0.25 * f_l +
        0.25 * h_l -
        e_l;

    float luma_max = max(max(max(b_l, d_l), max(e_l, f_l)), h_l);
    float luma_min = min(min(min(b_l, d_l), min(e_l, f_l)), h_l);

    noise_factor =
        clamp(
            abs(noise_factor) /
            max(luma_max - luma_min, EPSILON),
            0.0,
            1.0);

    noise_factor = 1.0 - 0.5 * noise_factor;

    vec3 ring_min = min(min(b, d), min(f, h));
    vec3 ring_max = max(max(b, d), max(f, h));

    vec3 hit_min;
    hit_min.r = safe_divide(min(ring_min.r, e.r), 4.0 * ring_max.r);
    hit_min.g = safe_divide(min(ring_min.g, e.g), 4.0 * ring_max.g);
    hit_min.b = safe_divide(min(ring_min.b, e.b), 4.0 * ring_max.b);

    vec3 hit_max;
    hit_max.r = safe_divide(
        1.0 - max(ring_max.r, e.r),
        4.0 * ring_min.r - 4.0);

    hit_max.g = safe_divide(
        1.0 - max(ring_max.g, e.g),
        4.0 * ring_min.g - 4.0);

    hit_max.b = safe_divide(
        1.0 - max(ring_max.b, e.b),
        4.0 * ring_min.b - 4.0);

    vec3 channel_lobes = max(-hit_min, hit_max);
    float limiting_lobe =
        min(max(max(channel_lobes.r, channel_lobes.g), channel_lobes.b), 0.0);

    /*
     * FSR RCAS uses sharpness in stops:
     *   0 stops = maximum
     *   2 stops = one quarter strength
     *
     * This UI maps 0..1 intuitively from weaker to stronger.
     */
    float sharpness_stops =
        mix(2.0, 0.0, clamp(rcas_strength, 0.0, 1.0));

    float sharpness_linear = exp2(-sharpness_stops);

    float lobe =
        max(-RCAS_LIMIT, limiting_lobe) *
        sharpness_linear;

    if (rcas_denoise != 0) {
        lobe *= noise_factor;
    }

    float reciprocal_weight =
        1.0 / max(1.0 + 4.0 * lobe, EPSILON);

    vec3 result =
        (lobe * b +
         lobe * d +
         lobe * f +
         lobe * h +
         e) *
        reciprocal_weight;

    return clamp_algorithm_input(result);
}

/* ------------------------------------------------------------------------- */
/* Standalone FidelityFX CAS, sharpen-only high-quality path.                  */
/* ------------------------------------------------------------------------- */

vec3 load_cas_linear(ivec2 pixel)
{
    vec3 perceptual = clamp_algorithm_input(load_rcas_pixel(pixel).rgb);
    return perceptual_to_linear(perceptual);
}

vec3 fidelityfx_cas(ivec2 pixel)
{
    /*
     * CAS uses a 3x3 neighborhood. The high-quality path computes adaptive
     * weights per color channel rather than sharing only the green weight.
     */
    vec3 a = load_cas_linear(pixel + ivec2(-1, -1));
    vec3 b = load_cas_linear(pixel + ivec2( 0, -1));
    vec3 c = load_cas_linear(pixel + ivec2( 1, -1));
    vec3 d = load_cas_linear(pixel + ivec2(-1,  0));
    vec3 e = load_cas_linear(pixel);
    vec3 f = load_cas_linear(pixel + ivec2( 1,  0));
    vec3 g = load_cas_linear(pixel + ivec2(-1,  1));
    vec3 h = load_cas_linear(pixel + ivec2( 0,  1));
    vec3 i = load_cas_linear(pixel + ivec2( 1,  1));

    /*
     * Better-diagonals soft minimum and maximum from the reference path.
     * The values are doubled by the two overlapping neighborhoods.
     */
    vec3 cross_min = min(min(d, e), min(f, min(b, h)));
    vec3 full_min = min(cross_min, min(min(a, c), min(g, i)));
    vec3 soft_min = cross_min + full_min;

    vec3 cross_max = max(max(d, e), max(f, max(b, h)));
    vec3 full_max = max(cross_max, max(max(a, c), max(g, i)));
    vec3 soft_max = cross_max + full_max;

    vec3 amplitude =
        clamp(
            min(soft_min, vec3(2.0) - soft_max) /
            max(soft_max, vec3(EPSILON)),
            vec3(0.0),
            vec3(1.0));

    amplitude = sqrt(amplitude);

    /*
     * Official CAS mapping:
     *   strength 0 -> peak -1/8, lower ringing
     *   strength 1 -> peak -1/5, maximum sharpening
     */
    float peak =
        -1.0 /
        mix(8.0, 5.0, clamp(cas_strength, 0.0, 1.0));

    vec3 weight = amplitude * peak;
    vec3 reciprocal_weight =
        1.0 /
        max(vec3(1.0) + 4.0 * weight, vec3(EPSILON));

    vec3 result =
        (b * weight +
         d * weight +
         f * weight +
         h * weight +
         e) *
        reciprocal_weight;

    return clamp(result, vec3(0.0), vec3(1.0));
}

/* ------------------------------------------------------------------------- */
/* Passes and final composition.                                              */
/* ------------------------------------------------------------------------- */

void render_easu_pass()
{
    ivec2 pixel = output_pixel();
    vec2 uv = output_uv();
    vec2 source_uv = zoom_crop_uv(uv);

    vec3 color = fsr_easu(pixel);
    float alpha = sample_input_linear(source_uv).a;

    out_color = vec4(color, alpha);
}

void render_rcas_pass()
{
    ivec2 pixel = output_pixel();

    vec4 easu = load_easu_pixel(pixel);
    vec3 color =
        enable_rcas != 0
        ? fsr_rcas(pixel)
        : easu.rgb;

    out_color = vec4(color, easu.a);
}

void render_main_pass()
{
    ivec2 pixel = output_pixel();
    vec2 uv = output_uv();
    vec2 source_uv = zoom_crop_uv(uv);

    /*
     * The comparison image is zoomed with ordinary bilinear sampling so
     * effect_mix blends sharpening/reconstruction only, without changing zoom.
     */
    vec4 original = sample_input_linear(source_uv);
    vec4 processed_stage = load_rcas_pixel(pixel);

    vec3 processed_linear;

    if (enable_cas != 0) {
        processed_linear = fidelityfx_cas(pixel);
    }
    else {
        processed_linear =
            perceptual_to_linear(
                clamp_algorithm_input(processed_stage.rgb));
    }

    vec3 processed_output =
        input_is_linear != 0
        ? processed_linear
        : linear_to_perceptual(processed_linear);

    vec3 mixed_color =
        mix(
            original.rgb,
            processed_output,
            clamp(effect_mix, 0.0, 1.0));

    float alpha =
        preserve_alpha != 0
        ? original.a
        : processed_stage.a;

    out_color = vec4(mixed_color, alpha);
}

void main()
{
    if (pass_index == 0u) {
        render_easu_pass();
    }
    else if (pass_index == 1u) {
        render_rcas_pass();
    }
    else {
        render_main_pass();
    }
}
